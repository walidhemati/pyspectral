"""CuPy backend: ``cupyx.scipy.fft`` (cached cuFFT plans) + fused ``ElementwiseKernel``s.

This is what an experienced CuPy user writes: the nonlinear-term assembly, dealiasing,
projection and IF-RK4 update are fused into one ``ElementwiseKernel`` per stage and the
physical-space products into another, while FFTs go through the high-level SciPy-compatible
API (out-of-place, memory-pool allocations, plan cache).
"""

from __future__ import annotations

from typing import Any

import numpy as np

from pyspectral.backends._cuda_env import import_cupy
from pyspectral.backends._fused_common import if_scalars, if_table
from pyspectral.backends.base import BackendUnavailable, SpectralSolver
from pyspectral.config import Config
from pyspectral.grid import spectral_shape, wavenumbers
from pyspectral.ops import CupyOps

_RK = r"""
template <typename C, typename R>
__device__ C rk_stage(int stage, R dt, R E, R Eh, C N, C& u0, C& acc) {
    C u = u0, nxt;
    if (stage == 0) { acc = E * N; nxt = Eh * (u + (R)0.5 * dt * N); }
    else if (stage == 1) { acc = acc + (R)2 * Eh * N; nxt = Eh * u + (R)0.5 * dt * N; }
    else if (stage == 2) { acc = acc + (R)2 * Eh * N; nxt = E * u + dt * Eh * N; }
    else { nxt = E * u + dt / (R)6 * (acc + N); u0 = nxt; }
    return nxt;
}
"""

_IF = r"""
    R E = tab[ax] * tab[ay] * (k2 <= ka2 ? ea1 : (R)1);
    R Eh = tab[nh + ax] * tab[nh + ay] * (k2 <= ka2 ? eah : (R)1);
"""


def _kernels(cp: Any) -> dict[str, Any]:
    ek = cp.ElementwiseKernel
    k = {}
    k["phys2d"] = ek("R u, R v", "R a, R b",
                     "R uu = u, vv = v; a = (vv - uu) * (vv + uu); b = uu * vv;",
                     "pyspectral_phys2d")
    k["post2d"] = ek(
        "C A, C B, raw R tab, int32 n, int32 nh, int32 stage, R dt, R ea1, R eah, R eb1, R ebh, "
        "R klo2, R khi2, R ka2",
        "C u0, C acc, C wu, C wv",
        r"""
        int j = i % nh, ii = i / nh;
        int kxi = ii < n / 2 ? ii : ii - n;
        int ax = abs(kxi), ay = j;
        R kx = kxi, ky = j, k2 = kx * kx + ky * ky;
        bool keep = (3 * ax < n) && (3 * j < n);
        C N = keep ? (kx * ky) * A + (kx * kx - ky * ky) * B : C(0, 0);
        """ + _IF + r"""
        if (k2 >= klo2 && k2 <= khi2) { E *= eb1; Eh *= ebh; }
        C x = rk_stage<C, R>(stage, dt, E, Eh, N, u0, acc);
        R ik2 = (keep && k2 > 0) ? (R)1 / k2 : (R)0;
        C psi = ik2 * x;
        wu = C(-ky * psi.imag(), ky * psi.real());
        wv = C(kx * psi.imag(), -kx * psi.real());
        """,
        "pyspectral_post2d", preamble=_RK)
    k["phys3d_basd"] = ek(
        "R u, R v, R w", "R t11, R t22, R t12, R t13, R t23",
        r"""R uu = u, vv = v, ww = w; R q = (uu * uu + vv * vv + ww * ww) / (R)3;
        t11 = uu * uu - q; t22 = vv * vv - q; t12 = uu * vv; t13 = uu * ww; t23 = vv * ww;""",
        "pyspectral_phys3d_basd")
    k["phys3d_rot"] = ek(
        "R u, R v, R w, R ox, R oy, R oz", "R cx, R cy, R cz",
        "cx = v * oz - w * oy; cy = w * ox - u * oz; cz = u * oy - v * ox;",
        "pyspectral_phys3d_rot")
    head3 = r"""
        int kz = i % nh; long long t = i / nh; int jj = t % n; int ii = t / n;
        int kxi = ii < n / 2 ? ii : ii - n, kyi = jj < n / 2 ? jj : jj - n;
        int ax = abs(kxi), ay = abs(kyi);
        R kx = kxi, ky = kyi, kzr = kz, k2 = kx * kx + ky * ky + kzr * kzr;
        bool keep = (3 * ax < n) && (3 * ay < n) && (3 * kz < n);
        C N0(0, 0), N1(0, 0), N2(0, 0);
    """
    tail3 = r"""
        if (keep) {
            R ik2 = k2 > 0 ? (R)1 / k2 : (R)0;
            C d = ik2 * (kx * N0 + ky * N1 + kzr * N2);
            N0 = N0 - kx * d; N1 = N1 - ky * d; N2 = N2 - kzr * d;
        }
        """ + _IF + r"""
        E *= tab[kz] ; Eh *= tab[nh + kz];
        if (k2 >= klo2 && k2 <= khi2) { E *= eb1; Eh *= ebh; }
        R s = keep ? (R)1 : (R)0;
        w0 = s * rk_stage<C, R>(stage, dt, E, Eh, N0, u0, a0);
        w1 = s * rk_stage<C, R>(stage, dt, E, Eh, N1, u1, a1);
        w2 = s * rk_stage<C, R>(stage, dt, E, Eh, N2, u2, a2);
    """
    common_in = ("raw R tab, int32 n, int32 nh, int32 stage, R dt, R ea1, R eah, R eb1, R ebh, "
                 "R klo2, R khi2, R ka2")
    k["post3d_basd"] = ek(
        "C t11, C t22, C t12, C t13, C t23, " + common_in,
        "C u0, C u1, C u2, C a0, C a1, C a2, C w0, C w1, C w2",
        head3 + r"""
        if (keep) {
            C t33 = -(t11 + t22);
            C mi(0, -1);
            N0 = mi * (kx * t11 + ky * t12 + kzr * t13);
            N1 = mi * (kx * t12 + ky * t22 + kzr * t23);
            N2 = mi * (kx * t13 + ky * t23 + kzr * t33);
        }
        """ + tail3,
        "pyspectral_post3d_basd", preamble=_RK)
    k["post3d_rot"] = ek(
        "C c0, C c1, C c2, " + common_in,
        "C u0, C u1, C u2, C a0, C a1, C a2, C w0, C w1, C w2, C o0, C o1, C o2",
        head3 + r"""
        if (keep) { N0 = c0; N1 = c1; N2 = c2; }
        """ + tail3 + r"""
        C ii1(0, 1);
        o0 = ii1 * (ky * w2 - kzr * w1);
        o1 = ii1 * (kzr * w0 - kx * w2);
        o2 = ii1 * (kx * w1 - ky * w0);
        """,
        "pyspectral_post3d_rot", preamble=_RK)
    return k


class CupyElementwiseSolver(SpectralSolver):
    """CuPy ElementwiseKernel fusion + cupyx.scipy.fft."""

    backend = "cupy"
    variant = "elementwise"
    default_form = "basdevant"
    forms = ("basdevant", "rotational")

    def __init__(self, cfg: Config) -> None:
        super().__init__(cfg)
        if self.dim == 2 and self.form != "basdevant":
            raise BackendUnavailable("cupy: 2D supports the basdevant form only")
        try:
            cp = import_cupy()
            import cupyx.scipy.fft as cfft
        except Exception as e:  # noqa: BLE001
            raise BackendUnavailable(str(e)) from e
        self.cp, self.cfft = cp, cfft
        self.rdt = np.float64 if cfg.is_fp64 else np.float32
        self.cdt = np.complex128 if cfg.is_fp64 else np.complex64
        n, d = self.n, self.dim
        self.nh = n // 2 + 1
        spec = spectral_shape(n, d)
        self.rot = self.form == "rotational"
        shape = spec if d == 2 else (3,) + spec
        self.u0 = cp.zeros(shape, self.cdt)
        self.acc = cp.zeros_like(self.u0)
        self.W: Any = None
        self.tab = cp.zeros(2 * self.nh, self.rdt)
        self._tab_key: tuple[float, float] | None = None
        self.k = _kernels(cp)
        self.s = (n,) * d
        self.axes = tuple(range(-d, 0))
        self._ops = CupyOps()

    @classmethod
    def available(cls) -> tuple[bool, str]:
        try:
            cp = import_cupy()
            from pyspectral.backends._cuda_env import import_cufft

            import_cufft()
            if cp.cuda.runtime.getDeviceCount() < 1:
                return False, "no CUDA device"
        except Exception as e:  # noqa: BLE001
            return False, str(e)
        return True, ""

    @property
    def ops(self) -> Any:
        return self._ops

    def set_state(self, state_hat: np.ndarray) -> None:
        cp = self.cp
        a = self.check_state(state_hat)
        self.u0[...] = cp.asarray(a.astype(self.cdt))
        ks = [cp.asarray(k) for k in wavenumbers(self.n, self.dim)]
        mask = cp.asarray(self.mask_np())
        if self.dim == 2:
            kx, ky = ks
            k2 = kx * kx + ky * ky
            psi = cp.where(k2 > 0, self.u0 / cp.where(k2 > 0, k2, 1.0), 0) * mask
            self.W = cp.stack([1j * ky * psi, -1j * kx * psi]).astype(self.cdt)
        else:
            uh = self.u0 * mask
            fields = [uh[0], uh[1], uh[2]]
            if self.rot:
                k = ks
                fields += [1j * (k[1] * uh[2] - k[2] * uh[1]), 1j * (k[2] * uh[0] - k[0] * uh[2]),
                           1j * (k[0] * uh[1] - k[1] * uh[0])]
            self.W = cp.stack(fields).astype(self.cdt)

    def state_native(self) -> Any:
        return self.u0

    def _scalars(self, dt: float) -> tuple[Any, ...]:
        key = (dt, self.cfg.nu)
        if self._tab_key != key:
            self.tab.set(if_table(self.n, self.cfg.nu, dt).astype(self.rdt))
            self._tab_key = key
        ea1, eah, eb1, ebh = if_scalars(self.cfg.alpha, self.force_amp, dt)
        on = self.force_amp != 0.0
        r = self.rdt
        klo2 = r(self.cfg.force_klo**2 if on else 1.0)
        khi2 = r(self.cfg.force_khi**2 if on else 0.0)
        return (r(dt), r(ea1), r(eah), r(eb1), r(ebh), klo2, khi2, r(self.cfg.drag_k2))

    def _step(self, dt: float) -> None:
        sc = self._scalars(dt)
        n, nh = np.int32(self.n), np.int32(self.nh)
        k = self.k
        for stage in range(4):
            st = np.int32(stage)
            r = self.cfft.irfftn(self.W, s=self.s, axes=self.axes, overwrite_x=True)
            if self.dim == 2:
                p = self.cp.empty_like(r)
                k["phys2d"](r[0], r[1], p[0], p[1])
                q = self.cfft.rfftn(p, axes=self.axes)
                W = self.cp.empty_like(q)
                k["post2d"](q[0], q[1], self.tab, n, nh, st, *sc, self.u0, self.acc, W[0], W[1])
            elif self.rot:
                p = self.cp.empty((3,) + r.shape[1:], r.dtype)
                k["phys3d_rot"](r[0], r[1], r[2], r[3], r[4], r[5], p[0], p[1], p[2])
                q = self.cfft.rfftn(p, axes=self.axes)
                W = self.cp.empty((6,) + q.shape[1:], q.dtype)
                u, a = self.u0, self.acc
                k["post3d_rot"](q[0], q[1], q[2], self.tab, n, nh, st, *sc, u[0], u[1], u[2],
                                a[0], a[1], a[2], W[0], W[1], W[2], W[3], W[4], W[5])
            else:
                p = self.cp.empty((5,) + r.shape[1:], r.dtype)
                k["phys3d_basd"](r[0], r[1], r[2], p[0], p[1], p[2], p[3], p[4])
                q = self.cfft.rfftn(p, axes=self.axes)
                W = self.cp.empty((3,) + q.shape[1:], q.dtype)
                u, a = self.u0, self.acc
                k["post3d_basd"](q[0], q[1], q[2], q[3], q[4], self.tab, n, nh, st, *sc, u[0],
                                 u[1], u[2], a[0], a[1], a[2], W[0], W[1], W[2])
            self.W = W

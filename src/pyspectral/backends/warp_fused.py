"""NVIDIA Warp backend: fused pointwise kernels written in Warp, FFTs via torch.fft (cuFFT).

Warp has no large-FFT primitive, so the transforms are torch's; Warp kernels operate
zero-copy on the same buffers (``wp.from_torch``) on torch's current stream. Complex data are
addressed as interleaved real arrays. Launches are pre-recorded (``record_cmd=True``) to keep
Python overhead low. Supports 2D and the 3D Basdevant form.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from pyspectral.backends._cuda_env import torch_cuda
from pyspectral.backends._fused_common import if_scalars, if_table, n_work
from pyspectral.backends.base import BackendUnavailable, SpectralSolver
from pyspectral.config import Config, fft_split
from pyspectral.grid import spectral_shape
from pyspectral.ops import TorchOps

try:
    import warp as wp

    wp.config.quiet = True
    wp.config.enable_backward = False
    wp.config.log_level = 30
    _HAVE_WARP = True
except ImportError:  # pragma: no cover
    _HAVE_WARP = False

_KERNELS: dict[str, Any] = {}


def _build() -> dict[str, Any]:
    """Define the Warp kernels (generic over float32/float64) once."""
    if _KERNELS:
        return _KERNELS
    wp.config.quiet = True
    wp.init()

    @wp.func
    def rk_stage(stage: int, dt: Any, E: Any, Eh: Any, nr: Any, ni: Any, u0: Any, acc: Any,
                 j: int):
        ur = u0[2 * j]
        ui = u0[2 * j + 1]
        half = dt * type(dt)(0.5)
        two = type(dt)(2.0)
        xr = ur
        xi = ui
        if stage == 0:
            acc[2 * j] = E * nr
            acc[2 * j + 1] = E * ni
            xr = Eh * (ur + half * nr)
            xi = Eh * (ui + half * ni)
        elif stage == 1:
            acc[2 * j] = acc[2 * j] + two * Eh * nr
            acc[2 * j + 1] = acc[2 * j + 1] + two * Eh * ni
            xr = Eh * ur + half * nr
            xi = Eh * ui + half * ni
        elif stage == 2:
            acc[2 * j] = acc[2 * j] + two * Eh * nr
            acc[2 * j + 1] = acc[2 * j + 1] + two * Eh * ni
            xr = E * ur + dt * Eh * nr
            xi = E * ui + dt * Eh * ni
        else:
            s6 = dt / type(dt)(6.0)
            xr = E * ur + s6 * (acc[2 * j] + nr)
            xi = E * ui + s6 * (acc[2 * j + 1] + ni)
            u0[2 * j] = xr
            u0[2 * j + 1] = xi
        return xr, xi

    @wp.kernel
    def phys2d(R: wp.array(dtype=Any), fsr: int):
        p = wp.tid()
        u = R[p]
        v = R[fsr + p]
        R[p] = (v - u) * (v + u)
        R[fsr + p] = u * v

    @wp.kernel
    def post2d(Q: wp.array(dtype=Any), u0: wp.array(dtype=Any), acc: wp.array(dtype=Any),
               tab: wp.array(dtype=Any), P: wp.array(dtype=Any), n: int, nh: int, fsc: int,
               stage: int):
        idx = wp.tid()
        j = idx % nh
        i = idx // nh
        kxi = i
        if i >= n // 2:
            kxi = i - n
        ax = wp.abs(kxi)
        zero = P[0] * type(P[0])(0.0)
        kx = zero + type(zero)(kxi)
        ky = zero + type(zero)(j)
        k2 = kx * kx + ky * ky
        keep = (3 * ax < n) and (3 * j < n)
        nr = zero
        ni = zero
        if keep:
            c1 = kx * ky
            c2 = kx * kx - ky * ky
            nr = c1 * Q[2 * idx] + c2 * Q[2 * (fsc + idx)]
            ni = c1 * Q[2 * idx + 1] + c2 * Q[2 * (fsc + idx) + 1]
        E = tab[ax] * tab[j]
        Eh = tab[nh + ax] * tab[nh + j]
        if k2 <= P[7]:
            E = E * P[1]
            Eh = Eh * P[2]
        if k2 >= P[5] and k2 <= P[6]:
            E = E * P[3]
            Eh = Eh * P[4]
        x = rk_stage(stage, P[0], E, Eh, nr, ni, u0, acc, idx)
        ik2 = zero
        if keep and k2 > zero:
            ik2 = type(zero)(1.0) / k2
        pr = x[0] * ik2
        pi = x[1] * ik2
        Q[2 * idx] = -ky * pi
        Q[2 * idx + 1] = ky * pr
        Q[2 * (fsc + idx)] = kx * pi
        Q[2 * (fsc + idx) + 1] = -kx * pr

    @wp.kernel
    def phys3d(R: wp.array(dtype=Any), fsr: int):
        p = wp.tid()
        u = R[p]
        v = R[fsr + p]
        w = R[2 * fsr + p]
        q = (u * u + v * v + w * w) / type(u)(3.0)
        R[p] = u * u - q
        R[fsr + p] = v * v - q
        R[2 * fsr + p] = u * v
        R[3 * fsr + p] = u * w
        R[4 * fsr + p] = v * w

    @wp.kernel
    def post3d(Q: wp.array(dtype=Any), u0: wp.array(dtype=Any), acc: wp.array(dtype=Any),
               tab: wp.array(dtype=Any), P: wp.array(dtype=Any), n: int, nh: int, fsc: int,
               stage: int):
        idx = wp.tid()
        kz = idx % nh
        t = idx // nh
        jj = t % n
        ii = t // n
        kxi = ii
        if ii >= n // 2:
            kxi = ii - n
        kyi = jj
        if jj >= n // 2:
            kyi = jj - n
        ax = wp.abs(kxi)
        ay = wp.abs(kyi)
        zero = P[0] * type(P[0])(0.0)
        kx = zero + type(zero)(kxi)
        ky = zero + type(zero)(kyi)
        kzr = zero + type(zero)(kz)
        k2 = kx * kx + ky * ky + kzr * kzr
        keep = (3 * ax < n) and (3 * ay < n) and (3 * kz < n)
        n0r = zero
        n0i = zero
        n1r = zero
        n1i = zero
        n2r = zero
        n2i = zero
        if keep:
            t11r = Q[2 * idx]
            t11i = Q[2 * idx + 1]
            t22r = Q[2 * (fsc + idx)]
            t22i = Q[2 * (fsc + idx) + 1]
            t12r = Q[2 * (2 * fsc + idx)]
            t12i = Q[2 * (2 * fsc + idx) + 1]
            t13r = Q[2 * (3 * fsc + idx)]
            t13i = Q[2 * (3 * fsc + idx) + 1]
            t23r = Q[2 * (4 * fsc + idx)]
            t23i = Q[2 * (4 * fsc + idx) + 1]
            t33r = -(t11r + t22r)
            t33i = -(t11i + t22i)
            s0r = kx * t11r + ky * t12r + kzr * t13r
            s0i = kx * t11i + ky * t12i + kzr * t13i
            s1r = kx * t12r + ky * t22r + kzr * t23r
            s1i = kx * t12i + ky * t22i + kzr * t23i
            s2r = kx * t13r + ky * t23r + kzr * t33r
            s2i = kx * t13i + ky * t23i + kzr * t33i
            ik2 = zero
            if k2 > zero:
                ik2 = type(zero)(1.0) / k2
            # N = -i s, then Leray projection
            n0r = s0i
            n0i = -s0r
            n1r = s1i
            n1i = -s1r
            n2r = s2i
            n2i = -s2r
            dr = (kx * n0r + ky * n1r + kzr * n2r) * ik2
            di = (kx * n0i + ky * n1i + kzr * n2i) * ik2
            n0r = n0r - kx * dr
            n0i = n0i - kx * di
            n1r = n1r - ky * dr
            n1i = n1i - ky * di
            n2r = n2r - kzr * dr
            n2i = n2i - kzr * di
        E = tab[ax] * tab[ay] * tab[kz]
        Eh = tab[nh + ax] * tab[nh + ay] * tab[nh + kz]
        if k2 <= P[7]:
            E = E * P[1]
            Eh = Eh * P[2]
        if k2 >= P[5] and k2 <= P[6]:
            E = E * P[3]
            Eh = Eh * P[4]
        a = rk_stage(stage, P[0], E, Eh, n0r, n0i, u0, acc, idx)
        b = rk_stage(stage, P[0], E, Eh, n1r, n1i, u0, acc, fsc + idx)
        c = rk_stage(stage, P[0], E, Eh, n2r, n2i, u0, acc, 2 * fsc + idx)
        s = zero
        if keep:
            s = type(zero)(1.0)
        Q[2 * idx] = s * a[0]
        Q[2 * idx + 1] = s * a[1]
        Q[2 * (fsc + idx)] = s * b[0]
        Q[2 * (fsc + idx) + 1] = s * b[1]
        Q[2 * (2 * fsc + idx)] = s * c[0]
        Q[2 * (2 * fsc + idx) + 1] = s * c[1]

    _KERNELS.update(phys2d=phys2d, post2d=post2d, phys3d=phys3d, post3d=post3d)
    return _KERNELS


class WarpSolver(SpectralSolver):
    """Warp fused pointwise kernels + torch.fft out-of-place transforms."""

    backend = "warp"
    variant = "fused"
    default_form = "basdevant"
    forms = ("basdevant",)

    def __init__(self, cfg: Config) -> None:
        super().__init__(cfg)
        ok, why = self.available()
        if not ok:
            raise BackendUnavailable(why)
        torch = torch_cuda()
        self.torch = torch
        self.k = _build()
        dev = torch.device("cuda")
        self.rdt = torch.float64 if cfg.is_fp64 else torch.float32
        self.cdt = torch.complex128 if cfg.is_fp64 else torch.complex64
        n, d = self.n, self.dim
        self.nh = n // 2 + 1
        spec = spectral_shape(n, d)
        self.fsc = int(np.prod(spec))
        self.fsr = n**d
        self.n_inv, self.n_fwd = fft_split(d, self.form)
        nw = n_work(d, self.form)
        shape = spec if d == 2 else (3,) + spec
        self.u0 = torch.zeros(shape, dtype=self.cdt, device=dev)
        self.acc = torch.zeros_like(self.u0)
        self.Q = torch.zeros((nw,) + spec, dtype=self.cdt, device=dev)
        self.R = torch.zeros((nw,) + (n,) * d, dtype=self.rdt, device=dev)
        self.tab = torch.zeros(2 * self.nh, dtype=self.rdt, device=dev)
        self.P = torch.zeros(8, dtype=self.rdt, device=dev)
        self._key: Any = None
        self.dims = tuple(range(-d, 0))
        self.s = (n,) * d
        self._ops = TorchOps()
        flat = lambda t: wp.from_torch(t.view(-1) if not t.is_complex()  # noqa: E731
                                       else torch.view_as_real(t).view(-1))
        wq, wu, wa, wr = flat(self.Q), flat(self.u0), flat(self.acc), flat(self.R)
        wt, wpar = flat(self.tab), flat(self.P)
        kphys = self.k["phys2d" if d == 2 else "phys3d"]
        kpost = self.k["post2d" if d == 2 else "post3d"]
        self.wdev = wp.device_from_torch(torch.device("cuda", torch.cuda.current_device()))
        self.cmd_phys = wp.launch(kphys, dim=self.fsr, inputs=[wr, self.fsr], device=self.wdev,
                                  record_cmd=True)
        self.cmd_post = [wp.launch(kpost, dim=self.fsc, inputs=[wq, wu, wa, wt, wpar, n,
                                                                self.nh, self.fsc, s],
                                   device=self.wdev, record_cmd=True) for s in range(4)]

    @classmethod
    def available(cls) -> tuple[bool, str]:
        if not _HAVE_WARP:
            return False, "warp not installed"
        try:
            torch_cuda()
        except Exception as e:  # noqa: BLE001
            return False, str(e)
        return True, ""

    @property
    def ops(self) -> Any:
        return self._ops

    def set_state(self, state_hat: np.ndarray) -> None:
        from pyspectral.backends.triton_fused import TritonSolver

        a = self.check_state(state_hat)
        self.u0.copy_(self.torch.as_tensor(a).to(self.cdt))
        self.rot = False
        TritonSolver._pre(self)  # same NumPy-formula initialisation of the FFT inputs

    def state_native(self) -> Any:
        return self.u0

    def _params(self, dt: float) -> None:
        on = self.force_amp != 0.0
        key = (dt, self.force_amp, self.cfg.nu)
        if key == self._key:
            return
        self.tab.copy_(self.torch.as_tensor(if_table(self.n, self.cfg.nu, dt)))
        ea1, eah, eb1, ebh = if_scalars(self.cfg.alpha, self.force_amp, dt)
        klo2 = self.cfg.force_klo**2 if on else 1.0
        khi2 = self.cfg.force_khi**2 if on else 0.0
        self.P.copy_(self.torch.tensor([dt, ea1, eah, eb1, ebh, klo2, khi2,
                                                 self.cfg.drag_k2], dtype=self.rdt))
        self._key = key

    def _step(self, dt: float) -> None:
        self._params(dt)
        fft = self.torch.fft
        stream = wp.stream_from_torch(self.torch.cuda.current_stream())
        with wp.ScopedStream(stream):
            for stage in range(4):
                fft.irfftn(self.Q[: self.n_inv], s=self.s, dim=self.dims,
                           out=self.R[: self.n_inv])
                self.cmd_phys.launch()
                fft.rfftn(self.R[: self.n_fwd], dim=self.dims, out=self.Q[: self.n_fwd])
                self.cmd_post[stage].launch()

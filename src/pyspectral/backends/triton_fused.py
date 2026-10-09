"""Triton backend: fused pointwise kernels between cuFFT calls made through ``torch.fft``.

Same schedule as the CUDA C backend (phys kernel + fused spectral post kernel per stage),
but the FFTs are out-of-place ``torch.fft.irfftn/rfftn(..., out=...)`` into preallocated
buffers. The forward-FFT output buffer doubles as the next inverse-FFT input (the post
kernel rewrites it element-wise in place).
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
    import triton
    import triton.language as tl

    _HAVE_TRITON = True
except ImportError:  # pragma: no cover - depends on environment
    _HAVE_TRITON = False

_BLOCK = 512

if _HAVE_TRITON:

    @triton.jit
    def _cload(ptr, idx, m):
        offs = idx[:, None] * 2 + tl.arange(0, 2)[None, :]
        x = tl.load(ptr + offs, mask=m[:, None], other=0.0)
        return tl.split(x)

    @triton.jit
    def _cstore(ptr, idx, m, re, im):
        offs = idx[:, None] * 2 + tl.arange(0, 2)[None, :]
        tl.store(ptr + offs, tl.join(re, im), mask=m[:, None])

    @triton.jit
    def _rk(stage: tl.constexpr, dt, E, Eh, nr, ni, ur, ui, u0p, accp, idx, m):
        """IF-RK4 stage update for one component; returns next stage state."""
        if stage == 0:
            _cstore(accp, idx, m, E * nr, E * ni)
            xr = Eh * (ur + 0.5 * dt * nr)
            xi = Eh * (ui + 0.5 * dt * ni)
        elif stage == 1:
            ar, ai = _cload(accp, idx, m)
            _cstore(accp, idx, m, ar + 2.0 * Eh * nr, ai + 2.0 * Eh * ni)
            xr = Eh * ur + 0.5 * dt * nr
            xi = Eh * ui + 0.5 * dt * ni
        elif stage == 2:
            ar, ai = _cload(accp, idx, m)
            _cstore(accp, idx, m, ar + 2.0 * Eh * nr, ai + 2.0 * Eh * ni)
            xr = E * ur + dt * Eh * nr
            xi = E * ui + dt * Eh * ni
        else:
            ar, ai = _cload(accp, idx, m)
            xr = E * ur + dt / 6.0 * (ar + nr)
            xi = E * ui + dt / 6.0 * (ai + ni)
            _cstore(u0p, idx, m, xr, xi)
        return xr, xi

    @triton.jit
    def _phys2d(R, fsr, BLOCK: tl.constexpr):
        p = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        m = p < fsr
        u = tl.load(R + p, mask=m)
        v = tl.load(R + fsr + p, mask=m)
        tl.store(R + p, (v - u) * (v + u), mask=m)
        tl.store(R + fsr + p, u * v, mask=m)

    @triton.jit
    def _post2d(Q, u0, acc, tab, n, nh, fsc, P,
                stage: tl.constexpr, BLOCK: tl.constexpr):
        idx = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        m = idx < fsc
        # scalars come from a device array so fp64 runs keep fp64 parameters
        dt = tl.load(P)
        ea1 = tl.load(P + 1)
        eah = tl.load(P + 2)
        eb1 = tl.load(P + 3)
        ebh = tl.load(P + 4)
        klo2 = tl.load(P + 5)
        khi2 = tl.load(P + 6)
        ka2 = tl.load(P + 7)
        j = idx % nh
        i = idx // nh
        kxi = tl.where(i < n // 2, i, i - n)
        akx = tl.abs(kxi)
        keep = (3 * akx < n) & (3 * j < n) & m
        kx = kxi.to(tab.dtype.element_ty)
        ky = j.to(tab.dtype.element_ty)
        k2 = kx * kx + ky * ky
        ar, ai = _cload(Q, idx, keep)
        br, bi = _cload(Q + 2 * fsc, idx, keep)
        c1 = kx * ky
        c2 = kx * kx - ky * ky
        nr = c1 * ar + c2 * br
        ni = c1 * ai + c2 * bi
        drag = k2 <= ka2
        E = (tl.load(tab + akx, mask=m, other=0.0) * tl.load(tab + j, mask=m, other=0.0)
             * tl.where(drag, ea1, 1.0))
        Eh = (tl.load(tab + nh + akx, mask=m, other=0.0) * tl.load(tab + nh + j, mask=m, other=0.0)
              * tl.where(drag, eah, 1.0))
        band = (k2 >= klo2) & (k2 <= khi2)
        E = tl.where(band, E * eb1, E)
        Eh = tl.where(band, Eh * ebh, Eh)
        ur, ui = _cload(u0, idx, m)
        xr, xi = _rk(stage, dt, E, Eh, nr, ni, ur, ui, u0, acc, idx, m)
        ik2 = tl.where(keep & (k2 > 0), 1.0 / tl.where(k2 > 0, k2, 1.0), 0.0)
        pr = xr * ik2
        pi = xi * ik2
        # u_hat = i ky psi, v_hat = -i kx psi
        _cstore(Q, idx, m, -ky * pi, ky * pr)
        _cstore(Q + 2 * fsc, idx, m, kx * pi, -kx * pr)

    @triton.jit
    def _phys3d_basd(R, fsr, BLOCK: tl.constexpr):
        p = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        m = p < fsr
        u = tl.load(R + p, mask=m)
        v = tl.load(R + fsr + p, mask=m)
        w = tl.load(R + 2 * fsr + p, mask=m)
        q = (u * u + v * v + w * w) / 3.0
        tl.store(R + p, u * u - q, mask=m)
        tl.store(R + fsr + p, v * v - q, mask=m)
        tl.store(R + 2 * fsr + p, u * v, mask=m)
        tl.store(R + 3 * fsr + p, u * w, mask=m)
        tl.store(R + 4 * fsr + p, v * w, mask=m)

    @triton.jit
    def _phys3d_rot(R, fsr, BLOCK: tl.constexpr):
        p = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        m = p < fsr
        u = tl.load(R + p, mask=m)
        v = tl.load(R + fsr + p, mask=m)
        w = tl.load(R + 2 * fsr + p, mask=m)
        ox = tl.load(R + 3 * fsr + p, mask=m)
        oy = tl.load(R + 4 * fsr + p, mask=m)
        oz = tl.load(R + 5 * fsr + p, mask=m)
        tl.store(R + p, v * oz - w * oy, mask=m)
        tl.store(R + fsr + p, w * ox - u * oz, mask=m)
        tl.store(R + 2 * fsr + p, u * oy - v * ox, mask=m)

    @triton.jit
    def _post3d(Q, u0, acc, tab, n, nh, fsc, P,
                rot: tl.constexpr, stage: tl.constexpr, BLOCK: tl.constexpr):
        idx = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        m = idx < fsc
        # scalars come from a device array so fp64 runs keep fp64 parameters
        dt = tl.load(P)
        ea1 = tl.load(P + 1)
        eah = tl.load(P + 2)
        eb1 = tl.load(P + 3)
        ebh = tl.load(P + 4)
        klo2 = tl.load(P + 5)
        khi2 = tl.load(P + 6)
        ka2 = tl.load(P + 7)
        kz = idx % nh
        t = idx // nh
        jj = t % n
        ii = t // n
        kxi = tl.where(ii < n // 2, ii, ii - n)
        kyi = tl.where(jj < n // 2, jj, jj - n)
        akx = tl.abs(kxi)
        aky = tl.abs(kyi)
        keep = (3 * akx < n) & (3 * aky < n) & (3 * kz < n) & m
        kx = kxi.to(tab.dtype.element_ty)
        ky = kyi.to(tab.dtype.element_ty)
        kzr = kz.to(tab.dtype.element_ty)
        k2 = kx * kx + ky * ky + kzr * kzr
        if rot:
            n0r, n0i = _cload(Q, idx, keep)
            n1r, n1i = _cload(Q + 2 * fsc, idx, keep)
            n2r, n2i = _cload(Q + 4 * fsc, idx, keep)
        else:
            t11r, t11i = _cload(Q, idx, keep)
            t22r, t22i = _cload(Q + 2 * fsc, idx, keep)
            t12r, t12i = _cload(Q + 4 * fsc, idx, keep)
            t13r, t13i = _cload(Q + 6 * fsc, idx, keep)
            t23r, t23i = _cload(Q + 8 * fsc, idx, keep)
            t33r = -(t11r + t22r)
            t33i = -(t11i + t22i)
            # N = -i (k . T): -i (a + ib) = b - ia
            s0r = kx * t11r + ky * t12r + kzr * t13r
            s0i = kx * t11i + ky * t12i + kzr * t13i
            s1r = kx * t12r + ky * t22r + kzr * t23r
            s1i = kx * t12i + ky * t22i + kzr * t23i
            s2r = kx * t13r + ky * t23r + kzr * t33r
            s2i = kx * t13i + ky * t23i + kzr * t33i
            n0r, n0i = s0i, -s0r
            n1r, n1i = s1i, -s1r
            n2r, n2i = s2i, -s2r
        ik2 = tl.where(k2 > 0, 1.0 / tl.where(k2 > 0, k2, 1.0), 0.0)
        dr = (kx * n0r + ky * n1r + kzr * n2r) * ik2
        di = (kx * n0i + ky * n1i + kzr * n2i) * ik2
        n0r = n0r - kx * dr
        n0i = n0i - kx * di
        n1r = n1r - ky * dr
        n1i = n1i - ky * di
        n2r = n2r - kzr * dr
        n2i = n2i - kzr * di
        drag = k2 <= ka2
        E = (tl.load(tab + akx, mask=m, other=0.0) * tl.load(tab + aky, mask=m, other=0.0)
             * tl.load(tab + kz, mask=m, other=0.0) * tl.where(drag, ea1, 1.0))
        Eh = (tl.load(tab + nh + akx, mask=m, other=0.0)
              * tl.load(tab + nh + aky, mask=m, other=0.0)
              * tl.load(tab + nh + kz, mask=m, other=0.0) * tl.where(drag, eah, 1.0))
        band = (k2 >= klo2) & (k2 <= khi2)
        E = tl.where(band, E * eb1, E)
        Eh = tl.where(band, Eh * ebh, Eh)
        ur, ui = _cload(u0, idx, m)
        ar, ai = _rk(stage, dt, E, Eh, n0r, n0i, ur, ui, u0, acc, idx, m)
        ur, ui = _cload(u0 + 2 * fsc, idx, m)
        br, bi = _rk(stage, dt, E, Eh, n1r, n1i, ur, ui, u0 + 2 * fsc, acc + 2 * fsc, idx, m)
        ur, ui = _cload(u0 + 4 * fsc, idx, m)
        cr, ci = _rk(stage, dt, E, Eh, n2r, n2i, ur, ui, u0 + 4 * fsc, acc + 4 * fsc, idx, m)
        z = tl.zeros_like(ar)
        ar = tl.where(keep, ar, z)
        ai = tl.where(keep, ai, z)
        br = tl.where(keep, br, z)
        bi = tl.where(keep, bi, z)
        cr = tl.where(keep, cr, z)
        ci = tl.where(keep, ci, z)
        _cstore(Q, idx, m, ar, ai)
        _cstore(Q + 2 * fsc, idx, m, br, bi)
        _cstore(Q + 4 * fsc, idx, m, cr, ci)
        if rot:
            # omega = i k x u: i (a + ib) = -b + ia
            wxr = ky * cr - kzr * br
            wxi = ky * ci - kzr * bi
            wyr = kzr * ar - kx * cr
            wyi = kzr * ai - kx * ci
            wzr = kx * br - ky * ar
            wzi = kx * bi - ky * ai
            _cstore(Q + 6 * fsc, idx, m, -wxi, wxr)
            _cstore(Q + 8 * fsc, idx, m, -wyi, wyr)
            _cstore(Q + 10 * fsc, idx, m, -wzi, wzr)


class TritonSolver(SpectralSolver):
    """Triton fused pointwise kernels + torch.fft (cuFFT) out-of-place transforms."""

    backend = "triton"
    variant = "fused"
    default_form = "basdevant"
    forms = ("basdevant", "rotational")

    def __init__(self, cfg: Config) -> None:
        super().__init__(cfg)
        if self.dim == 2 and self.form != "basdevant":
            raise BackendUnavailable("triton: 2D supports the basdevant form only")
        ok, why = self.available()
        if not ok:
            raise BackendUnavailable(why)
        torch = torch_cuda()
        self.torch = torch
        dev = torch.device("cuda")
        self.rdt = torch.float64 if cfg.is_fp64 else torch.float32
        self.cdt = torch.complex128 if cfg.is_fp64 else torch.complex64
        n, d = self.n, self.dim
        self.nh = n // 2 + 1
        spec = spectral_shape(n, d)
        self.fsc = int(np.prod(spec))
        self.fsr = n**d
        self.rot = self.form == "rotational"
        self.n_inv, self.n_fwd = fft_split(d, self.form)
        nw = n_work(d, self.form)
        shape = spec if d == 2 else (3,) + spec
        self.u0 = torch.zeros(shape, dtype=self.cdt, device=dev)
        self.acc = torch.zeros_like(self.u0)
        self.Q = torch.zeros((nw,) + spec, dtype=self.cdt, device=dev)
        self.R = torch.zeros((nw,) + (n,) * d, dtype=self.rdt, device=dev)
        self.tab = torch.zeros(2 * self.nh, dtype=self.rdt, device=dev)
        self._tab_key: tuple[float, float] | None = None
        self.P = torch.zeros(8, dtype=self.rdt, device=dev)
        self._pkey: tuple[float, ...] | None = None
        self.dims = tuple(range(-d, 0))
        self.s = (n,) * d
        self._ops = TorchOps()

    @classmethod
    def available(cls) -> tuple[bool, str]:
        if not _HAVE_TRITON:
            return False, "triton not installed"
        try:
            torch_cuda()
        except Exception as e:  # noqa: BLE001
            return False, str(e)
        return True, ""

    @property
    def ops(self) -> Any:
        return self._ops

    def set_state(self, state_hat: np.ndarray) -> None:
        a = self.check_state(state_hat)
        self.u0.copy_(self.torch.as_tensor(a).to(self.cdt))
        self._pre()

    def _pre(self) -> None:
        """Write the first-stage FFT inputs from the state (NumPy formulas, rarely called)."""
        torch = self.torch
        mask = torch.as_tensor(self.mask_np(), device=self.u0.device)
        from pyspectral.grid import wavenumbers

        ks = [torch.as_tensor(k, dtype=self.rdt, device=self.u0.device)
              for k in wavenumbers(self.n, self.dim)]
        if self.dim == 2:
            kx, ky = ks
            k2 = kx * kx + ky * ky
            psi = torch.where(k2 > 0, self.u0 / torch.where(k2 > 0, k2, 1.0), 0) * mask
            self.Q[0] = 1j * ky * psi
            self.Q[1] = -1j * kx * psi
            return
        uh = self.u0 * mask
        self.Q[:3] = uh
        if self.rot:
            k = ks
            self.Q[3] = 1j * (k[1] * uh[2] - k[2] * uh[1])
            self.Q[4] = 1j * (k[2] * uh[0] - k[0] * uh[2])
            self.Q[5] = 1j * (k[0] * uh[1] - k[1] * uh[0])

    def state_native(self) -> Any:
        return self.u0

    def _scalars(self, dt: float) -> tuple[float, ...]:
        key = (dt, self.cfg.nu)
        if self._tab_key != key:
            self.tab.copy_(self.torch.as_tensor(if_table(self.n, self.cfg.nu, dt)))
            self._tab_key = key
        ea1, eah, eb1, ebh = if_scalars(self.cfg.alpha, self.force_amp, dt)
        on = self.force_amp != 0.0
        klo2 = float(self.cfg.force_klo**2) if on else 1.0
        khi2 = float(self.cfg.force_khi**2) if on else 0.0
        vals = (dt, ea1, eah, eb1, ebh, klo2, khi2, self.cfg.drag_k2)
        if vals != self._pkey:
            self.P.copy_(self.torch.tensor(vals, dtype=self.rdt))
            self._pkey = vals
        return (self.P,)

    def _step(self, dt: float) -> None:
        fft = self.torch.fft
        vr = self.torch.view_as_real
        sc = self._scalars(dt)
        gr = (triton.cdiv(self.fsr, _BLOCK),)
        gc = (triton.cdiv(self.fsc, _BLOCK),)
        Qr, u0r, accr = vr(self.Q), vr(self.u0), vr(self.acc)
        for stage in range(4):
            fft.irfftn(self.Q[: self.n_inv], s=self.s, dim=self.dims, out=self.R[: self.n_inv])
            if self.dim == 2:
                _phys2d[gr](self.R, self.fsr, BLOCK=_BLOCK)
            elif self.rot:
                _phys3d_rot[gr](self.R, self.fsr, BLOCK=_BLOCK)
            else:
                _phys3d_basd[gr](self.R, self.fsr, BLOCK=_BLOCK)
            fft.rfftn(self.R[: self.n_fwd], dim=self.dims, out=self.Q[: self.n_fwd])
            if self.dim == 2:
                _post2d[gc](Qr, u0r, accr, self.tab, self.n, self.nh, self.fsc, *sc,
                            stage=stage, BLOCK=_BLOCK)
            else:
                _post3d[gc](Qr, u0r, accr, self.tab, self.n, self.nh, self.fsc, *sc,
                            rot=self.rot, stage=stage, BLOCK=_BLOCK)

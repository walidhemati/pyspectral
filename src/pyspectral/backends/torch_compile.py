"""``torch.compile`` backend: Inductor-generated fused kernels around cuFFT calls.

The pointwise parts (physical-space products; nonlinear-term assembly + dealiasing +
projection + IF-RK4 update + next-stage FFT inputs) are plain PyTorch functions on real views
of the complex buffers, compiled by TorchInductor into Triton kernels. FFTs stay eager
``torch.fft`` calls. Stage coefficients and dt are passed as a device tensor so CFL-adaptive
time steps never trigger recompilation.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from pyspectral.backends._cuda_env import torch_cuda
from pyspectral.backends.base import BackendUnavailable, SpectralSolver
from pyspectral.config import Config, fft_split
from pyspectral.grid import wavenumbers
from pyspectral.ops import TorchOps

# coefficient vector layout: [dt, nu, alpha, amp, klo2, khi2,
#                             aE, aEh, a0, pE, pEh, qEh, q1, r]
_IDX = {k: i for i, k in enumerate(
    ["dt", "nu", "alpha", "amp", "klo2", "khi2", "aE", "aEh", "a0", "pE", "pEh", "qEh", "q1", "r"])}


def _stage_coeffs(stage: int, dt: float) -> dict[str, float]:
    """IF-RK4 (Lawson) stage update written as linear combinations (see module docstring)."""
    if stage == 0:
        return {"aE": 1.0, "pEh": 1.0, "qEh": 0.5 * dt}
    if stage == 1:
        return {"aEh": 2.0, "a0": 1.0, "pEh": 1.0, "q1": 0.5 * dt}
    if stage == 2:
        return {"aEh": 2.0, "a0": 1.0, "pE": 1.0, "qEh": dt}
    return {"pE": 1.0, "q1": dt / 6.0, "r": dt / 6.0}


def _factors(c: Any, k2: Any) -> tuple[Any, Any]:
    L = -(c[1] * k2 + c[2]) + c[3] * ((k2 >= c[4]) & (k2 <= c[5])).to(k2.dtype)
    return (L * c[0]).exp(), (L * (0.5 * c[0])).exp()


def _rk(c: Any, E: Any, Eh: Any, nr: Any, ni: Any, ur: Any, ui: Any, ar: Any, ai: Any
        ) -> tuple[Any, Any, Any, Any]:
    ca = c[6] * E + c[7] * Eh
    acc_r = c[8] * ar + ca * nr
    acc_i = c[8] * ai + ca * ni
    cu = c[9] * E + c[10] * Eh
    cn = c[11] * Eh + c[12]
    xr = cu * ur + cn * nr + c[13] * ar
    xi = cu * ui + cn * ni + c[13] * ai
    return xr, xi, acc_r, acc_i


def _phys2d(R: Any) -> Any:
    import torch

    u, v = R[0], R[1]
    return torch.stack([(v - u) * (v + u), u * v])


def _phys3d_basd(R: Any) -> Any:
    import torch

    u, v, w = R[0], R[1], R[2]
    q = (u * u + v * v + w * w) / 3.0
    return torch.stack([u * u - q, v * v - q, u * v, u * w, v * w])


def _phys3d_rot(R: Any) -> Any:
    import torch

    u, v, w, ox, oy, oz = R[0], R[1], R[2], R[3], R[4], R[5]
    return torch.stack([v * oz - w * oy, w * ox - u * oz, u * oy - v * ox])


def _post2d(Q: Any, u0: Any, acc: Any, kx: Any, ky: Any, keep: Any, c: Any, final: bool
            ) -> tuple[Any, Any]:
    import torch

    k2 = kx * kx + ky * ky
    ar, ai, br, bi = Q[0, ..., 0], Q[0, ..., 1], Q[1, ..., 0], Q[1, ..., 1]
    c1, c2 = kx * ky * keep, (kx * kx - ky * ky) * keep
    nr, ni = c1 * ar + c2 * br, c1 * ai + c2 * bi
    E, Eh = _factors(c, k2)
    xr, xi, accr, acci = _rk(c, E, Eh, nr, ni, u0[..., 0], u0[..., 1], acc[..., 0], acc[..., 1])
    ik2 = torch.where(k2 > 0, 1.0 / torch.where(k2 > 0, k2, 1.0), 0.0) * keep
    pr, pi = xr * ik2, xi * ik2
    W = torch.stack([torch.stack([-ky * pi, ky * pr], -1), torch.stack([kx * pi, -kx * pr], -1)])
    if final:
        return W, torch.stack([xr, xi], -1)
    return W, torch.stack([accr, acci], -1)


def _post3d(Q: Any, u0: Any, acc: Any, kx: Any, ky: Any, kz: Any, keep: Any, c: Any,
            rot: bool, final: bool) -> tuple[Any, Any]:
    import torch

    k2 = kx * kx + ky * ky + kz * kz
    if rot:
        n = [(Q[i, ..., 0] * keep, Q[i, ..., 1] * keep) for i in range(3)]
    else:
        t = [(Q[i, ..., 0] * keep, Q[i, ..., 1] * keep) for i in range(5)]
        t33 = (-(t[0][0] + t[1][0]), -(t[0][1] + t[1][1]))
        T = [[t[0], t[2], t[3]], [t[2], t[1], t[4]], [t[3], t[4], t33]]
        n = []
        for i in range(3):
            sr = kx * T[i][0][0] + ky * T[i][1][0] + kz * T[i][2][0]
            si = kx * T[i][0][1] + ky * T[i][1][1] + kz * T[i][2][1]
            n.append((si, -sr))  # -i * s
    ik2 = torch.where(k2 > 0, 1.0 / torch.where(k2 > 0, k2, 1.0), 0.0)
    dr = (kx * n[0][0] + ky * n[1][0] + kz * n[2][0]) * ik2
    di = (kx * n[0][1] + ky * n[1][1] + kz * n[2][1]) * ik2
    kk = (kx, ky, kz)
    n = [(n[i][0] - kk[i] * dr, n[i][1] - kk[i] * di) for i in range(3)]
    E, Eh = _factors(c, k2)
    xs, accs = [], []
    for i in range(3):
        xr, xi, accr, acci = _rk(c, E, Eh, n[i][0], n[i][1], u0[i, ..., 0], u0[i, ..., 1],
                                 acc[i, ..., 0], acc[i, ..., 1])
        xs.append((xr * keep, xi * keep))
        accs.append(torch.stack([accr, acci], -1))
    fields = [torch.stack(x, -1) for x in xs]
    if rot:
        a, b, cc = xs
        wx = (ky * cc[0] - kz * b[0], ky * cc[1] - kz * b[1])
        wy = (kz * a[0] - kx * cc[0], kz * a[1] - kx * cc[1])
        wz = (kx * b[0] - ky * a[0], kx * b[1] - ky * a[1])
        fields += [torch.stack([-w[1], w[0]], -1) for w in (wx, wy, wz)]
    if final:
        return torch.stack(fields), torch.stack(fields[:3])
    return torch.stack(fields), torch.stack(accs)


class TorchCompileSolver(SpectralSolver):
    """TorchInductor-fused pointwise kernels + eager torch.fft."""

    backend = "torch-compile"
    variant = "inductor"
    default_form = "basdevant"
    forms = ("basdevant", "rotational")

    def __init__(self, cfg: Config, mode: str | None = None) -> None:
        super().__init__(cfg)
        if self.dim == 2 and self.form != "basdevant":
            raise BackendUnavailable("torch-compile: 2D supports the basdevant form only")
        try:
            torch = torch_cuda()
        except ImportError as e:
            raise BackendUnavailable(str(e)) from e
        self.torch = torch
        dev = torch.device("cuda")
        self.rdt = torch.float64 if cfg.is_fp64 else torch.float32
        self.cdt = torch.complex128 if cfg.is_fp64 else torch.complex64
        n, d = self.n, self.dim
        self.k = [torch.as_tensor(k, dtype=self.rdt, device=dev) for k in wavenumbers(n, d)]
        self.keep = torch.as_tensor(self.mask_np(), dtype=self.rdt, device=dev)
        self.rot = self.form == "rotational"
        self.n_inv, self.n_fwd = fft_split(d, self.form)
        self.s = (n,) * d
        self.dims = tuple(range(-d, 0))
        shape = self.state_shape()
        self.u0 = torch.zeros(shape + (2,), dtype=self.rdt, device=dev)
        self.acc = torch.zeros_like(self.u0)
        self.W: Any = None
        self.c = torch.zeros(len(_IDX), dtype=self.rdt, device=dev)
        self._ckey: Any = None
        kw = {"dynamic": False}
        if mode:
            kw["mode"] = mode
        if d == 2:
            self._phys = torch.compile(_phys2d, **kw)
            self._post = torch.compile(_post2d, **kw)
        else:
            self._phys = torch.compile(_phys3d_rot if self.rot else _phys3d_basd, **kw)
            self._post = torch.compile(_post3d, **kw)
        self._ops = TorchOps()

    @classmethod
    def available(cls) -> tuple[bool, str]:
        try:
            torch_cuda()
            import triton  # noqa: F401  (Inductor's GPU code generator)
        except Exception as e:  # noqa: BLE001
            return False, str(e)
        return True, ""

    @property
    def ops(self) -> Any:
        return self._ops

    def set_state(self, state_hat: np.ndarray) -> None:
        torch = self.torch
        a = self.check_state(state_hat)
        u = torch.as_tensor(a).to(self.cdt).to(self.u0.device)
        self.u0 = torch.view_as_real(u).contiguous()
        k = self.k
        if self.dim == 2:
            kx, ky = k
            k2 = kx * kx + ky * ky
            psi = torch.where(k2 > 0, u / torch.where(k2 > 0, k2, 1.0), 0)
            self.W = torch.stack([1j * ky * psi, -1j * kx * psi])
        else:
            fields = [u[0], u[1], u[2]]
            if self.rot:
                fields += [1j * (k[1] * u[2] - k[2] * u[1]), 1j * (k[2] * u[0] - k[0] * u[2]),
                           1j * (k[0] * u[1] - k[1] * u[0])]
            self.W = torch.stack(fields)

    def state_native(self) -> Any:
        return self.torch.view_as_complex(self.u0)

    def _coeffs(self, stage: int, dt: float) -> Any:
        cfg = self.cfg
        on = self.force_amp != 0.0
        key = (stage, dt, self.force_amp)
        if key != self._ckey:
            v = np.zeros(len(_IDX))
            v[_IDX["dt"]], v[_IDX["nu"]], v[_IDX["alpha"]] = dt, cfg.nu, cfg.alpha
            v[_IDX["amp"]] = self.force_amp
            v[_IDX["klo2"]] = cfg.force_klo**2 if on else 1.0
            v[_IDX["khi2"]] = cfg.force_khi**2 if on else 0.0
            for name, val in _stage_coeffs(stage, dt).items():
                v[_IDX[name]] = val
            self.c.copy_(self.torch.as_tensor(v))
            self._ckey = key
        return self.c

    def _step(self, dt: float) -> None:
        torch = self.torch
        fft = torch.fft
        for stage in range(4):
            c = self._coeffs(stage, dt)
            R = fft.irfftn(self.W, s=self.s, dim=self.dims)
            P = self._phys(R)
            Q = torch.view_as_real(fft.rfftn(P, dim=self.dims))
            fin = stage == 3
            if self.dim == 2:
                W, out = self._post(Q, self.u0, self.acc, *self.k, self.keep, c, fin)
            else:
                W, out = self._post(Q, self.u0, self.acc, *self.k, self.keep, c, self.rot, fin)
            self.W = torch.view_as_complex(W)
            if fin:
                self.u0 = out
            else:
                self.acc = out

"""PyTorch eager backend: a line-by-line port of the NumPy reference onto torch.fft (cuFFT).

Variants form the first rungs of the optimisation ladder:

* ``c2c-textbook``  complex-to-complex FFTs on the full spectrum, textbook FFT count
* ``r2c-textbook``  real-to-complex FFTs, textbook FFT count (2D advective 5, 3D rotational 9)
* ``r2c-minfft``    real-to-complex FFTs, minimal FFT count (Basdevant: 2D 4, 3D 8)
"""

from __future__ import annotations

from typing import Any

import numpy as np

from pyspectral.backends._cuda_env import torch_cuda
from pyspectral.backends.base import BackendUnavailable, SpectralSolver
from pyspectral.config import Config
from pyspectral.ops import TorchOps


class TorchEagerSolver(SpectralSolver):
    """Eager PyTorch solver; every arithmetic op is its own kernel launch."""

    backend = "torch-eager"
    variant = "r2c-minfft"
    phase_shift = True

    def __init__(self, cfg: Config, real: bool = True, seed: int = 1234) -> None:
        super().__init__(cfg)
        try:
            torch = torch_cuda()
        except ImportError as e:
            raise BackendUnavailable(str(e)) from e
        self.torch = torch
        self.real = real
        self.dev = torch.device("cuda")
        self.rdt = torch.float64 if cfg.is_fp64 else torch.float32
        self.cdt = torch.complex128 if cfg.is_fp64 else torch.complex64
        n, d = self.n, self.dim
        k = np.fft.fftfreq(n, 1.0 / n)
        klast = np.arange(n // 2 + 1) if real else k
        vecs = [k] * (d - 1) + [klast]
        self.k = []
        for i, v in enumerate(vecs):
            shape = [1] * d
            shape[i] = len(v)
            self.k.append(torch.tensor(v, dtype=self.rdt, device=self.dev).reshape(shape))
        self.k2 = sum(kk * kk for kk in self.k)
        self.inv_k2 = torch.where(self.k2 > 0, 1.0 / torch.where(self.k2 > 0, self.k2, 1.0), 0.0)
        mask = self.k2 < 2.0 * (n / 3.0) ** 2 if cfg.dealias == "phase-shift" else None
        if mask is None:
            mask = torch.ones_like(self.k2, dtype=torch.bool)
            for kk in self.k:
                mask = mask & (3.0 * kk.abs() < n)
        self.mask = mask.to(self.rdt)
        self.s = (n,) * d
        self.dims = tuple(range(-d, 0))
        self.rng = np.random.default_rng(seed)
        self.shift: np.ndarray | None = None
        shape = self.state_shape() if real else ((n,) * d if d == 2 else (3,) + (n,) * d)
        self.state = torch.zeros(shape, dtype=self.cdt, device=self.dev)
        self._ops = TorchOps()

    @classmethod
    def available(cls) -> tuple[bool, str]:
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
        if not self.real:
            axes = tuple(range(-self.dim, 0))
            a = np.fft.fftn(np.fft.irfftn(a, s=self.s, axes=axes), axes=axes)
        self.state = self.torch.as_tensor(a, device=self.dev).to(self.cdt)

    def state_native(self) -> Any:
        if self.real:
            return self.state
        return self.state[..., : self.n // 2 + 1]

    # ------------------------------------------------------------------ transforms
    def _inv(self, a: Any) -> Any:
        fft = self.torch.fft
        if self.real:
            return fft.irfftn(a, s=self.s, dim=self.dims)
        return fft.ifftn(a, s=self.s, dim=self.dims).real

    def _fwd(self, a: Any) -> Any:
        fft = self.torch.fft
        if self.real:
            return fft.rfftn(a, dim=self.dims)
        return fft.fftn(a, dim=self.dims)

    def _phase(self, sign: float) -> Any:
        if self.shift is None:
            return 1.0
        arg = sum(k * float(dx) for k, dx in zip(self.k, self.shift, strict=True))
        return self.torch.exp(sign * 1j * arg)

    # ------------------------------------------------------------------ physics
    def nonlinear(self, st: Any) -> Any:
        """Dealiased nonlinear term (projected in 3D)."""
        torch = self.torch
        ps, ms = self._phase(1.0), self._phase(-1.0)
        if self.dim == 2:
            kx, ky = self.k
            w = st * ps
            psi = w * self.inv_k2
            if self.form == "basdevant":
                r = self._inv(torch.stack([1j * ky * psi, -1j * kx * psi]))
                u, v = r[0], r[1]
                q = self._fwd(torch.stack([v * v - u * u, u * v]))
                nl = kx * ky * q[0] + (kx * kx - ky * ky) * q[1]
            else:
                r = self._inv(torch.stack([1j * ky * psi, -1j * kx * psi, 1j * kx * w, 1j * ky * w]))
                nl = -self._fwd(r[0] * r[2] + r[1] * r[3])
            return nl * ms * self.mask
        k = self.k
        uh = st * ps
        if self.form == "rotational":
            wh = torch.stack([1j * (k[1] * uh[2] - k[2] * uh[1]),
                              1j * (k[2] * uh[0] - k[0] * uh[2]),
                              1j * (k[0] * uh[1] - k[1] * uh[0])])
            r = self._inv(torch.cat([uh, wh]))
            u, w = r[:3], r[3:]
            nl = self._fwd(torch.stack([u[1] * w[2] - u[2] * w[1],
                                        u[2] * w[0] - u[0] * w[2],
                                        u[0] * w[1] - u[1] * w[0]]))
        else:
            u = self._inv(uh)
            q = (u[0] * u[0] + u[1] * u[1] + u[2] * u[2]) / 3.0
            t = self._fwd(torch.stack([u[0] * u[0] - q, u[1] * u[1] - q, u[0] * u[1],
                                       u[0] * u[2], u[1] * u[2]]))
            t11, t22, t12, t13, t23 = t
            t33 = -(t11 + t22)
            nl = -1j * torch.stack([k[0] * t11 + k[1] * t12 + k[2] * t13,
                                    k[0] * t12 + k[1] * t22 + k[2] * t23,
                                    k[0] * t13 + k[1] * t23 + k[2] * t33])
        nl = nl * ms * self.mask
        div = (k[0] * nl[0] + k[1] * nl[1] + k[2] * nl[2]) * self.inv_k2
        return torch.stack([nl[i] - k[i] * div for i in range(3)])

    def _step(self, dt: float) -> None:
        torch = self.torch
        if self.cfg.dealias == "phase-shift":
            h = 2.0 * np.pi / self.n
            self.shift = self.rng.uniform(0.0, h, size=self.dim)
        u0 = self.state
        L = self.linear_operator(self.k2)
        E = torch.exp(L * dt)
        Eh = torch.exp(L * (0.5 * dt))
        k1 = self.nonlinear(u0)
        k2 = self.nonlinear(Eh * (u0 + 0.5 * dt * k1))
        if self.shift is not None:
            self.shift = self.shift + np.pi / self.n
        k3 = self.nonlinear(Eh * u0 + 0.5 * dt * k2)
        k4 = self.nonlinear(E * u0 + dt * Eh * k3)
        self.state = E * u0 + dt / 6.0 * (E * k1 + 2.0 * Eh * (k2 + k3) + k4)


class TorchEagerC2C(TorchEagerSolver):
    """Complex-to-complex FFTs on the full spectrum (the naive first rung)."""

    variant = "c2c-textbook"
    default_form = "advective"

    def __init__(self, cfg: Config) -> None:
        super().__init__(cfg if cfg.form != "auto" else cfg.with_(form=self._tb(cfg)), real=False)

    @staticmethod
    def _tb(cfg: Config) -> str:
        return "advective" if cfg.dim == 2 else "rotational"


class TorchEagerTextbook(TorchEagerSolver):
    """Real-to-complex FFTs, textbook FFT count (2D advective, 3D rotational)."""

    variant = "r2c-textbook"

    def __init__(self, cfg: Config) -> None:
        if cfg.form == "auto":
            cfg = cfg.with_(form="advective" if cfg.dim == 2 else "rotational")
        super().__init__(cfg, real=True)

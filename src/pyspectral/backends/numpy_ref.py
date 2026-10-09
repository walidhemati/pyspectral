"""Pure NumPy/SciPy reference backend: the correctness oracle (CPU, multithreaded FFTs)."""

from __future__ import annotations

from typing import Any

import numpy as np
import scipy.fft as sfft

from pyspectral.backends.base import SpectralSolver
from pyspectral.config import Config
from pyspectral.grid import k_squared, wavenumbers
from pyspectral.ops import NumpyOps


class NumpySolver(SpectralSolver):
    """Textbook implementation with every option (forms, phase-shift dealiasing, plain RK4).

    ``integrator="ifrk4"`` (default) integrates the linear term exactly (Lawson IF-RK4);
    ``integrator="rk4"`` treats it explicitly (used to demonstrate 4th-order convergence on
    the exact Taylor-Green decay, which IF-RK4 reproduces to round-off).
    """

    backend = "numpy"
    variant = "scipy-fft"
    phase_shift = True

    def __init__(self, cfg: Config, integrator: str = "ifrk4", workers: int | None = None,
                 seed: int = 1234) -> None:
        super().__init__(cfg)
        if integrator not in ("ifrk4", "rk4"):
            raise ValueError(integrator)
        self.integrator = integrator
        # threads only pay off on large grids
        self.workers = workers if workers is not None else (-1 if cfg.n**cfg.dim >= 2**16 else 1)
        self.rdt = np.float64 if cfg.is_fp64 else np.float32
        self.cdt = np.complex128 if cfg.is_fp64 else np.complex64
        n, d = self.n, self.dim
        self.k = tuple(k.astype(self.rdt) for k in wavenumbers(n, d))
        self.k2 = k_squared(n, d).astype(self.rdt)
        self.inv_k2 = np.where(self.k2 > 0, 1.0 / np.where(self.k2 > 0, self.k2, 1), 0).astype(
            self.rdt)
        self.mask = self.mask_np()
        self.s = (n,) * d
        self.axes = tuple(range(-d, 0))
        self.rng = np.random.default_rng(seed)
        self.shift: np.ndarray | None = None
        self.state = np.zeros(self.state_shape(), self.cdt)
        self._ops = NumpyOps()

    @property
    def ops(self) -> Any:
        return self._ops

    def set_state(self, state_hat: np.ndarray) -> None:
        self.state = self.check_state(state_hat).astype(self.cdt)

    def state_native(self) -> Any:
        return self.state

    # ------------------------------------------------------------------ transforms
    def _irfft(self, a: np.ndarray) -> np.ndarray:
        return sfft.irfftn(a, s=self.s, axes=self.axes, workers=self.workers)

    def _rfft(self, a: np.ndarray) -> np.ndarray:
        return sfft.rfftn(a, axes=self.axes, workers=self.workers)

    def _phase(self, sign: float) -> np.ndarray | float:
        if self.shift is None:
            return 1.0
        arg = sum(k * float(dx) for k, dx in zip(self.k, self.shift, strict=True))
        return np.exp(sign * 1j * arg).astype(self.cdt)

    # ------------------------------------------------------------------ physics
    def nonlinear(self, st: np.ndarray) -> np.ndarray:
        """Dealiased nonlinear term (and pressure projection in 3D) of a spectral state."""
        ps, ms = self._phase(+1.0), self._phase(-1.0)
        if self.dim == 2:
            kx, ky = self.k
            w = st * ps
            psi = w * self.inv_k2
            if self.form == "basdevant":
                u = self._irfft(1j * ky * psi)
                v = self._irfft(-1j * kx * psi)
                a = self._rfft(v * v - u * u)
                b = self._rfft(u * v)
                nl = kx * ky * a + (kx * kx - ky * ky) * b
            else:  # advective: -(u w_x + v w_y)
                u = self._irfft(1j * ky * psi)
                v = self._irfft(-1j * kx * psi)
                wx = self._irfft(1j * kx * w)
                wy = self._irfft(1j * ky * w)
                nl = -self._rfft(u * wx + v * wy)
            return (nl * ms * self.mask).astype(self.cdt)
        k = self.k
        uh = st * ps
        u = self._irfft(uh)
        if self.form == "rotational":
            wh = np.stack([1j * (k[1] * uh[2] - k[2] * uh[1]),
                           1j * (k[2] * uh[0] - k[0] * uh[2]),
                           1j * (k[0] * uh[1] - k[1] * uh[0])])
            w = self._irfft(wh)
            c = np.stack([u[1] * w[2] - u[2] * w[1],
                          u[2] * w[0] - u[0] * w[2],
                          u[0] * w[1] - u[1] * w[0]])
            nl = self._rfft(c)
        else:  # basdevant: -div(u u - |u|^2/3 I), 5 forward transforms
            q = (u[0] * u[0] + u[1] * u[1] + u[2] * u[2]) / 3.0
            t = self._rfft(np.stack([u[0] * u[0] - q, u[1] * u[1] - q, u[0] * u[1],
                                     u[0] * u[2], u[1] * u[2]]))
            t11, t22, t12, t13, t23 = t
            t33 = -(t11 + t22)
            nl = -1j * np.stack([k[0] * t11 + k[1] * t12 + k[2] * t13,
                                 k[0] * t12 + k[1] * t22 + k[2] * t23,
                                 k[0] * t13 + k[1] * t23 + k[2] * t33])
        nl = nl * ms * self.mask
        div = (k[0] * nl[0] + k[1] * nl[1] + k[2] * nl[2]) * self.inv_k2
        nl = np.stack([nl[i] - k[i] * div for i in range(3)])
        return nl.astype(self.cdt)

    def rhs(self, st: np.ndarray) -> np.ndarray:
        """Full right-hand side ``N(u) + L u`` (used by the explicit RK4 integrator)."""
        return self.nonlinear(st) + (self.linear_operator(self.k2) * st).astype(self.cdt)

    def _new_shift(self) -> None:
        if self.cfg.dealias == "phase-shift":
            h = 2.0 * np.pi / self.n
            self.shift = self.rng.uniform(0.0, h, size=self.dim)

    def _half_shift(self) -> None:
        if self.shift is not None:
            self.shift = self.shift + np.pi / self.n

    def _step(self, dt: float) -> None:
        u0 = self.state
        self._new_shift()
        if self.integrator == "rk4":
            k1 = self.rhs(u0)
            k2 = self.rhs(u0 + 0.5 * dt * k1)
            self._half_shift()
            k3 = self.rhs(u0 + 0.5 * dt * k2)
            k4 = self.rhs(u0 + dt * k3)
            self.state = (u0 + dt / 6.0 * (k1 + 2 * k2 + 2 * k3 + k4)).astype(self.cdt)
            return
        L = self.linear_operator(self.k2)
        E = np.exp(L * dt).astype(self.rdt)
        Eh = np.exp(L * (0.5 * dt)).astype(self.rdt)
        k1 = self.nonlinear(u0)
        k2 = self.nonlinear(Eh * (u0 + 0.5 * dt * k1))
        self._half_shift()
        k3 = self.nonlinear(Eh * u0 + 0.5 * dt * k2)
        k4 = self.nonlinear(E * u0 + dt * Eh * k3)
        self.state = (E * u0 + dt / 6.0 * (E * k1 + 2.0 * Eh * (k2 + k3) + k4)).astype(self.cdt)

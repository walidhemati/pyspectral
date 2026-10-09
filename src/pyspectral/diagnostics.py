"""Spectral diagnostics: energy, enstrophy, dissipation and shell spectra.

All functions take the spectral state (unnormalised r2c coefficients, NumPy convention) and
work for NumPy, CuPy and PyTorch arrays through :mod:`pyspectral.ops`.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from pyspectral.grid import hermitian_weights, k_squared
from pyspectral.ops import NumpyOps


def _abs2(a: Any) -> Any:
    return a.real * a.real + a.imag * a.imag


class Diagnostics:
    """Pre-computes k^2, Hermitian weights and shell indices for one grid on one device."""

    def __init__(self, n: int, dim: int, ops: Any = None, real_dtype: Any = None) -> None:
        self.n, self.dim = n, dim
        self.ops = ops or NumpyOps()
        k2 = k_squared(n, dim)
        w = np.ascontiguousarray(hermitian_weights(n, dim))
        norm = float(n) ** (2 * dim)
        inv_k2 = np.where(k2 > 0, 1.0 / np.where(k2 > 0, k2, 1.0), 0.0)
        shell = np.floor(np.sqrt(k2) + 0.5).astype(np.int64)
        self.nshell = int(shell.max()) + 1
        o = self.ops
        self.k2 = o.asarray(k2)
        self.wn = o.asarray(w / norm)
        self.inv_k2 = o.asarray(inv_k2)
        self.shell = o.asarray(shell.reshape(-1))
        from pyspectral.grid import wavenumbers

        self.kvec = tuple(o.asarray(np.ascontiguousarray(k)) for k in wavenumbers(n, dim))

    def max_velocity_sum(self, state: Any) -> float:
        """``sum_i max|u_i|`` in physical space (for the CFL condition)."""
        s = (self.n,) * self.dim
        o = self.ops
        if self.dim == 2:
            kx, ky = self.kvec
            psi = state * self.inv_k2
            u = o.irfftn(psi * ky * 1j, s)
            v = o.irfftn(psi * kx * (-1j), s)
            return o.amax_abs(u) + o.amax_abs(v)
        return sum(o.amax_abs(o.irfftn(state[i], s)) for i in range(3))

    def _density(self, state: Any) -> Any:
        """|state|^2 summed over components, times Hermitian weight / normalisation."""
        a = _abs2(state)
        if self.dim == 3:
            a = a.sum(0)
        return a * self.wn

    def energy(self, state: Any) -> float:
        """Kinetic energy ``<|u|^2>/2``."""
        d = self._density(state)
        if self.dim == 2:
            d = d * self.inv_k2
        return 0.5 * self.ops.scalar(d.sum())

    def enstrophy(self, state: Any) -> float:
        """Enstrophy ``<|omega|^2>/2``."""
        d = self._density(state)
        if self.dim == 3:
            d = d * self.k2
        return 0.5 * self.ops.scalar(d.sum())

    def palinstrophy(self, state: Any) -> float:
        """``<|grad omega|^2>/2`` (2D) — enstrophy dissipation is ``2 nu P``."""
        d = self._density(state) * self.k2
        if self.dim == 3:
            d = d * self.k2
        return 0.5 * self.ops.scalar(d.sum())

    def band_energy(self, state: Any, klo: float, khi: float) -> float:
        """Kinetic energy contained in ``klo <= |k| <= khi``."""
        d = self._density(state)
        if self.dim == 2:
            d = d * self.inv_k2
        sel = (self.k2 >= klo * klo) * (self.k2 <= khi * khi)
        return 0.5 * self.ops.scalar((d * sel).sum())

    def spectrum(self, state: Any) -> tuple[np.ndarray, np.ndarray]:
        """Shell-averaged energy spectrum ``E(k)`` with ``sum_k E(k) = E``."""
        d = self._density(state)
        if self.dim == 2:
            d = d * self.inv_k2
        e = self.ops.bincount(self.shell, 0.5 * d.reshape(-1), self.nshell)
        e = self.ops.to_numpy(e)
        return np.arange(len(e)), np.asarray(e, dtype=np.float64)

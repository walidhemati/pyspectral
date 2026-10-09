"""Spectral diagnostics: energy, enstrophy, dissipation and shell spectra.

All functions take the spectral state (unnormalised r2c coefficients, NumPy convention) and
work for NumPy, CuPy and PyTorch arrays through :mod:`pyspectral.ops`.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from pyspectral.ops import NumpyOps


def _abs2(a: Any) -> Any:
    return a.real * a.real + a.imag * a.imag


class Diagnostics:
    """Energy/enstrophy/spectra for one grid on one device.

    Only 1D wavenumber vectors are stored; k^2, weights and shell indices are rebuilt by
    broadcasting on each call (transient memory of a few fields, nothing persistent), so the
    diagnostics fit next to large runs.
    """

    def __init__(self, n: int, dim: int, ops: Any = None, real_dtype: Any = None) -> None:
        from pyspectral.grid import wavenumbers

        self.n, self.dim = n, dim
        self.ops = ops or NumpyOps()
        o = self.ops
        self.norm = float(n) ** (2 * dim)
        self.kvec = tuple(o.asarray(np.ascontiguousarray(k)) for k in wavenumbers(n, dim))
        w = np.full(n // 2 + 1, 2.0)
        w[0] = w[-1] = 1.0
        self.w1 = o.asarray(w.reshape((1,) * (dim - 1) + (-1,)) / self.norm)
        self.nshell = int(np.floor(np.sqrt(dim) * (n // 2) + 0.5)) + 1

    @property
    def k2(self) -> Any:
        out = self.kvec[0] * self.kvec[0]
        for k in self.kvec[1:]:
            out = out + k * k
        return out

    def _inv_k2(self, k2: Any) -> Any:
        return (k2 > 0) / (k2 + (k2 == 0))

    def _density(self, state: Any) -> Any:
        """|state|^2 summed over components, times Hermitian weight / normalisation."""
        if self.dim == 3:
            a = _abs2(state[0])
            for i in (1, 2):
                a = a + _abs2(state[i])
        else:
            a = _abs2(state)
        return a * self.w1

    def _energy_density(self, state: Any) -> Any:
        d = self._density(state)
        return d * self._inv_k2(self.k2) if self.dim == 2 else d

    def energy(self, state: Any) -> float:
        """Kinetic energy ``<|u|^2>/2``."""
        return 0.5 * self.ops.scalar(self._energy_density(state).sum())

    def enstrophy(self, state: Any) -> float:
        """Enstrophy ``<|omega|^2>/2``."""
        d = self._density(state)
        if self.dim == 3:
            d = d * self.k2
        return 0.5 * self.ops.scalar(d.sum())

    def palinstrophy(self, state: Any) -> float:
        """``<|grad omega|^2>/2`` (2D) — enstrophy dissipation is ``2 nu P``."""
        k2 = self.k2
        d = self._density(state) * k2
        if self.dim == 3:
            d = d * k2
        return 0.5 * self.ops.scalar(d.sum())

    def band_energy(self, state: Any, klo: float, khi: float) -> float:
        """Kinetic energy contained in ``klo <= |k| <= khi``."""
        k2 = self.k2
        d = self._energy_density(state)
        return 0.5 * self.ops.scalar((d * ((k2 >= klo * klo) * (k2 <= khi * khi))).sum())

    def spectrum(self, state: Any) -> tuple[np.ndarray, np.ndarray]:
        """Shell-averaged energy spectrum ``E(k)`` with ``sum_k E(k) = E``."""
        o = self.ops
        d = self._energy_density(state)
        shell = o.to_int(o.sqrt(self.k2) + 0.5).reshape(-1)
        e = o.bincount(shell, 0.5 * d.reshape(-1), self.nshell)
        e = np.asarray(o.to_numpy(e), dtype=np.float64)
        return np.arange(len(e)), e

    def max_velocity_sum(self, state: Any) -> float:
        """``sum_i max|u_i|`` in physical space (for the CFL condition)."""
        s = (self.n,) * self.dim
        o = self.ops
        if self.dim == 2:
            kx, ky = self.kvec
            psi = state * self._inv_k2(self.k2)
            u = o.irfftn(psi * ky * 1j, s)
            v = o.irfftn(psi * kx * (-1j), s)
            return o.amax_abs(u) + o.amax_abs(v)
        return sum(o.amax_abs(o.irfftn(state[i], s)) for i in range(3))

"""Wavenumbers, dealiasing masks and Hermitian weights for a periodic ``[0, 2*pi)^d`` box.

Layout convention (all backends): physical arrays are ``(n,)*d`` with axis order (x, y[, z]);
spectral arrays come from a real-to-complex FFT over the *last* axis and have shape
``(n,)*(d-1) + (n//2+1,)``. Wavenumbers are integers because the box side is ``2*pi``.
"""

from __future__ import annotations

import numpy as np


def spectral_shape(n: int, dim: int) -> tuple[int, ...]:
    """Shape of one scalar field in r2c spectral space."""
    return (n,) * (dim - 1) + (n // 2 + 1,)


def wavenumbers(n: int, dim: int) -> tuple[np.ndarray, ...]:
    """Broadcastable integer wavenumber arrays (float64) for the r2c spectral layout."""
    k = np.fft.fftfreq(n, 1.0 / n)
    kr = np.arange(n // 2 + 1, dtype=np.float64)
    if dim == 2:
        return k[:, None], kr[None, :]
    return k[:, None, None], k[None, :, None], kr[None, None, :]


def k_squared(n: int, dim: int) -> np.ndarray:
    """|k|^2 on the spectral grid."""
    ks = wavenumbers(n, dim)
    out = np.zeros(spectral_shape(n, dim))
    for k in ks:
        out = out + k * k
    return out


def dealias_mask(n: int, dim: int, kind: str = "2/3") -> np.ndarray:
    """Boolean mask of retained modes.

    ``"2/3"``: Orszag's rule as a cubic truncation, keep ``3|k_i| < n`` in every direction.
    ``"phase-shift"``: spherical truncation ``|k| < sqrt(2) n / 3`` used with phase shifts.
    """
    ks = wavenumbers(n, dim)
    if kind == "2/3":
        m = np.ones(spectral_shape(n, dim), dtype=bool)
        for k in ks:
            m = m & (3.0 * np.abs(k) < n)
        return m
    if kind == "phase-shift":
        return k_squared(n, dim) < 2.0 * (n / 3.0) ** 2
    raise ValueError(kind)


def hermitian_weights(n: int, dim: int) -> np.ndarray:
    """Weights that turn half-spectrum sums into full-spectrum sums (1 on kz=0 and kz=n/2)."""
    w = np.full(n // 2 + 1, 2.0)
    w[0] = 1.0
    w[-1] = 1.0
    shape = (1,) * (dim - 1) + (n // 2 + 1,)
    return np.broadcast_to(w.reshape(shape), spectral_shape(n, dim))


def physical_grid(n: int, dim: int) -> tuple[np.ndarray, ...]:
    """Cell-vertex coordinates ``x_j = 2*pi*j/n`` as broadcastable arrays."""
    x = 2.0 * np.pi * np.arange(n) / n
    if dim == 2:
        return x[:, None], x[None, :]
    return x[:, None, None], x[None, :, None], x[None, None, :]

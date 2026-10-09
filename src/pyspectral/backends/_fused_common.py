"""Helpers shared by the fused GPU backends (integrating-factor tables, buffer counts)."""

from __future__ import annotations

import numpy as np

from pyspectral.config import fft_split


def if_table(n: int, nu: float, dt: float) -> np.ndarray:
    """1D integrating-factor table ``[exp(-nu m^2 dt), exp(-nu m^2 dt/2)]`` for m=0..n/2."""
    m = np.arange(n // 2 + 1, dtype=np.float64)
    return np.concatenate([np.exp(-nu * m * m * dt), np.exp(-nu * m * m * 0.5 * dt)])


def if_scalars(alpha: float, amp: float, dt: float) -> tuple[float, float, float, float]:
    """Drag and band-forcing factors ``(e^{-a dt}, e^{-a dt/2}, e^{A dt}, e^{A dt/2})``."""
    return (float(np.exp(-alpha * dt)), float(np.exp(-alpha * 0.5 * dt)),
            float(np.exp(amp * dt)), float(np.exp(amp * 0.5 * dt)))


def n_work(dim: int, form: str) -> int:
    """Number of complex work fields (= max(inverse, forward) FFT batch)."""
    ni, nf = fft_split(dim, form)
    return max(ni, nf)

"""Initial conditions (returned as physical-space NumPy arrays, float64)."""

from __future__ import annotations

import numpy as np

from pyspectral.grid import dealias_mask, hermitian_weights, physical_grid, wavenumbers


def taylor_green_2d(n: int) -> np.ndarray:
    """2D Taylor-Green vorticity ``omega = 2 sin x sin y`` (u = sin x cos y, v = -cos x sin y).

    It is an exact steady Euler solution, so under Navier-Stokes it decays as
    ``omega(t) = omega(0) * exp(-2 nu t)``.
    """
    x, y = physical_grid(n, 2)
    return 2.0 * np.sin(x) * np.sin(y)


def smooth_2d(n: int) -> np.ndarray:
    """Smooth, non-band-limited, genuinely nonlinear 2D vorticity field (zero mean)."""
    x, y = physical_grid(n, 2)
    w = np.exp(0.8 * np.cos(x) + 0.6 * np.sin(2.0 * y + 0.4 * np.cos(x)))
    w = w + 0.5 * np.sin(x + y) * np.cos(y)
    return w - w.mean()


def _shaped_noise(n: int, dim: int, k0: float, seed: int, ncomp: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    axes = tuple(range(-dim, 0))
    noise = rng.standard_normal((ncomp,) + (n,) * dim)
    nh = np.fft.rfftn(noise, axes=axes)
    k2 = sum(k * k for k in wavenumbers(n, dim))
    kk = np.sqrt(k2)
    # amplitude so that the shell spectrum is ~ k^4 exp(-2 (k/k0)^2)
    shell_area = kk ** (dim - 1) if dim > 1 else 1.0
    with np.errstate(divide="ignore", invalid="ignore"):
        amp = np.sqrt(kk**4 * np.exp(-2.0 * (kk / k0) ** 2) / np.where(kk > 0, shell_area, 1.0))
    amp[kk == 0] = 0.0
    nh = nh * amp * dealias_mask(n, dim)
    return nh


def random_2d(n: int, k0: float = 8.0, seed: int = 0, energy: float = 0.5) -> np.ndarray:
    """Random-phase vorticity, energy spectrum ~ k^4 exp(-2(k/k0)^2), kinetic energy ``energy``."""
    nh = _shaped_noise(n, 2, k0, seed, 1)[0]
    k2 = sum(k * k for k in wavenumbers(n, 2))
    w8 = hermitian_weights(n, 2)
    with np.errstate(divide="ignore", invalid="ignore"):
        e = 0.5 * np.sum(np.where(k2 > 0, w8 * np.abs(nh) ** 2 / k2, 0.0)) / float(n) ** 4
    return np.fft.irfftn(nh, s=(n, n), axes=(0, 1)) * np.sqrt(energy / e)


def taylor_green_3d(n: int) -> np.ndarray:
    """3D Taylor-Green vortex ``u = sin x cos y cos z, v = -cos x sin y cos z, w = 0``.

    Returns an array of shape ``(3, n, n, n)``.
    """
    x, y, z = physical_grid(n, 3)
    u = np.sin(x) * np.cos(y) * np.cos(z)
    v = -np.cos(x) * np.sin(y) * np.cos(z)
    w = np.zeros_like(u * v)
    return np.stack(np.broadcast_arrays(u, v, w))


def random_3d(n: int, k0: float = 4.0, seed: int = 0, energy: float = 0.5) -> np.ndarray:
    """Random solenoidal velocity with spectrum ~ k^4 exp(-2(k/k0)^2) and energy ``energy``."""
    uh = _shaped_noise(n, 3, k0, seed, 3)
    ks = wavenumbers(n, 3)
    k2 = sum(k * k for k in ks)
    inv = np.where(k2 > 0, 1.0 / np.where(k2 > 0, k2, 1.0), 0.0)
    div = sum(k * c for k, c in zip(ks, uh, strict=True))
    uh = np.stack([c - k * div * inv for k, c in zip(ks, uh, strict=True)])
    u = np.fft.irfftn(uh, s=(n, n, n), axes=(1, 2, 3))
    e = 0.5 * np.mean(np.sum(u * u, axis=0))
    return u * np.sqrt(energy / e)


def to_spectral(field: np.ndarray, dim: int, dealias: str = "2/3") -> np.ndarray:
    """Physical -> dealiased r2c spectral state (complex128).

    2D: scalar vorticity ``(n, n)``; 3D: velocity ``(3, n, n, n)``.
    """
    n = field.shape[-1]
    axes = tuple(range(-dim, 0))
    fh = np.fft.rfftn(field, axes=axes)
    return fh * dealias_mask(n, dim, dealias)

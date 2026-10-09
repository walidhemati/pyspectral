"""Common solver interface implemented by every backend."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, ClassVar

import numpy as np

from pyspectral.config import Config, ffts_per_stage
from pyspectral.diagnostics import Diagnostics
from pyspectral.grid import dealias_mask, spectral_shape


class BackendUnavailable(RuntimeError):
    """Raised when a backend cannot run here (missing package, no GPU, unsupported option)."""


class SpectralSolver(ABC):
    """Pseudo-spectral Navier-Stokes solver on a periodic ``[0, 2*pi)^d`` box.

    State: 2D vorticity ``omega_hat`` with shape ``(n, n//2+1)``; 3D velocity ``u_hat`` with
    shape ``(3, n, n, n//2+1)``. Coefficients use the unnormalised forward FFT convention of
    ``numpy.fft.rfftn``. Time integration is integrating-factor RK4 (Lawson) unless a backend
    documents otherwise.
    """

    backend: ClassVar[str] = "base"
    variant: ClassVar[str] = "default"
    default_form: ClassVar[str] = "basdevant"
    dims: ClassVar[tuple[int, ...]] = (2, 3)
    dtypes: ClassVar[tuple[str, ...]] = ("float32", "float64")
    phase_shift: ClassVar[bool] = False
    forms: ClassVar[tuple[str, ...] | None] = None  # None: all forms

    def __init__(self, cfg: Config) -> None:
        if cfg.dim not in self.dims:
            raise BackendUnavailable(f"{self.backend}: dim={cfg.dim} not supported")
        if cfg.dtype not in self.dtypes:
            raise BackendUnavailable(f"{self.backend}: dtype={cfg.dtype} not supported")
        if cfg.dealias != "2/3" and not self.phase_shift:
            raise BackendUnavailable(f"{self.backend}: only 2/3-rule dealiasing")
        self.cfg = cfg
        self.n = cfg.n
        self.dim = cfg.dim
        self.form = cfg.resolved_form(self.default_form)
        if self.forms is not None and self.form not in self.forms:
            raise BackendUnavailable(f"{self.backend}: form={self.form} not supported")
        self.force_amp = cfg.force_amp
        self.t = 0.0
        self.nstep = 0
        self._diag: Diagnostics | None = None

    # ------------------------------------------------------------------ availability
    @classmethod
    def available(cls) -> tuple[bool, str]:
        """Whether the backend can run on this machine, and why not."""
        return True, ""

    # ------------------------------------------------------------------ abstract API
    @abstractmethod
    def set_state(self, state_hat: np.ndarray) -> None:
        """Load a spectral state (host complex array, NumPy convention)."""

    @abstractmethod
    def state_native(self) -> Any:
        """Current spectral state as a native (device) array with NumPy layout."""

    @abstractmethod
    def _step(self, dt: float) -> None:
        """Advance one IF-RK4 step of size ``dt``."""

    @property
    @abstractmethod
    def ops(self) -> Any:
        """Array shim for diagnostics (:mod:`pyspectral.ops`)."""

    # ------------------------------------------------------------------ shared API
    def get_state(self) -> np.ndarray:
        """Spectral state copied to the host as complex128."""
        return np.asarray(self.ops.to_numpy(self.state_native()), dtype=np.complex128)

    def step(self, dt: float) -> None:
        """Advance one time step."""
        self._step(dt)
        self.t += dt
        self.nstep += 1

    def advance(self, nsteps: int, dt: float) -> None:
        """Advance ``nsteps`` fixed steps."""
        for _ in range(nsteps):
            self.step(dt)

    def synchronize(self) -> None:
        """Block until queued device work is done."""
        self.ops.synchronize()

    def set_forcing_amplitude(self, amp: float) -> None:
        """Change the band-forcing growth rate ``A`` (used by constant-power control)."""
        self.force_amp = float(amp)

    @property
    def diag(self) -> Diagnostics:
        """Lazily built diagnostics on the backend's device."""
        if self._diag is None:
            self._diag = Diagnostics(self.n, self.dim, self.ops)
        return self._diag

    @property
    def fft_per_step(self) -> int:
        """Number of r2c/c2r FFTs per time step (4 RK stages)."""
        return 4 * ffts_per_stage(self.dim, self.form)

    def linear_operator(self, k2: Any) -> Any:
        """``L(k)`` for the integrating factor, evaluated on a k^2 array (NumPy/CuPy/torch)."""
        c = self.cfg
        L = -(c.nu * k2)
        if c.alpha != 0.0:
            L = L - (k2 * 0.0 + c.alpha) * (k2 <= c.drag_k2)
        if self.force_amp != 0.0:
            band = (k2 >= c.force_klo**2) * (k2 <= c.force_khi**2)
            L = L + (k2 * 0.0 + self.force_amp) * band  # keeps k2's dtype in torch
        return L

    def mask_np(self) -> np.ndarray:
        """Dealiasing mask on the host."""
        return dealias_mask(self.n, self.dim, self.cfg.dealias)

    def state_shape(self) -> tuple[int, ...]:
        """Shape of the spectral state."""
        s = spectral_shape(self.n, self.dim)
        return s if self.dim == 2 else (3,) + s

    def check_state(self, state_hat: np.ndarray) -> np.ndarray:
        """Validate shape and apply the dealiasing mask (host side)."""
        a = np.asarray(state_hat, dtype=np.complex128)
        if a.shape != self.state_shape():
            raise ValueError(f"state shape {a.shape} != {self.state_shape()}")
        return a * self.mask_np()

    def describe(self) -> str:
        """Human-readable one-liner."""
        return (f"{self.backend}/{self.variant} dim={self.dim} n={self.n} {self.cfg.dtype} "
                f"form={self.form} ({self.fft_per_step} FFTs/step)")

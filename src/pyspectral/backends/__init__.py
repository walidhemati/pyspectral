"""Backend registry: every rung of the optimisation ladder is one named variant.

Variants are imported lazily so a missing package (torch, cupy, triton, warp) or a missing GPU
only disables the affected entries instead of breaking the CLI.
"""

from __future__ import annotations

import importlib
from dataclasses import dataclass, field
from typing import Any

from pyspectral.backends.base import BackendUnavailable, SpectralSolver
from pyspectral.config import Config


@dataclass(frozen=True)
class Variant:
    """One registry entry (``backend/variant``)."""

    backend: str
    variant: str
    module: str
    cls: str
    color: str
    gpu: bool
    description: str
    kwargs: dict[str, Any] = field(default_factory=dict)
    default: bool = False  # what a bare backend name resolves to

    @property
    def name(self) -> str:
        return f"{self.backend}/{self.variant}"

    def load(self) -> type[SpectralSolver]:
        """Import and return the solver class."""
        mod = importlib.import_module(f"pyspectral.backends.{self.module}")
        return getattr(mod, self.cls)

    def available(self) -> tuple[bool, str]:
        """(ok, reason) without constructing a solver."""
        try:
            return self.load().available()
        except Exception as e:  # noqa: BLE001
            return False, f"{type(e).__name__}: {e}"

    def make(self, cfg: Config) -> SpectralSolver:
        """Construct a solver for ``cfg`` (raises :class:`BackendUnavailable`)."""
        cls = self.load()
        ok, why = cls.available()
        if not ok:
            raise BackendUnavailable(f"{self.name}: {why}")
        return cls(cfg, **self.kwargs)


COLORS = {
    "numpy": "#7f7f7f",
    "torch-eager": "#9467bd",
    "torch-compile": "#8c564b",
    "triton": "#1f77b4",
    "warp": "#2ca02c",
    "numba": "#ff7f0e",
    "cupy-cuda-c": "#d62728",
    "cupy": "#e377c2",
}

REGISTRY: list[Variant] = [
    Variant("numpy", "scipy-fft", "numpy_ref", "NumpySolver", COLORS["numpy"], False,
            "NumPy + scipy.fft (all CPU threads); the correctness oracle"),
    Variant("torch-eager", "c2c-textbook", "torch_eager", "TorchEagerC2C", COLORS["torch-eager"],
            True, "eager torch, complex-to-complex FFTs, textbook FFT count"),
    Variant("torch-eager", "r2c-textbook", "torch_eager", "TorchEagerTextbook",
            COLORS["torch-eager"], True, "eager torch, real-to-complex FFTs, textbook FFT count"),
    Variant("torch-eager", "r2c-minfft", "torch_eager", "TorchEagerSolver", COLORS["torch-eager"],
            True, "eager torch, r2c FFTs, Basdevant minimal FFT count"),
    Variant("torch-compile", "inductor", "torch_compile", "TorchCompileSolver",
            COLORS["torch-compile"], True, "torch.compile (Inductor) fused pointwise + torch.fft"),
    Variant("cupy", "elementwise", "cupy_elementwise", "CupyElementwiseSolver", COLORS["cupy"],
            True, "CuPy ElementwiseKernel fusion + cupyx.scipy.fft plan cache"),
    Variant("warp", "fused", "warp_fused", "WarpSolver", COLORS["warp"], True,
            "NVIDIA Warp fused kernels + torch.fft"),
    Variant("triton", "fused", "triton_fused", "TritonSolver", COLORS["triton"], True,
            "Triton fused kernels + torch.fft into preallocated buffers"),
    Variant("cupy-cuda-c", "fused-inplace", "cupy_cuda", "CupyCudaSolver", COLORS["cupy-cuda-c"],
            True, "hand-written CUDA C (NVRTC) fused kernels + in-place batched cuFFT plans",
            default=True),
    Variant("cupy-cuda-c", "fused-inplace-lean", "cupy_cuda", "CupyCudaLeanSolver",
            COLORS["cupy-cuda-c"], True, "as above, single-field FFT plans (minimal memory)"),
    Variant("cupy-cuda-c", "fused-inplace-graph", "cupy_cuda", "CupyCudaGraphSolver",
            COLORS["cupy-cuda-c"], True, "as above, whole time step replayed as a CUDA graph"),
]


def names() -> list[str]:
    """All registry names in ladder order."""
    return [v.name for v in REGISTRY]


def get(name: str) -> Variant:
    """Look up ``backend/variant``; a bare backend name returns its default (else last) variant."""
    for v in REGISTRY:
        if v.name == name:
            return v
    matches = [v for v in REGISTRY if v.backend == name]
    if matches:
        return next((v for v in matches if v.default), matches[-1])
    raise KeyError(f"unknown backend {name!r}; choose from {names()}")


def make_solver(name: str, cfg: Config) -> SpectralSolver:
    """Construct the named solver."""
    return get(name).make(cfg)


def available() -> list[tuple[Variant, bool, str]]:
    """Availability of every variant on this machine."""
    return [(v, *v.available()) for v in REGISTRY]


__all__ = ["REGISTRY", "Variant", "BackendUnavailable", "SpectralSolver", "get", "make_solver",
           "names", "available", "COLORS"]

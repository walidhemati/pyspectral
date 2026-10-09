"""Tiny array-module shim (NumPy / CuPy / PyTorch) for the non-hot-path code.

Hot loops are written per backend; diagnostics, CFL estimates and state transfer use this
shim so they are written once.
"""

from __future__ import annotations

from typing import Any

import numpy as np


class NumpyOps:
    """NumPy implementation of the shim."""

    name = "numpy"

    def asarray(self, a: np.ndarray, dtype: Any = None) -> Any:
        return np.asarray(a, dtype=dtype)

    def to_numpy(self, a: Any) -> np.ndarray:
        return np.asarray(a)

    def irfftn(self, a: Any, s: tuple[int, ...]) -> Any:
        import scipy.fft

        axes = tuple(range(-len(s), 0))
        return scipy.fft.irfftn(a, s=s, axes=axes, workers=-1)

    def bincount(self, idx: Any, weights: Any, minlength: int) -> Any:
        return np.bincount(idx, weights, minlength)

    def to_int(self, a: Any) -> Any:
        return a.astype(np.int64)

    def sqrt(self, a: Any) -> Any:
        return np.sqrt(a)

    def amax_abs(self, a: Any) -> float:
        return float(np.abs(a).max())

    def scalar(self, a: Any) -> float:
        return float(a)

    def synchronize(self) -> None:
        return None


class CupyOps(NumpyOps):
    """CuPy implementation (NumPy-compatible API)."""

    name = "cupy"

    def __init__(self) -> None:
        from pyspectral.backends._cuda_env import import_cupy

        self.cp = import_cupy()

    def asarray(self, a: np.ndarray, dtype: Any = None) -> Any:
        return self.cp.asarray(a, dtype=dtype)

    def to_numpy(self, a: Any) -> np.ndarray:
        return self.cp.asnumpy(a)

    def irfftn(self, a: Any, s: tuple[int, ...]) -> Any:
        import cupyx.scipy.fft as cfft

        axes = tuple(range(-len(s), 0))
        return cfft.irfftn(a, s=s, axes=axes)

    def bincount(self, idx: Any, weights: Any, minlength: int) -> Any:
        return self.cp.bincount(idx, weights, minlength)

    def sqrt(self, a: Any) -> Any:
        return self.cp.sqrt(a)

    def amax_abs(self, a: Any) -> float:
        return float(self.cp.abs(a).max())

    def synchronize(self) -> None:
        self.cp.cuda.get_current_stream().synchronize()


class TorchOps:
    """PyTorch implementation of the shim."""

    name = "torch"

    def __init__(self, device: str = "cuda") -> None:
        import torch

        self.torch = torch
        self.device = device

    def asarray(self, a: np.ndarray, dtype: Any = None) -> Any:
        t = self.torch.as_tensor(np.ascontiguousarray(a), device=self.device)
        return t if dtype is None else t.to(dtype)

    def to_numpy(self, a: Any) -> np.ndarray:
        return a.detach().cpu().numpy()

    def irfftn(self, a: Any, s: tuple[int, ...]) -> Any:
        dims = tuple(range(-len(s), 0))
        return self.torch.fft.irfftn(a, s=s, dim=dims)

    def bincount(self, idx: Any, weights: Any, minlength: int) -> Any:
        return self.torch.bincount(idx, weights=weights, minlength=minlength)

    def to_int(self, a: Any) -> Any:
        return a.to(self.torch.int64)

    def sqrt(self, a: Any) -> Any:
        return self.torch.sqrt(a)

    def amax_abs(self, a: Any) -> float:
        return float(a.abs().max())

    def scalar(self, a: Any) -> float:
        return float(a)

    def synchronize(self) -> None:
        self.torch.cuda.synchronize()

"""Import helpers for GPU packages.

Some environments ship CuPy without the cuFFT shared library it was built against while
PyTorch's wheels carry it (``nvidia/cufft/lib``). We preload it from there before importing
``cupy.cuda.cufft`` so the CuPy backends work in such environments too.
"""

from __future__ import annotations

import ctypes
import functools
import os
import sys
import warnings
from types import ModuleType
from typing import Any


@functools.lru_cache(maxsize=1)
def cuda_available() -> bool:
    """True if a CUDA device is usable from PyTorch or CuPy."""
    try:
        import torch

        if torch.cuda.is_available():
            return True
    except Exception:  # noqa: BLE001
        pass
    try:
        cp = import_cupy()
        return cp.cuda.runtime.getDeviceCount() > 0
    except Exception:  # noqa: BLE001
        return False


def _preload(subdir: str, names: tuple[str, ...]) -> bool:
    """dlopen the first ``nvidia/<subdir>/lib/<name>`` found on ``sys.path`` (RTLD_GLOBAL)."""
    for nm in names:
        try:
            ctypes.CDLL(nm, mode=ctypes.RTLD_GLOBAL)  # already resolvable by the loader
            return True
        except OSError:
            pass
    for p in sys.path:
        d = os.path.join(p, "nvidia", subdir, "lib")
        for nm in names:
            f = os.path.join(d, nm)
            if os.path.exists(f):
                try:
                    ctypes.CDLL(f, mode=ctypes.RTLD_GLOBAL)
                    return True
                except OSError:
                    continue
    return False


@functools.lru_cache(maxsize=1)
def import_cupy() -> ModuleType:
    """Import CuPy (silencing duplicate-wheel warnings) with cuFFT available."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        import cupy

    _preload("cuda_nvrtc", ("libnvrtc.so.12",))
    try:
        from cupy.cuda import cufft  # noqa: F401
    except ImportError:
        _preload("cufft", ("libcufft.so.11",))
        from cupy.cuda import cufft  # noqa: F401
    return cupy


def import_cufft() -> Any:
    """Return ``cupy.cuda.cufft`` (after :func:`import_cupy`)."""
    import_cupy()
    from cupy.cuda import cufft

    return cufft


def torch_cuda() -> Any:
    """Import torch and require CUDA."""
    import torch

    if not torch.cuda.is_available():
        raise ImportError("torch has no CUDA device")
    return torch

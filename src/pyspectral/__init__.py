"""pyspectral: pseudo-spectral incompressible Navier-Stokes DNS on the GPU from Python.

The package solves the 2D vorticity equation and the 3D rotational-form Navier-Stokes
equations on a periodic box ``[0, 2*pi)^d`` with Fourier pseudo-spectral discretisation,
2/3-rule dealiasing and an integrating-factor RK4 time integrator. Several interchangeable
backends (NumPy reference, PyTorch, CuPy, Triton, CUDA C via NVRTC, NVIDIA Warp) implement
the same :class:`~pyspectral.backends.base.SpectralSolver` interface.
"""

from pyspectral.config import Config

__all__ = ["Config", "__version__"]
__version__ = "0.1.0"

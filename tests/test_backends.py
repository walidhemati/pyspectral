"""Every GPU variant against the NumPy fp64 oracle (skipped without CUDA)."""

import math

import numpy as np
import pytest

from pyspectral import initial
from pyspectral.backends import REGISTRY, BackendUnavailable
from pyspectral.backends.numpy_ref import NumpySolver
from pyspectral.config import Config

GPU = [v for v in REGISTRY if v.gpu]
TOL = {"float64": 1e-12, "float32": 5e-6}


def _make(v, cfg):
    try:
        return v.make(cfg)
    except BackendUnavailable as e:
        pytest.skip(str(e))


@pytest.mark.gpu
@pytest.mark.parametrize("v", GPU, ids=lambda v: v.name)
@pytest.mark.parametrize("dtype", ["float64", "float32"])
@pytest.mark.parametrize("dim,form", [(2, "basdevant"), (3, "basdevant"), (3, "rotational")])
@pytest.mark.parametrize("alpha_kmax", [0.0, 2.0])
def test_matches_reference(v, dtype, dim, form, alpha_kmax):
    n = 32 if dim == 2 else 16
    cfg = Config(n=n, dim=dim, nu=0.01, alpha=0.5, alpha_kmax=alpha_kmax, force_amp=0.3,
                 force_klo=1, force_khi=2.5, dtype=dtype, form=form)
    s = _make(v, cfg)
    ref = NumpySolver(cfg.with_(dtype="float64"))
    ic = (initial.to_spectral(initial.random_2d(n, k0=4), 2) if dim == 2
          else initial.to_spectral(initial.random_3d(n, k0=2), 3))
    s.set_state(ic)
    ref.set_state(ic)
    s.advance(5, 0.01)
    ref.advance(5, 0.01)
    s.set_forcing_amplitude(0.1)  # changing the forcing mid-run must be honoured
    ref.set_forcing_amplitude(0.1)
    s.advance(3, 0.007)  # and so must a changed time step
    ref.advance(3, 0.007)
    r = ref.get_state()
    err = np.abs(s.get_state() - r).max() / np.abs(r).max()
    assert err < TOL[dtype], err


@pytest.mark.gpu
@pytest.mark.parametrize("v", GPU, ids=lambda v: v.name)
def test_tgv2d_exact(v):
    n, nu = 32, 0.05
    s = _make(v, Config(n=n, dim=2, nu=nu, dtype="float64"))
    w0 = initial.to_spectral(initial.taylor_green_2d(n), 2)
    s.set_state(w0)
    s.advance(20, 0.05)
    exact = w0 * math.exp(-2 * nu * 1.0)
    assert np.abs(s.get_state() - exact).max() / np.abs(exact).max() < 1e-13


@pytest.mark.gpu
def test_diagnostics_on_device_match_host():
    from pyspectral.backends import get
    from pyspectral.diagnostics import Diagnostics

    s = _make(get("cupy-cuda-c/fused-inplace"), Config(n=16, dim=3, dtype="float64"))
    u = initial.to_spectral(initial.random_3d(16, k0=2), 3)
    s.set_state(u)
    h = Diagnostics(16, 3)
    assert s.diag.energy(s.state_native()) == pytest.approx(h.energy(u), rel=1e-12)
    assert s.diag.max_velocity_sum(s.state_native()) == pytest.approx(
        h.max_velocity_sum(u), rel=1e-12)

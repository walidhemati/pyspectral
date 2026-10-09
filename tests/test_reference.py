"""NumPy reference: physics checks (CPU, run in CI)."""

import math

import numpy as np
import pytest

from pyspectral import initial
from pyspectral.backends.numpy_ref import NumpySolver
from pyspectral.config import Config
from pyspectral.diagnostics import Diagnostics
from pyspectral.validation import pad_spectrum, tgv2d_error, tgv2d_rk4_order


@pytest.mark.parametrize("form", ["basdevant", "advective"])
@pytest.mark.parametrize("dtype,tol", [("float64", 1e-13), ("float32", 2e-6)])
def test_tgv2d_exact_decay(form, dtype, tol):
    cfg = Config(n=16, dim=2, nu=0.1, dtype=dtype, form=form)
    s = NumpySolver(cfg)
    w0 = initial.to_spectral(initial.taylor_green_2d(16), 2)
    s.set_state(w0)
    s.advance(20, 0.05)
    exact = w0 * math.exp(-2 * 0.1 * 1.0)
    assert np.abs(s.get_state() - exact).max() / np.abs(exact).max() < tol


def test_tgv2d_registry_entry():
    assert tgv2d_error("numpy", n=16) < 1e-13


def test_rk4_is_fourth_order():
    r = tgv2d_rk4_order()
    assert 3.8 < r["order_rk4"] < 4.3
    assert max(r["err_ifrk4"]) < 1e-13


def test_forms_agree_2d_and_3d():
    w0 = initial.to_spectral(initial.smooth_2d(32), 2)
    a = NumpySolver(Config(n=32, dim=2, nu=0.01, dtype="float64", form="basdevant"))
    b = NumpySolver(Config(n=32, dim=2, nu=0.01, dtype="float64", form="advective"))
    for s in (a, b):
        s.set_state(w0)
        s.advance(5, 0.01)
    assert np.abs(a.get_state() - b.get_state()).max() < 1e-12 * np.abs(a.get_state()).max()
    u0 = initial.to_spectral(initial.random_3d(16, k0=2), 3)
    a = NumpySolver(Config(n=16, dim=3, nu=0.01, dtype="float64", form="basdevant"))
    b = NumpySolver(Config(n=16, dim=3, nu=0.01, dtype="float64", form="rotational"))
    for s in (a, b):
        s.set_state(u0)
        s.advance(5, 0.01)
    assert np.abs(a.get_state() - b.get_state()).max() < 1e-12 * np.abs(a.get_state()).max()


def test_inviscid_energy_conservation_and_divergence_free():
    n = 16
    u0 = initial.to_spectral(initial.random_3d(n, k0=2), 3)
    s = NumpySolver(Config(n=n, dim=3, nu=0.0, dtype="float64"))
    s.set_state(u0)
    d = Diagnostics(n, 3)
    e0 = d.energy(u0)
    s.advance(10, 0.002)
    u = s.get_state()
    assert abs(d.energy(u) - e0) / e0 < 1e-9
    from pyspectral.grid import wavenumbers

    kx, ky, kz = wavenumbers(n, 3)
    div = kx * u[0] + ky * u[1] + kz * u[2]
    assert np.abs(div).max() < 1e-10 * np.abs(u).max()


def test_energy_budget_2d():
    """dE/dt = -2 nu Z for unforced 2D flow."""
    n, nu, dt = 32, 0.01, 1e-3
    s = NumpySolver(Config(n=n, dim=2, nu=nu, dtype="float64"))
    s.set_state(initial.to_spectral(initial.smooth_2d(n), 2))
    d = Diagnostics(n, 2)
    e = []
    z = []
    for _ in range(3):
        e.append(d.energy(s.state))
        z.append(d.enstrophy(s.state))
        s.step(dt)
    dedt = (e[2] - e[0]) / (2 * dt)
    assert abs(dedt + 2 * nu * z[1]) < 1e-6 * abs(dedt)


def test_tgv3d_initial_diagnostics():
    d = Diagnostics(16, 3)
    u = initial.to_spectral(initial.taylor_green_3d(16), 3)
    assert d.energy(u) == pytest.approx(0.125)
    assert d.enstrophy(u) == pytest.approx(0.375)
    k, e = d.spectrum(u)
    assert e.sum() == pytest.approx(0.125)


def test_phase_shift_close_to_two_thirds():
    n = 32
    u0 = initial.to_spectral(initial.random_3d(n, k0=2), 3)
    a = NumpySolver(Config(n=n, dim=3, nu=0.02, dtype="float64"))
    b = NumpySolver(Config(n=n, dim=3, nu=0.02, dtype="float64", dealias="phase-shift"))
    for s in (a, b):
        s.set_state(u0)
        s.advance(10, 0.01)
    d = Diagnostics(n, 3)
    assert d.energy(b.get_state()) == pytest.approx(d.energy(a.get_state()), rel=1e-3)


def test_pad_spectrum_roundtrip():
    w = initial.to_spectral(initial.taylor_green_2d(16), 2)
    big = pad_spectrum(w, 32, 2)
    f = np.fft.irfftn(big, s=(32, 32), axes=(0, 1))
    x = 2 * np.pi * np.arange(32) / 32
    assert np.allclose(f, 2 * np.sin(x)[:, None] * np.sin(x)[None, :], atol=1e-12)


def test_forcing_band_injects_energy():
    n = 32
    cfg = Config(n=n, dim=2, nu=0.0, dtype="float64", force_amp=1.0, force_klo=1, force_khi=1.5)
    s = NumpySolver(cfg)
    w0 = initial.to_spectral(initial.taylor_green_2d(n), 2)  # |k| = sqrt(2) inside the band
    s.set_state(w0)
    s.advance(10, 0.01)
    assert np.allclose(s.get_state(), w0 * math.exp(0.1), rtol=1e-12, atol=1e-9)

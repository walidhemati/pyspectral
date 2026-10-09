from pyspectral.cli import main
from pyspectral.simulate import run


def test_cli_list(capsys):
    assert main(["list"]) == 0
    assert "numpy/scipy-fft" in capsys.readouterr().out


def test_cli_run_numpy(capsys):
    assert main(["run", "--backend", "numpy", "--n", "16", "--t-end", "0.1", "--case", "tgv"]) == 0
    assert "steps" in capsys.readouterr().out


def test_cfl_run_lands_on_t_end():
    from pyspectral import initial
    from pyspectral.backends.numpy_ref import NumpySolver
    from pyspectral.config import Config

    s = NumpySolver(Config(n=16, dim=2, nu=0.01, dtype="float64"))
    s.set_state(initial.to_spectral(initial.random_2d(16, k0=3), 2))
    res = run(s, 0.3, cfl=0.5, diag_dt=0.1)
    assert abs(s.t - 0.3) < 1e-12
    assert len(res.t) == 4

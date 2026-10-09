"""Validation experiments. Each returns a JSON-serialisable dict (written to ``results/``)."""

from __future__ import annotations

import json
import math
import time
from importlib import resources
from pathlib import Path
from typing import Any

import numpy as np

from pyspectral import initial
from pyspectral.backends import REGISTRY, BackendUnavailable, get, make_solver
from pyspectral.backends.numpy_ref import NumpySolver
from pyspectral.config import Config
from pyspectral.diagnostics import Diagnostics
from pyspectral.grid import spectral_shape
from pyspectral.simulate import run

TGV_NU = 1.0 / 1600.0


# ----------------------------------------------------------------------------- helpers
def pad_spectrum(a: np.ndarray, n_to: int, dim: int) -> np.ndarray:
    """Zero-pad (or truncate) an r2c spectrum to grid ``n_to`` keeping physical amplitudes."""
    n = a.shape[-2] if dim == 2 else a.shape[-2]
    out = np.zeros(a.shape[: a.ndim - dim] + spectral_shape(n_to, dim), dtype=a.dtype)
    m = min(n, n_to) // 2  # keep |k| < m on full axes, k < m on the half axis
    full = list(range(0, m)) + list(range(-m + 1, 0))
    idx_src = [np.array([k % n for k in full])] * (dim - 1) + [np.arange(m)]
    idx_dst = [np.array([k % n_to for k in full])] * (dim - 1) + [np.arange(m)]
    src = a[(...,) + np.ix_(*idx_src)]
    out[(...,) + np.ix_(*idx_dst)] = src * (n_to / n) ** dim
    return out


def rel_err(a: np.ndarray, b: np.ndarray) -> float:
    """max|a-b| / max|b|."""
    return float(np.abs(a - b).max() / np.abs(b).max())


def fit_slope(x: np.ndarray, y: np.ndarray) -> float:
    """Least-squares slope of log(y) vs log(x)."""
    return float(np.polyfit(np.log(x), np.log(y), 1)[0])


def load_tgv_reference() -> dict[str, Any]:
    """The HiOCFD C3.5 512^3 spectral reference (t, E, -dE/dt, enstrophy)."""
    for p in (Path(__file__).resolve().parents[2] / "data" / "reference",):
        f = p / "tgv_re1600_reference.json"
        if f.exists():
            return json.loads(f.read_text())
    txt = resources.files("pyspectral").joinpath("tgv_re1600_reference.json")
    return json.loads(txt.read_text())


# ----------------------------------------------------------------------------- 2D TGV
def tgv2d_error(variant: str = "numpy", n: int = 32, nu: float = 0.05, t_end: float = 1.0,
                dt: float = 0.01, dtype: str = "float64") -> float:
    """Max relative error of the 2D Taylor-Green vortex vs the exact decay ``exp(-2 nu t)``."""
    cfg = Config(n=n, dim=2, nu=nu, dtype=dtype)
    s = make_solver(variant, cfg)
    w0 = initial.to_spectral(initial.taylor_green_2d(n), 2)
    s.set_state(w0)
    s.advance(int(round(t_end / dt)), dt)
    return rel_err(s.get_state(), w0 * math.exp(-2.0 * nu * s.t))


def tgv2d_rk4_order(dts: tuple[float, ...] = (0.2, 0.1, 0.05, 0.025), nu: float = 0.5,
                    t_end: float = 1.0) -> dict[str, Any]:
    """Temporal order on the exact TGV decay with *explicit* RK4 (IF-RK4 is exact here)."""
    errs, errs_if = [], []
    for dt in dts:
        for integ, store in (("rk4", errs), ("ifrk4", errs_if)):
            cfg = Config(n=16, dim=2, nu=nu, dtype="float64")
            s = NumpySolver(cfg, integrator=integ)
            w0 = initial.to_spectral(initial.taylor_green_2d(16), 2)
            s.set_state(w0)
            s.advance(int(round(t_end / dt)), dt)
            store.append(rel_err(s.get_state(), w0 * math.exp(-2 * nu * s.t)))
    return {"dt": list(dts), "err_rk4": errs, "err_ifrk4": errs_if,
            "order_rk4": fit_slope(np.array(dts), np.array(errs))}


def temporal_convergence(dts: tuple[float, ...] = (0.04, 0.02, 0.01, 0.005, 0.0025),
                         n: int = 64, nu: float = 0.01, t_end: float = 1.0) -> dict[str, Any]:
    """IF-RK4 self-convergence on a nonlinear 2D flow (reference: dt_min / 8)."""
    w0 = initial.to_spectral(initial.smooth_2d(n), 2)

    def solve(dt: float) -> np.ndarray:
        s = NumpySolver(Config(n=n, dim=2, nu=nu, dtype="float64"))
        s.set_state(w0)
        s.advance(int(round(t_end / dt)), dt)
        return s.get_state()

    ref = solve(min(dts) / 8)
    errs = [rel_err(solve(dt), ref) for dt in dts]
    return {"dt": list(dts), "err": errs, "order": fit_slope(np.array(dts), np.array(errs)),
            "n": n, "nu": nu, "t_end": t_end}


def spatial_convergence(ns: tuple[int, ...] = (8, 12, 16, 20, 24, 32, 40, 48, 64),
                        n_ref: int = 192, nu: float = 0.02, t_end: float = 0.5,
                        dt: float = 2.5e-3) -> dict[str, Any]:
    """Spectral (exponential) convergence in n for a smooth non-band-limited 2D flow."""
    def solve(n: int) -> np.ndarray:
        s = NumpySolver(Config(n=n, dim=2, nu=nu, dtype="float64"))
        s.set_state(initial.to_spectral(initial.smooth_2d(n), 2))
        s.advance(int(round(t_end / dt)), dt)
        return s.get_state()

    ref = solve(n_ref)
    w_ref = np.fft.irfftn(ref, s=(n_ref, n_ref), axes=(0, 1))
    errs = []
    for n in ns:
        w = np.fft.irfftn(pad_spectrum(solve(n), n_ref, 2), s=(n_ref, n_ref), axes=(0, 1))
        errs.append(float(np.abs(w - w_ref).max() / np.abs(w_ref).max()))
    return {"n": list(ns), "err": errs, "n_ref": n_ref, "nu": nu, "t_end": t_end, "dt": dt}


# ----------------------------------------------------------------------------- backends
def backend_agreement(steps: int = 10) -> list[dict[str, Any]]:
    """Every available GPU variant vs the NumPy reference (fp64 oracle), 2D and 3D."""
    rows = []
    cases = [(2, 64, "basdevant"), (3, 32, "basdevant"), (3, 32, "rotational"),
             (2, 64, "advective")]
    for dim, n, form in cases:
        ic = (initial.to_spectral(initial.random_2d(n, k0=6), 2) if dim == 2
              else initial.to_spectral(initial.random_3d(n, k0=3), 3))
        base = Config(n=n, dim=dim, nu=0.005, alpha=0.01, form=form, dtype="float64")
        ref = NumpySolver(base)
        ref.set_state(ic)
        ref.advance(steps, 0.01)
        r = ref.get_state()
        for v in REGISTRY:
            if not v.gpu:
                continue
            for dtype in ("float64", "float32"):
                try:
                    s = v.make(base.with_(dtype=dtype))
                except BackendUnavailable:
                    continue
                s.set_state(ic)
                s.advance(steps, 0.01)
                rows.append({"variant": v.name, "dim": dim, "n": n, "form": s.form,
                             "dtype": dtype, "steps": steps, "rel_err": rel_err(s.get_state(), r)})
    return rows


# ----------------------------------------------------------------------------- 3D TGV
def tgv3d(variant: str = "cupy-cuda-c/fused-inplace", n: int = 128, t_end: float = 20.0,
          cfl: float = 0.5, dtype: str = "float32", diag_dt: float = 0.05,
          form: str = "auto", checkpoint: str | Path | None = None,
          wall_limit: float | None = None) -> dict[str, Any]:
    """Taylor-Green vortex at Re=1600; dissipation compared with the 512^3 reference.

    With ``checkpoint`` and ``wall_limit`` the run stops after ``wall_limit`` seconds, saves
    its state and returns ``{"finished": False}``; calling again resumes it.
    """
    from pyspectral.simulate import RunResult

    cfg = Config(n=n, dim=3, nu=TGV_NU, dtype=dtype, form=form)
    s = make_solver(variant, cfg)
    prev = None
    ck = Path(checkpoint) if checkpoint else None
    if ck is not None and ck.exists():
        z = np.load(ck, allow_pickle=False)
        s.set_state(z["state"])
        s.t = float(z["t"])
        h = json.loads(str(z["hist"]))
        prev = RunResult(t=h["t"], energy=h["energy"], enstrophy=h["enstrophy"], dt=h["dt"],
                         steps=h["steps"], wall=h["wall_s"])
    else:
        s.set_state(initial.to_spectral(initial.taylor_green_3d(n), 3))
    res = run(s, t_end, cfl=cfl, dt_max=0.05, diag_dt=diag_dt, wall_limit=wall_limit,
              result=prev)
    if not res.finished:
        if ck is None:
            raise RuntimeError("wall_limit reached without a checkpoint path")
        ck.parent.mkdir(parents=True, exist_ok=True)
        np.savez(ck, state=s.get_state().astype(np.complex64 if dtype == "float32" else complex),
                 t=s.t,
                 hist=json.dumps(res.as_dict()))
        return {"finished": False, "t": s.t, "steps": res.steps}
    if ck is not None and ck.exists():
        ck.unlink()
    t = np.array(res.t)
    E = np.array(res.energy)
    eps = 2.0 * TGV_NU * np.array(res.enstrophy)
    dEdt = -np.gradient(E, t)
    ref = load_tgv_reference()
    tr, er = np.array(ref["t"]), np.array(ref["dissipation"])
    sel = t <= tr.max()
    eps_ref_i = np.interp(t[sel], tr, er)
    peak = er.max()
    i = int(np.argmax(eps))
    out = {
        "finished": True, "variant": variant, "n": n, "dtype": dtype, "form": s.form, "cfl": cfl,
        "steps": res.steps, "wall_s": res.wall, "t": t.tolist(), "energy": E.tolist(),
        "dissipation_2nuZ": eps.tolist(), "dissipation_dEdt": dEdt.tolist(),
        "max_abs_err_over_peak": float(np.abs(eps[sel] - eps_ref_i).max() / peak),
        "peak_value": float(eps[i]), "peak_time": float(t[i]),
        "ref_peak_value": float(peak), "ref_peak_time": float(tr[int(np.argmax(er))]),
        "peak_rel_err": float((eps[i] - peak) / peak),
        "energy_rel_err_t20": float(abs(E[-1] - np.interp(min(t[-1], tr.max()), tr,
                                                          np.array(ref["E"]))) /
                                    np.interp(min(t[-1], tr.max()), tr, np.array(ref["E"]))),
    }
    return out


# ----------------------------------------------------------------------------- forced
def forced(variant: str, dim: int, n: int, nu: float, alpha: float, klo: float, khi: float,
           power: float, t_end: float, t_avg: float, cfl: float = 0.5, seed: int = 0,
           dtype: str = "float32", fit_range: tuple[float, float] = (8.0, 30.0)
           ) -> dict[str, Any]:
    """Constant-power band-forced turbulence; time-averaged spectrum over ``[t_avg, t_end]``."""
    cfg = Config(n=n, dim=dim, nu=nu, alpha=alpha, force_amp=0.0, force_klo=klo,
                 force_khi=khi, dtype=dtype)
    s = make_solver(variant, cfg)
    s.set_forcing_amplitude(1.0)  # non-zero so the band test is active
    ic = (initial.random_2d(n, k0=(klo + khi) / 2, seed=seed, energy=0.05) if dim == 2
          else initial.random_3d(n, k0=2.0, seed=seed, energy=0.5))
    s.set_state(initial.to_spectral(ic, dim))
    spectra: list[np.ndarray] = []

    def cb(sol: Any) -> None:
        if sol.t >= t_avg:
            spectra.append(sol.diag.spectrum(sol.state_native())[1])

    t0 = time.perf_counter()
    res = run(s, t_end, cfl=cfl, dt_max=0.05, diag_dt=t_end / 200, target_power=power,
              callback=cb, callback_dt=(t_end - t_avg) / 50)
    k = np.arange(len(spectra[0]))
    Ek = np.mean(spectra, axis=0)
    sel = (k >= fit_range[0]) & (k <= fit_range[1])
    slope = fit_slope(k[sel], Ek[sel])
    E = res.energy[-1]
    out = {"variant": variant, "dim": dim, "n": n, "nu": nu, "alpha": alpha, "band": [klo, khi],
           "power": power, "t_end": t_end, "t_avg": t_avg, "k": k.tolist(), "E_k": Ek.tolist(),
           "nspectra": len(spectra), "fit_range": list(fit_range), "slope": slope,
           "t": res.t, "energy": res.energy, "enstrophy": res.enstrophy,
           "steps": res.steps, "wall_s": time.perf_counter() - t0}
    if dim == 3:
        eps = 2 * nu * float(np.mean(res.enstrophy[len(res.enstrophy) // 2:]))
        urms = math.sqrt(2.0 / 3.0 * E)
        lam = math.sqrt(15.0 * nu * urms**2 / eps)
        eta = (nu**3 / eps) ** 0.25
        out.update(epsilon=eps, re_lambda=urms * lam / nu, eta=eta, kmax_eta=(n / 3) * eta)
    return out


def write_json(obj: Any, path: str | Path) -> None:
    """Write JSON with a trailing newline."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(obj, indent=1) + "\n")


def diag_for(n: int, dim: int) -> Diagnostics:
    """Host diagnostics helper."""
    return Diagnostics(n, dim)


__all__ = ["tgv2d_error", "tgv2d_rk4_order", "temporal_convergence", "spatial_convergence",
           "backend_agreement", "tgv3d", "forced", "pad_spectrum", "load_tgv_reference", "get"]

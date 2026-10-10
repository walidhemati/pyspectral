"""Same-machine head-to-heads against established pseudo-spectral codes.

    ~/.venvs/pyhpc/bin/python bench/counterparts.py --quick   # < 60 s, < 1 GB GPU, smoke test
    ~/.venvs/pyhpc/bin/python bench/counterparts.py --full    # idle GPU + idle CPU, publishable

Writes ``results/counterparts.json``. Every counterpart runs in its own environment, driven
through a small script in this folder, so nothing is installed into the pyspectral env:

* **JAX-CFD** (``jax_cfd.spectral``, 2D vorticity, GPU via XLA/cuFFT): ``bench/cp_jaxcfd.py``
  in the venv ``$PYSPECTRAL_JAXCFD_PY`` (default ``~/.venvs/cp-jaxcfd/bin/python``).
* **spectralDNS** (3D triply periodic NS, CPU, MPI + FFTW): ``bench/cp_spectraldns.py`` under
  ``mpiexec`` from the conda env ``$PYSPECTRAL_SDNS_ENV`` (default
  ``~/python-hpc/_ref/envs/spectraldns``), sources in ``$PYSPECTRAL_SDNS_SRC``.
* **Dedalus** (``gpu`` branch, 3D, CuPy): ``bench/cp_dedalus.py`` in ``$PYSPECTRAL_DEDALUS_PY``.

Environments used for the paper (sources pinned in ``results/counterparts.json``)::

    uv venv -p 3.12 ~/.venvs/cp-jaxcfd
    uv pip install --python ~/.venvs/cp-jaxcfd/bin/python "jax[cuda12]" tree-math scipy
    uv pip install --python ~/.venvs/cp-jaxcfd/bin/python --no-deps -e _ref/jax-cfd
    conda create -p _ref/envs/spectraldns -c conda-forge --override-channels python=3.12 \
        shenfun=4.3 mpi4py-fft=2.0.6 mpi4py mpich cython h5py numba   # spectralDNS via PYTHONPATH
    conda create -p _ref/envs/dedalus-gpu -c conda-forge --override-channels python=3.12 \
        "fftw=*=mpi_mpich*" mpich mpi4py cython numpy scipy h5py numexpr docopt xarray \
        matplotlib-base array-api-compat compilers
    _ref/envs/dedalus-gpu/bin/pip install "cupy-cuda12x[ctk]==14.2.0"
    _ref/envs/dedalus-gpu/bin/pip install --no-deps --no-build-isolation -e _ref/dedalus-gpu

A counterpart whose environment is missing is skipped (recorded under ``skipped``). pyspectral
itself is timed with the same harness as ``bench/run.py`` (``pyspectral.benchmark``).
Each code also advances the same initial condition and the energy/enstrophy are compared with
the pyspectral NumPy oracle (``verification``).
"""

from __future__ import annotations

import argparse
import json
import math
import os
import platform
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from pyspectral import initial  # noqa: E402
from pyspectral.backends.numpy_ref import NumpySolver  # noqa: E402
from pyspectral.benchmark import _free_gpu, machine_info, time_solver  # noqa: E402
from pyspectral.config import Config  # noqa: E402
from pyspectral.diagnostics import Diagnostics  # noqa: E402

HOME = Path.home()
JAX_PY = Path(os.environ.get("PYSPECTRAL_JAXCFD_PY", HOME / ".venvs/cp-jaxcfd/bin/python"))
SDNS_ENV = Path(os.environ.get("PYSPECTRAL_SDNS_ENV", HOME / "python-hpc/_ref/envs/spectraldns"))
SDNS_SRC = Path(os.environ.get("PYSPECTRAL_SDNS_SRC", HOME / "python-hpc/_ref/spectralDNS"))
DED_PY = Path(os.environ.get("PYSPECTRAL_DEDALUS_PY",
                             HOME / "python-hpc/_ref/envs/dedalus-gpu/bin/python"))
JAX_SRC = HOME / "python-hpc/_ref/jax-cfd"
DED_SRC = HOME / "python-hpc/_ref/dedalus-gpu"
HERE = Path(__file__).resolve().parent


# ----------------------------------------------------------------------------- plans
def plan(mode: str) -> dict[str, Any]:
    """Sizes and timing parameters. ``quick`` stays < 60 s and < 1 GB on a shared GPU."""
    if mode == "quick":
        return {"min_time": 0.05, "repeats": 5, "jax_mem_fraction": "0.12",
                "sizes2d": [256, 1024], "dtypes2d": ["float32"],
                "py2d": ["cupy-cuda-c/fused-inplace", "torch-eager/r2c-textbook"],
                "verify2d_n": 256, "verify2d_steps": 100,
                "sdns": [(64, "double", 8)], "sdns_verify_steps": 10, "verify3d_nmax": 64,
                "py3d_cpu": [(64, "float64")],
                "py3d_gpu": [(48, "float64"), (64, "float64"), (64, "float32")],
                "ded": [(48, "float64")], "ded_verify_steps": 10}
    return {"min_time": 0.5, "repeats": 5, "jax_mem_fraction": "0.75",
            "sizes2d": [256, 512, 1024, 2048, 4096, 8192], "dtypes2d": ["float32", "float64"],
            "max_n_fp64": 4096,
            "py2d": ["cupy-cuda-c/fused-inplace", "triton/fused", "torch-compile/inductor",
                     "torch-eager/r2c-textbook"],
            "verify2d_n": 256, "verify2d_steps": 200,
            "sdns": [(n, p, r) for n in (64, 128, 256) for p in ("double", "single")
                     for r in (8, 16)],
            "sdns_verify_steps": 50, "verify3d_nmax": 128,
            "py3d_cpu": [(n, d) for n in (64, 128, 256) for d in ("float64", "float32")],
            "py3d_gpu": [(n, d) for n in (48, 64, 96, 128, 256) for d in ("float32", "float64")],
            # Dedalus keeps M = 2n/3 modes (n % 3 == 0); M = 64 already needs 2-3.4 GB
            "ded": [(n, d) for n in (48, 96) for d in ("float64", "float32")],
            "ded_verify_steps": 50}


# ----------------------------------------------------------------------------- helpers
def _stats(ts: list[float], npts: float, nfft: int) -> dict[str, Any]:
    a = np.asarray(ts)
    tm = float(np.median(a))
    return {"step_ms": 1e3 * tm, "step_ms_min": 1e3 * float(a.min()),
            "step_ms_max": 1e3 * float(a.max()), "repeats": len(ts),
            "Mpts_per_s": npts / tm / 1e6, "fft_Gpts_per_s": nfft * npts / tm / 1e9}


def _git(path: Path) -> str | None:
    try:
        out = subprocess.run(["git", "-C", str(path), "rev-parse", "HEAD"], capture_output=True,
                             text=True, timeout=10)
        return out.stdout.strip() or None
    except Exception:  # noqa: BLE001
        return None


def _rel(a: float, b: float) -> float:
    return abs(a - b) / max(abs(b), 1e-300)


def _diag2(n: int, w: np.ndarray) -> tuple[float, float]:
    d = Diagnostics(n, 2)
    return d.energy(w), d.enstrophy(w)


# ----------------------------------------------------------------------------- pyspectral
def time_pyspectral(name: str, n: int, dim: int, dtype: str, form: str, p: dict[str, Any],
                    notes: str) -> dict[str, Any]:
    """Time one pyspectral variant with the bench/run.py harness."""
    r = time_solver(name, n, dim, dtype, form, p["min_time"], p["repeats"])
    _free_gpu()
    gpu = not name.startswith("numpy")
    row = {"code": "pyspectral", "variant": name, "device": "gpu" if gpu else "cpu", "dim": dim,
           "n": n, "size": [n] * dim, "dtype": dtype, "form": r["form"],
           "stepper": "IF-RK4 (Lawson)", "stages_per_step": 4,
           "ffts_per_step": r["ffts_per_step"], "inner_steps": r["inner_steps"],
           "peak_mem_MB": r["peak_mem_MB"] if gpu else None, "finite": r["finite"],
           "notes": notes}
    row.update(_stats(r["times_s"], float(n) ** dim, r["ffts_per_step"]))
    return row


# ----------------------------------------------------------------------------- JAX-CFD
def run_jax(tasks: list[dict[str, Any]], x64: bool, p: dict[str, Any]) -> dict[str, Any]:
    """Run tasks in the JAX-CFD venv (one process per precision)."""
    with tempfile.TemporaryDirectory() as td:
        tin, tout = Path(td) / "tasks.json", Path(td) / "out.json"
        for t in tasks:
            if "ic" in t:
                np.save(Path(td) / f"ic{id(t)}.npy", t.pop("ic"))
                t["ic"] = str(Path(td) / f"ic{id(t)}.npy")
                t["out"] = str(Path(td) / f"out{id(t)}.npy")
        tin.write_text(json.dumps(tasks))
        env = dict(os.environ, XLA_PYTHON_CLIENT_PREALLOCATE="false",
                   XLA_PYTHON_CLIENT_MEM_FRACTION=p["jax_mem_fraction"])
        cmd = [str(JAX_PY), str(HERE / "cp_jaxcfd.py"), str(tin), str(tout)]
        if x64:
            cmd.append("--x64")
        pr = subprocess.run(cmd, env=env, capture_output=True, text=True, timeout=3600)
        if pr.returncode != 0 or not tout.exists():
            raise RuntimeError(f"JAX-CFD driver failed: {pr.stderr[-800:]}")
        doc = json.loads(tout.read_text())
        for r in doc["results"]:
            t = r["task"]
            if t["kind"] == "verify" and r.get("ok"):
                r["state"] = np.load(t["out"])
        return doc


JAX_NOTES = {
    "cn-rk4": "stock jax_cfd.spectral NavierStokes2D (advective form, 4 irfft + 1 rfft per "
              "stage) + time_stepping.crank_nicolson_rk4 (Carpenter-Kennedy low-storage RK4, "
              "5 stages, Crank-Nicolson viscous term); 10 steps per jitted lax.fori_loop",
    "if-rk4": "jax_cfd NavierStokes2D.explicit_terms inside the Lawson IF-RK4 used by "
              "pyspectral (4 stages, same algorithm as pyspectral's advective form); 10 steps "
              "per jitted lax.fori_loop",
}


def jax_section(p: dict[str, Any], doc: dict[str, Any]) -> None:
    """2D GPU: JAX-CFD timings + verification against the pyspectral NumPy oracle."""
    if not JAX_PY.exists():
        doc["skipped"].append({"code": "jax-cfd", "why": f"no interpreter at {JAX_PY}"})
        return
    # --- verification (fp64): exact 2D Taylor-Green decay and decaying turbulence
    nu_tg, dt_tg, st_tg = 0.01, 0.01, 100
    tg = initial.to_spectral(initial.taylor_green_2d(64), 2)
    nv, sv = p["verify2d_n"], p["verify2d_steps"]
    nu_t = 1e-3
    # well-resolved decaying turbulence: JAX-CFD's brick-wall filter differs from the 2/3 mask by
    # one row of modes at the kx truncation edge, which matters only if the edge is populated
    w0 = initial.to_spectral(initial.random_2d(nv, k0=4, seed=1), 2)
    dt_t = 0.25 * (2 * math.pi / nv) / 3.0
    vt = [{"kind": "verify", "n": 64, "nu": nu_tg, "dt": dt_tg, "nsteps": st_tg, "stepper": s,
           "ic": tg.copy(), "case": "tg2d"} for s in ("if-rk4", "cn-rk4")]
    vt += [{"kind": "verify", "n": nv, "nu": nu_t, "dt": dt_t, "nsteps": sv, "stepper": s,
            "ic": w0.copy(), "case": "turb2d"} for s in ("if-rk4", "cn-rk4")]
    jd = run_jax(vt, True, p)
    doc["versions"]["jax-cfd"] = jd["meta"] | {"commit": _git(JAX_SRC)}
    exact = tg * math.exp(-2 * nu_tg * dt_tg * st_tg)
    refs: dict[str, np.ndarray] = {}
    for form in ("advective", "basdevant"):
        s = NumpySolver(Config(n=nv, dim=2, nu=nu_t, dtype="float64", form=form))
        s.set_state(w0)
        s.advance(sv, dt_t)
        refs[form] = s.get_state()
    e_ref, z_ref = _diag2(nv, refs["advective"])
    e_b, z_b = _diag2(nv, refs["basdevant"])
    doc["verification"].append({
        "case": "turb2d", "code": "pyspectral", "variant": "numpy/scipy-fft basdevant",
        "ref": "pyspectral numpy/scipy-fft advective", "n": nv, "dtype": "float64",
        "nsteps": sv, "dt": dt_t, "nu": nu_t, "energy": e_b, "energy_ref": e_ref,
        "enstrophy": z_b, "enstrophy_ref": z_ref, "rel_diff_energy": _rel(e_b, e_ref),
        "rel_diff_enstrophy": _rel(z_b, z_ref),
        "rel_l2_state": float(np.linalg.norm(refs["basdevant"] - refs["advective"])
                              / np.linalg.norm(refs["advective"]))})
    for r in jd["results"]:
        t = r["task"]
        if not r.get("ok"):
            doc["skipped"].append({"code": "jax-cfd", "task": t, "why": r.get("error")})
            continue
        st = r["state"]
        if t["case"] == "tg2d":
            doc["verification"].append({
                "case": "tg2d", "code": "jax-cfd", "variant": t["stepper"],
                "ref": "exact exp(-2 nu t) decay", "n": 64, "dtype": "float64",
                "nsteps": st_tg, "dt": dt_tg, "nu": nu_tg,
                "rel_l2_state": float(np.linalg.norm(st - exact) / np.linalg.norm(exact))})
        else:
            e, z = _diag2(nv, st)
            ref = refs["advective"]
            doc["verification"].append({
                "case": "turb2d", "code": "jax-cfd", "variant": t["stepper"],
                "ref": "pyspectral numpy/scipy-fft advective", "n": nv, "dtype": "float64",
                "nsteps": sv, "dt": dt_t, "nu": nu_t, "energy": e, "energy_ref": e_ref,
                "enstrophy": z, "enstrophy_ref": z_ref, "rel_diff_energy": _rel(e, e_ref),
                "rel_diff_enstrophy": _rel(z, z_ref),
                "rel_l2_state": float(np.linalg.norm(st - ref) / np.linalg.norm(ref))})
    # --- timing
    for dtype in p["dtypes2d"]:
        sizes = [n for n in p["sizes2d"] if dtype == "float32" or n <= p.get("max_n_fp64", 0)]
        tasks = [{"kind": "time", "n": n, "nu": 1e-4, "dt": 0.2 * (2 * math.pi / n) / 4.0,
                  "stepper": s, "min_time": p["min_time"], "repeats": p["repeats"]}
                 for n in sizes for s in ("cn-rk4", "if-rk4")]
        jd = run_jax(tasks, dtype == "float64", p)
        for r in jd["results"]:
            t = r["task"]
            if not r.get("ok"):
                doc["skipped"].append({"code": "jax-cfd", "task": t, "why": r.get("error")})
                continue
            n = t["n"]
            row = {"code": "jax-cfd", "variant": t["stepper"], "device": "gpu", "dim": 2,
                   "n": n, "size": [n, n], "dtype": dtype, "form": "advective",
                   "stepper": ("CN-RK4 (Carpenter-Kennedy, low storage)"
                               if t["stepper"] == "cn-rk4" else "IF-RK4 (Lawson)"),
                   "stages_per_step": 5 if t["stepper"] == "cn-rk4" else 4,
                   "ffts_per_step": r["ffts_per_step"], "inner_steps": r["inner_steps"],
                   "compile_s": r["compile_s"], "peak_mem_MB": r["peak_mem_MB"],
                   "finite": r["finite"], "notes": JAX_NOTES[t["stepper"]]}
            row.update(_stats(r["times_s"], float(n) ** 2, r["ffts_per_step"]))
            doc["runs"].append(row)
            print(f"  jax-cfd/{t['stepper']:7s} 2D n={n:5d} {dtype}: {row['step_ms']:8.3f} ms",
                  flush=True)
        for n in sizes:
            for v in p["py2d"]:
                form = "advective" if "textbook" in v else "basdevant"
                note = ("same algorithm as jax-cfd/if-rk4 (advective, 20 FFTs/step), eager "
                        "PyTorch" if "textbook" in v else "pyspectral fused variant")
                try:
                    row = time_pyspectral(v, n, 2, dtype, form, p, note)
                except Exception as e:  # noqa: BLE001
                    _free_gpu()
                    doc["skipped"].append({"code": "pyspectral", "variant": v, "n": n,
                                           "dtype": dtype, "why": f"{type(e).__name__}: {e}"})
                    continue
                doc["runs"].append(row)
                print(f"  pyspectral/{v:28s} 2D n={n:5d} {dtype}: {row['step_ms']:8.3f} ms",
                      flush=True)


# ----------------------------------------------------------------------------- spectralDNS
def _tg3d_numpy(n: int, nsteps: int, dt: float, nu: float, integrator: str
                ) -> tuple[float, float]:
    s = NumpySolver(Config(n=n, dim=3, nu=nu, dtype="float64", form="rotational"),
                    integrator=integrator)
    s.set_state(initial.to_spectral(initial.taylor_green_3d(n), 3))
    s.advance(nsteps, dt)
    st = s.get_state()
    return s.diag.energy(st), s.diag.enstrophy(st)


def sdns_section(p: dict[str, Any], doc: dict[str, Any]) -> None:
    """3D CPU: spectralDNS (MPI) vs pyspectral NumPy on the same CPU, plus pyspectral GPU."""
    py = SDNS_ENV / "bin" / "python"
    mpiexec = SDNS_ENV / "bin" / "mpiexec"
    if not (py.exists() and mpiexec.exists() and SDNS_SRC.exists()):
        doc["skipped"].append({"code": "spectralDNS", "why": f"no env at {SDNS_ENV}"})
        return
    nu, dt = 1.0 / 1600.0, 0.01
    ref_cache: dict[int, tuple[float, float]] = {}
    for n, prec, ranks in p["sdns"]:
        verify = prec == "double" and n <= p["verify3d_nmax"]
        task = {"n": n, "precision": prec, "nu": nu, "dt": dt,
                "nsteps": p["sdns_verify_steps"] if verify else 2, "min_time": p["min_time"],
                "repeats": p["repeats"]}
        with tempfile.TemporaryDirectory() as td:
            tin, tout = Path(td) / "task.json", Path(td) / "out.json"
            tin.write_text(json.dumps(task))
            env = dict(os.environ, PYTHONPATH=str(SDNS_SRC), OMP_NUM_THREADS="1",
                       NUMBA_NUM_THREADS="1", PATH=f"{SDNS_ENV / 'bin'}:{os.environ['PATH']}")
            cmd = [str(mpiexec), "-n", str(ranks), str(py), str(HERE / "cp_spectraldns.py"),
                   str(tin), str(tout)]
            pr = subprocess.run(cmd, env=env, capture_output=True, text=True, timeout=3600,
                                cwd=td)
            if pr.returncode != 0 or not tout.exists():
                doc["skipped"].append({"code": "spectralDNS", "task": task, "ranks": ranks,
                                       "why": pr.stderr[-600:]})
                continue
            r = json.loads(tout.read_text())
        doc["versions"]["spectralDNS"] = r["meta"] | {"commit": _git(SDNS_SRC)}
        dtype = "float64" if prec == "double" else "float32"
        row = {"code": "spectralDNS", "variant": f"rk4-vortex/mpi{ranks}", "device": "cpu",
               "dim": 3, "n": n, "size": [n] * 3, "dtype": dtype, "form": "rotational",
               "stepper": "RK4 (classical, explicit viscous term)", "stages_per_step": 4,
               "ffts_per_step": 36, "inner_steps": r["inner_steps"], "mpi_ranks": ranks,
               "finite": r["finite"], "peak_mem_MB": None,
               "notes": "spectralDNS NS solver, convection=Vortex (u x omega), RK4, 2/3-rule, "
                        f"slab decomposition on {ranks} MPI ranks, mpi4py-fft/FFTW "
                        "FFTW_MEASURE, numba optimisation"}
        row.update(_stats(r["times_s"], float(n) ** 3, 36))
        doc["runs"].append(row)
        print(f"  spectralDNS mpi{ranks} 3D n={n:4d} {dtype}: {row['step_ms']:9.2f} ms",
              flush=True)
        if verify:
            if n not in ref_cache:
                ref_cache[n] = _tg3d_numpy(n, r["nsteps"], dt, nu, "rk4")
            e_ref, z_ref = ref_cache[n]
            doc["verification"].append({
                "case": "tgv3d", "code": "spectralDNS", "variant": f"rk4-vortex/mpi{ranks}",
                "ref": "pyspectral numpy/scipy-fft rotational, explicit RK4", "n": n,
                "dtype": "float64", "nsteps": r["nsteps"], "dt": dt, "nu": nu,
                "energy": r["energy"], "energy_ref": e_ref, "enstrophy": r["enstrophy"],
                "enstrophy_ref": z_ref, "rel_diff_energy": _rel(r["energy"], e_ref),
                "rel_diff_enstrophy": _rel(r["enstrophy"], z_ref)})
    for n, dtype in p["py3d_cpu"]:
        row = time_pyspectral("numpy/scipy-fft", n, 3, dtype, "rotational", p,
                              "pyspectral NumPy oracle, scipy.fft on all 16 CPU threads, "
                              "rotational form (same FFT count as spectralDNS Vortex)")
        doc["runs"].append(row)
        print(f"  pyspectral/numpy 3D n={n:4d} {dtype}: {row['step_ms']:9.2f} ms", flush=True)
    for n, dtype in p["py3d_gpu"]:
        for form in ("rotational", "basdevant"):
            try:
                row = time_pyspectral("cupy-cuda-c/fused-inplace", n, 3, dtype, form, p,
                                      "pyspectral fastest GPU variant")
            except Exception as e:  # noqa: BLE001
                _free_gpu()
                doc["skipped"].append({"code": "pyspectral", "n": n, "dtype": dtype,
                                       "form": form, "why": f"{type(e).__name__}: {e}"})
                continue
            doc["runs"].append(row)
            print(f"  pyspectral/cupy-cuda-c {form} 3D n={n:4d} {dtype}: "
                  f"{row['step_ms']:9.2f} ms", flush=True)


# ----------------------------------------------------------------------------- Dedalus
def dedalus_section(p: dict[str, Any], doc: dict[str, Any]) -> None:
    """3D GPU: Dedalus (gpu branch, CuPy) Taylor-Green vortex, if its env exists."""
    drv = HERE / "cp_dedalus.py"
    if not (DED_PY.exists() and drv.exists()):
        doc["skipped"].append({"code": "dedalus", "why": f"no interpreter at {DED_PY}"})
        return
    nu, dt = 1.0 / 1600.0, 0.01
    with tempfile.TemporaryDirectory() as td:
        tasks = [{"n": n, "dtype": d, "nu": nu, "dt": dt, "nsteps": p["ded_verify_steps"],
                  "min_time": p["min_time"], "repeats": p["repeats"]} for n, d in p["ded"]]
        tin, tout = Path(td) / "tasks.json", Path(td) / "out.json"
        tin.write_text(json.dumps(tasks))
        env = dict(os.environ, OMP_NUM_THREADS="1", PATH=f"{DED_PY.parent}:{os.environ['PATH']}")
        pr = subprocess.run([str(DED_PY), str(drv), str(tin), str(tout)], env=env,
                            capture_output=True, text=True, timeout=3600, cwd=td)
        if pr.returncode != 0 or not tout.exists():
            doc["skipped"].append({"code": "dedalus", "why": pr.stderr[-800:]})
            return
        dd = json.loads(tout.read_text())
    doc["versions"]["dedalus"] = dd["meta"] | {"commit": _git(DED_SRC)}
    ref_cache: dict[int, tuple[float, float]] = {}
    for r in dd["results"]:
        t = r["task"]
        if not r.get("ok"):
            doc["skipped"].append({"code": "dedalus", "task": t, "why": r.get("error")})
            continue
        n = t["n"]
        row = {"code": "dedalus", "variant": "rk443-cupy", "device": "gpu", "dim": 3, "n": n,
               "size": [n] * 3, "dtype": t["dtype"], "form": r["form"],
               "stepper": r["stepper"], "stages_per_step": r["stages_per_step"],
               "ffts_per_step": None, "inner_steps": r["inner_steps"],
               "peak_mem_MB": r.get("peak_mem_MB"), "finite": r["finite"], "notes": r["notes"]}
        row.update(_stats(r["times_s"], float(n) ** 3, 0))
        row["fft_Gpts_per_s"] = None
        doc["runs"].append(row)
        print(f"  dedalus 3D n={n:4d} {t['dtype']}: {row['step_ms']:9.2f} ms", flush=True)
        if t["dtype"] == "float64":
            if n not in ref_cache:
                ref_cache[n] = _tg3d_numpy(n, t["nsteps"], dt, nu, "ifrk4")
            e_ref, z_ref = ref_cache[n]
            doc["verification"].append({
                "case": "tgv3d", "code": "dedalus", "variant": "rk443-cupy",
                "ref": "pyspectral numpy/scipy-fft rotational, IF-RK4", "n": n,
                "dtype": "float64", "nsteps": t["nsteps"], "dt": dt, "nu": nu,
                "energy": r["energy"], "energy_ref": e_ref, "enstrophy": r["enstrophy"],
                "enstrophy_ref": z_ref, "rel_diff_energy": _rel(r["energy"], e_ref),
                "rel_diff_enstrophy": _rel(r["enstrophy"], z_ref)})


# ----------------------------------------------------------------------------- main
def run(mode: str, out: Path) -> dict[str, Any]:
    """Run every section and write the results document."""
    p = plan(mode)
    t0 = time.perf_counter()
    doc: dict[str, Any] = {"machine": machine_info(), "mode": mode,
                           "provisional": mode != "full", "runs": [], "verification": [],
                           "skipped": [], "versions": {}}
    doc["machine"]["cpu_model"] = _cpu_model()
    doc["machine"]["logical_cpus"] = os.cpu_count()
    doc["machine"]["platform"] = platform.platform()
    if mode != "full":
        doc["provisional_reason"] = "quick run on a shared GPU/CPU; not publishable numbers"
    for name, fn in (("jax-cfd", jax_section), ("spectralDNS", sdns_section),
                     ("dedalus", dedalus_section)):
        print(f"[{name}]", flush=True)
        try:
            fn(p, doc)
        except Exception as e:  # noqa: BLE001
            doc["skipped"].append({"code": name, "why": f"{type(e).__name__}: {str(e)[:800]}"})
            print(f"  FAILED: {e!s:.300}", flush=True)
        for r in doc["runs"]:  # one RK4-equivalent step = 4 right-hand-side evaluations
            r.setdefault("rk4_equiv_ms", r["step_ms"] * 4 / r["stages_per_step"])
        out.write_text(json.dumps(doc, indent=1) + "\n")
    doc["wall_s"] = time.perf_counter() - t0
    out.write_text(json.dumps(doc, indent=1) + "\n")
    return doc


def _cpu_model() -> str:
    try:
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.startswith("model name"):
                return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor()


def main(argv: list[str] | None = None) -> int:
    """CLI."""
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    g = ap.add_mutually_exclusive_group()
    g.add_argument("--quick", action="store_true", help="< 60 s smoke run (shared GPU ok)")
    g.add_argument("--full", action="store_true", help="publishable run (idle GPU and CPU)")
    ap.add_argument("--out", default=str(ROOT / "results" / "counterparts.json"))
    a = ap.parse_args(argv)
    mode = "full" if a.full else "quick"
    doc = run(mode, Path(a.out))
    print(f"wrote {a.out} ({len(doc['runs'])} runs, {len(doc['verification'])} checks, "
          f"{len(doc['skipped'])} skipped, {doc['wall_s']:.0f} s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())

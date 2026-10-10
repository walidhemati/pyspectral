"""JAX-CFD counterpart driver (runs in its own venv, NOT in pyhpc; see bench/counterparts.py).

Usage: ``python cp_jaxcfd.py tasks.json out.json [--x64]``. Each task is a dict with
``kind`` ("time" | "verify"), ``n``, ``nu``, ``dt``, ``stepper`` ("cn-rk4" | "if-rk4") and,
for "verify", ``ic`` (path of a complex .npy spectral vorticity in the unnormalised
``numpy.fft.rfftn`` convention, axes (x, y)), ``nsteps`` and ``out`` (.npy for the final state).

Physics: ``jax_cfd.spectral.equations.NavierStokes2D`` (vorticity form, advective nonlinear term
with 4 inverse + 1 forward real FFTs, 2/3-rule brick-wall filter on the nonlinear term).

* ``cn-rk4``: the stock JAX-CFD stepper ``crank_nicolson_rk4`` (Carpenter-Kennedy low-storage
  RK4, 5 explicit stages, Crank-Nicolson for the viscous term): 25 FFTs per step.
* ``if-rk4``: JAX-CFD's ``explicit_terms`` inside the Lawson integrating-factor RK4 that
  pyspectral uses (4 stages): 20 FFTs per step, identical algorithm to pyspectral's
  ``advective`` form.
"""

from __future__ import annotations

import os

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.12")

import json  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402

X64 = "--x64" in sys.argv
import jax  # noqa: E402

jax.config.update("jax_enable_x64", X64)
import jax.numpy as jnp  # noqa: E402
from jax_cfd.base import grids  # noqa: E402
from jax_cfd.spectral import equations, time_stepping  # noqa: E402

INNER = 10  # time steps per jitted call (lax.fori_loop), the way JAX-CFD trajectories run


def make_step(n: int, nu: float, dt: float, stepper: str):
    """Return (step_fn, ffts_per_step)."""
    grid = grids.Grid((n, n), domain=((0, 2 * np.pi), (0, 2 * np.pi)))
    eq = equations.NavierStokes2D(nu, grid, smooth=True)
    if stepper == "cn-rk4":
        return time_stepping.crank_nicolson_rk4(eq, dt), 25
    if stepper != "if-rk4":
        raise ValueError(stepper)
    L = eq.linear_term
    E = jnp.exp(L * dt)
    Eh = jnp.exp(L * (0.5 * dt))
    F = eq.explicit_terms

    def step(u0):
        k1 = F(u0)
        k2 = F(Eh * (u0 + 0.5 * dt * k1))
        k3 = F(Eh * u0 + 0.5 * dt * k2)
        k4 = F(E * u0 + dt * Eh * k3)
        return E * u0 + dt / 6.0 * (E * k1 + 2.0 * Eh * (k2 + k3) + k4)

    return step, 20


def _peak_mb() -> float | None:
    try:
        st = jax.devices()[0].memory_stats() or {}
        return st.get("peak_bytes_in_use", 0) / 1e6
    except Exception:  # noqa: BLE001
        return None


def run_task(t: dict) -> dict:
    n, nu, dt, stepper = int(t["n"]), float(t["nu"]), float(t["dt"]), t["stepper"]
    rdt = np.float64 if X64 else np.float32
    cdt = np.complex128 if X64 else np.complex64
    step, nfft = make_step(n, nu, dt, stepper)
    if t["kind"] == "verify":
        w0 = jnp.asarray(np.load(t["ic"]).astype(cdt))
        f = jax.jit(lambda x: jax.lax.fori_loop(0, int(t["nsteps"]), lambda i, y: step(y), x))
        w = np.asarray(jax.block_until_ready(f(w0)))
        np.save(t["out"], w.astype(np.complex128))
        return {"ok": True, "dtype": str(np.dtype(rdt)), "ffts_per_step": nfft}
    # timing: random smooth field (the step cost does not depend on the data)
    rng = np.random.default_rng(1)
    w = np.fft.rfftn(rng.standard_normal((n, n))).astype(cdt)
    k = np.fft.fftfreq(n, 1.0 / n)
    w = w * (np.abs(k)[:, None] < n / 8) * (np.arange(n // 2 + 1)[None, :] < n / 8)
    x = jnp.asarray(w)
    f = jax.jit(lambda y: jax.lax.fori_loop(0, INNER, lambda i, z: step(z), y))
    t0 = time.perf_counter()
    x = jax.block_until_ready(f(x))
    compile_s = time.perf_counter() - t0
    min_time, repeats = float(t["min_time"]), int(t["repeats"])
    calls = 1
    while True:
        t0 = time.perf_counter()
        for _ in range(calls):
            x = f(x)
        x = jax.block_until_ready(x)
        el = time.perf_counter() - t0
        if el >= 0.5 * min_time or calls >= 1 << 14:
            break
        calls *= 2
    calls = max(1, int(np.ceil(calls * min_time / max(el, 1e-9))))
    times = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        for _ in range(calls):
            x = f(x)
        x = jax.block_until_ready(x)
        times.append((time.perf_counter() - t0) / (calls * INNER))
    finite = bool(np.isfinite(np.asarray(x)).all())
    return {"ok": True, "times_s": times, "inner_steps": calls * INNER, "compile_s": compile_s,
            "ffts_per_step": nfft, "dtype": str(np.dtype(rdt)), "finite": finite,
            "peak_mem_MB": _peak_mb()}


def main() -> int:
    tasks = json.loads(Path(sys.argv[1]).read_text())
    out = []
    for t in tasks:
        try:
            r = run_task(t)
        except Exception as e:  # noqa: BLE001
            r = {"ok": False, "error": f"{type(e).__name__}: {str(e)[:300]}"}
        r["task"] = t
        out.append(r)
    import jax_cfd

    meta = {"jax": jax.__version__, "jaxlib": jax.lib.__version__ if hasattr(jax, "lib") else None,
            "jax_cfd": getattr(jax_cfd, "__version__", "0.2.1"),
            "device": str(jax.devices()[0].device_kind), "x64": X64,
            "python": sys.version.split()[0]}
    with open(sys.argv[2], "w") as fh:
        json.dump({"meta": meta, "results": out}, fh, indent=1)
    return 0


if __name__ == "__main__":
    sys.exit(main())

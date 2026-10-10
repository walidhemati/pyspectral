"""Dedalus counterpart driver (``gpu`` branch, CuPy; runs in its own conda env, see
bench/counterparts.py).

Usage: ``python cp_dedalus.py tasks.json out.json``. Each task holds ``n`` (the physical grid
of the pyspectral run it is compared with), ``dtype``, ``nu``, ``dt``, ``nsteps``,
``min_time`` and ``repeats``.

Configuration: 3D incompressible Navier-Stokes on ``RealFourier`` bases (box ``[0, 2 pi)^3``)
in advective form ``dt(u) + grad(P) - nu lap(u) = -u@grad(u)``, ``div(u) + tau = 0``,
``integ(P) = 0`` (the form of the Dedalus examples; ``cross`` still goes through numexpr on the
host in the gpu branch). Dedalus keeps ``M = 2n/3`` modes per axis and evaluates the products on
the ``3/2``-padded grid of ``n`` points, i.e. exactly the modes and the product grid of
pyspectral's 2/3 rule on an ``n``-point grid. Time stepper ``RK443`` (4-stage, 3rd-order IMEX
Runge-Kutta; Dedalus has no classical RK4), constant ``dt`` (no matrix refactorisation).
"""

from __future__ import annotations

import os

os.environ.setdefault("OMP_NUM_THREADS", "1")

import json  # noqa: E402
import logging  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

import cupy as cp  # noqa: E402
import numpy as np  # noqa: E402

logging.disable(logging.WARNING)
import dedalus  # noqa: E402
import dedalus.public as d3  # noqa: E402


def run_task(t: dict) -> dict:
    n = int(t["n"])
    if (2 * n) % 3:
        raise ValueError("n must be a multiple of 3 (Dedalus keeps M = 2n/3 modes)")
    M = 2 * n // 3
    dtype = np.float64 if t["dtype"] == "float64" else np.float32
    nu, dt = float(t["nu"]), float(t["dt"])
    coords = d3.CartesianCoordinates("x", "y", "z")
    dist = d3.Distributor(coords, dtype=dtype, array_namespace=cp)
    bases = tuple(d3.RealFourier(coords[c], size=M, bounds=(0, 2 * np.pi), dealias=3 / 2)
                  for c in "xyz")
    u = dist.VectorField(coords, name="u", bases=bases)
    P = dist.Field(name="P", bases=bases)
    tau = dist.Field(name="tau")
    problem = d3.IVP([u, P, tau], namespace={"u": u, "P": P, "tau": tau, "nu": nu,
                                              "dt": d3.dt, "grad": d3.grad, "lap": d3.lap,
                                              "div": d3.div, "cross": d3.cross,
                                              "curl": d3.curl, "integ": d3.integ})
    problem.add_equation("dt(u) + grad(P) - nu*lap(u) = -u@grad(u)")
    problem.add_equation("div(u) + tau = 0")
    problem.add_equation("integ(P) = 0")
    t0 = time.perf_counter()
    solver = problem.build_solver(d3.RK443)
    build_s = time.perf_counter() - t0
    x, y, z = dist.local_grids(*bases)
    u["g"][0] = cp.sin(x) * cp.cos(y) * cp.cos(z)
    u["g"][1] = -cp.cos(x) * cp.sin(y) * cp.cos(z)
    u["g"][2] = 0
    vol = (2 * np.pi) ** 3
    e_op = d3.integ(u @ u) / (2 * vol)
    w = d3.curl(u)
    z_op = d3.integ(w @ w) / (2 * vol)

    def diag() -> tuple[float, float]:
        return (float(cp.asarray(e_op.evaluate()["g"]).ravel()[0]),
                float(cp.asarray(z_op.evaluate()["g"]).ravel()[0]))

    e0, z0 = diag()
    t0 = time.perf_counter()
    for _ in range(int(t["nsteps"])):
        solver.step(dt)
    cp.cuda.Device().synchronize()
    first_s = time.perf_counter() - t0
    e1, z1 = diag()

    def run(k: int) -> float:
        cp.cuda.Device().synchronize()
        t1 = time.perf_counter()
        for _ in range(k):
            solver.step(dt)
        cp.cuda.Device().synchronize()
        return time.perf_counter() - t1

    min_time, repeats = float(t["min_time"]), int(t["repeats"])
    k = 1
    while True:
        el = run(k)
        if el >= 0.5 * min_time or k >= 1 << 12:
            break
        k *= 2
    k = max(1, int(np.ceil(k * min_time / max(el, 1e-9))))
    times = [run(k) / k for _ in range(repeats)]
    finite = bool(cp.isfinite(u["c"]).all())
    pool = cp.get_default_memory_pool()
    return {"ok": True, "times_s": times, "inner_steps": k, "build_s": build_s,
            "first_steps_s": first_s, "energy0": e0, "enstrophy0": z0, "energy": e1,
            "enstrophy": z1, "finite": finite, "modes": M,
            "peak_mem_MB": pool.total_bytes() / 1e6, "form": "advective",
            "stepper": "RK443 (4-stage 3rd-order IMEX RK)", "stages_per_step": 4,
            "notes": f"Dedalus gpu branch, CuPy, RealFourier x3 with M={M} modes and 3/2 "
                     f"dealiasing (= 2/3 rule on {n}^3), -u.grad(u), RK443, constant dt"}


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
        cp.get_default_memory_pool().free_all_blocks()
    meta = {"dedalus": dedalus.__version__, "cupy": cp.__version__,
            "cuda_runtime": cp.cuda.runtime.runtimeGetVersion(), "numpy": np.__version__,
            "python": sys.version.split()[0]}
    with open(sys.argv[2], "w") as fh:
        json.dump({"meta": meta, "results": out}, fh, indent=1)
    return 0


if __name__ == "__main__":
    sys.exit(main())

"""spectralDNS counterpart driver (runs under ``mpiexec`` in its own conda env; see
bench/counterparts.py).

Usage: ``mpiexec -n P python cp_spectraldns.py task.json out.json``. The task dict holds ``n``,
``precision`` ("single" | "double"), ``nu``, ``dt``, ``nsteps`` (verification steps from the
3D Taylor-Green vortex), ``min_time`` and ``repeats``. One launch = one grid size.

Configuration: the triply periodic ``NS`` solver of spectralDNS (Mortensen & Langtangen) with
``convection=Vortex`` (rotational form u x omega: 6 inverse + 3 forward real FFTs per stage),
``integrator=RK4`` (classical RK4, viscous term explicit), ``dealias=2/3-rule``, slab
decomposition over P MPI ranks, FFTW via mpi4py-fft with ``FFTW_MEASURE`` plans, and the
``numba`` optimisation module (the Cython modules do not build against NumPy 2 headers).
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np


def main() -> int:
    task = json.loads(Path(sys.argv[1]).read_text())
    n = int(task["n"])
    from spectralDNS import config, get_solver

    config.update({"nu": float(task["nu"]), "dt": float(task["dt"]), "T": 1e9,
                   "L": [2 * np.pi] * 3,
                   "planner_effort": {"fft": "FFTW_MEASURE", "rfftn": "FFTW_MEASURE",
                                      "irfftn": "FFTW_MEASURE"}}, "triplyperiodic")
    # as in spectralDNS/demo/TG.py: --N trumps the power-of-two exponents --M
    config.triplyperiodic.add_argument("--N", default=[n, n, n], nargs=3)
    args = ["--N", str(n), str(n), str(n), "--precision", task["precision"],
            "--optimization", "numba", "--dealias", "2/3-rule", "--decomposition", "slab",
            "--threads", "1", "--convection", "Vortex", "--integrator", "RK4", "NS"]
    sol = get_solver(mesh="triplyperiodic", parse_args=args)
    ctx = sol.get_context()
    comm = sol.comm
    rank = comm.Get_rank()
    U, X = ctx.U, ctx.X
    U[0] = np.sin(X[0]) * np.cos(X[1]) * np.cos(X[2])
    U[1] = -np.cos(X[0]) * np.sin(X[1]) * np.cos(X[2])
    U[2] = 0
    sol.set_velocity(**ctx)
    sol.conv = sol.getConvection(sol.params.convection)
    integrate = sol.getintegrator(ctx.dU, ctx.u, sol, ctx)

    def diag() -> tuple[float, float]:
        u = sol.get_velocity(**ctx).astype(np.float64)
        w = sol.get_curl(**ctx).astype(np.float64)
        e = comm.allreduce(float(np.sum(u * u))) / n**3 / 2
        z = comm.allreduce(float(np.sum(w * w))) / n**3 / 2
        return e, z

    e0, z0 = diag()
    t0 = time.perf_counter()
    for _ in range(int(task["nsteps"])):
        integrate()
    comm.Barrier()
    first_s = time.perf_counter() - t0
    e1, z1 = diag()

    def run(k: int) -> float:
        comm.Barrier()
        t = time.perf_counter()
        for _ in range(k):
            integrate()
        comm.Barrier()
        return comm.allreduce(time.perf_counter() - t, op=_max())

    min_time, repeats = float(task["min_time"]), int(task["repeats"])
    k = 1
    while True:
        el = run(k)
        if el >= 0.5 * min_time or k >= 1 << 12:
            break
        k *= 2
    k = max(1, int(np.ceil(k * min_time / max(el, 1e-9))))
    times = [run(k) / k for _ in range(repeats)]
    finite = bool(comm.allreduce(int(np.isfinite(ctx.U_hat).all()), op=_min()))
    if rank == 0:
        import mpi4py
        import mpi4py_fft
        import shenfun
        import spectralDNS

        meta = {"spectralDNS": spectralDNS.__version__, "shenfun": shenfun.__version__,
                "mpi4py_fft": mpi4py_fft.__version__, "mpi4py": mpi4py.__version__,
                "numpy": np.__version__, "mpi_ranks": comm.Get_size(),
                "mpi_library": mpi4py.MPI.Get_library_version().splitlines()[0].strip(),
                "python": sys.version.split()[0]}
        out = {"ok": True, "meta": meta, "task": task, "times_s": times, "inner_steps": k,
               "ffts_per_step": 36, "energy0": e0, "enstrophy0": z0, "energy": e1,
               "enstrophy": z1, "nsteps": int(task["nsteps"]), "first_steps_s": first_s,
               "finite": finite}
        with open(sys.argv[2], "w") as fh:
            json.dump(out, fh, indent=1)
    return 0


def _max():  # noqa: ANN202
    from mpi4py import MPI

    return MPI.MAX


def _min():  # noqa: ANN202
    from mpi4py import MPI

    return MPI.MIN


if __name__ == "__main__":
    sys.exit(main())

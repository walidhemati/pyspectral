"""Command line interface: ``pyspectral list|run|validate|bench``."""

from __future__ import annotations

import argparse
import sys
import time
import warnings
from pathlib import Path

import numpy as np

from pyspectral import initial
from pyspectral.backends import REGISTRY, available, make_solver
from pyspectral.config import Config


def _cmd_list(_: argparse.Namespace) -> int:
    for v, ok, why in available():
        flag = "ok " if ok else "-- "
        print(f"{flag} {v.name:36s} {v.description}" + ("" if ok else f"  [{why}]"))
    return 0


def _initial(case: str, n: int, dim: int, seed: int) -> np.ndarray:
    if dim == 2:
        f = {"tgv": initial.taylor_green_2d, "smooth": initial.smooth_2d,
             "random": lambda m: initial.random_2d(m, seed=seed)}[case](n)
    else:
        f = {"tgv": initial.taylor_green_3d,
             "random": lambda m: initial.random_3d(m, seed=seed)}[case](n)
    return initial.to_spectral(f, dim)


def _cmd_run(a: argparse.Namespace) -> int:
    from pyspectral.simulate import run

    cfg = Config(n=a.n, dim=a.dim, nu=a.nu, alpha=a.alpha, dtype=a.dtype, form=a.form)
    s = make_solver(a.backend, cfg)
    s.set_state(_initial(a.case, a.n, a.dim, a.seed))
    print(s.describe())
    res = run(s, a.t_end, dt=a.dt, cfl=a.cfl, diag_dt=a.t_end / 10)
    for t, e, z in zip(res.t, res.energy, res.enstrophy, strict=True):
        print(f"t={t:8.4f}  E={e:.8e}  Z={z:.8e}")
    ms = 1e3 * res.wall / max(res.steps, 1)
    print(f"{res.steps} steps in {res.wall:.2f} s ({ms:.3f} ms/step)")
    if a.out:
        np.savez(a.out, state=s.get_state(), t=s.t)
    return 0


def _cmd_validate(a: argparse.Namespace) -> int:
    from pyspectral import validation as V

    out = Path(a.out_dir)
    what = set(a.what.split(","))
    allw = "all" in what
    if allw or "2d" in what:
        t0 = time.perf_counter()
        tgv = []
        for v in REGISTRY:
            for dtype in ("float64", "float32"):
                try:
                    err = V.tgv2d_error(v.name, dtype=dtype)
                except Exception as e:  # noqa: BLE001
                    print(f"skip {v.name} {dtype}: {e}")
                    continue
                tgv.append({"variant": v.name, "dtype": dtype, "rel_err": err})
                print(f"TGV2D {v.name:36s} {dtype}: rel err {err:.2e}")
        res = {"tgv2d": tgv, "rk4_order": V.tgv2d_rk4_order()}
        if not a.quick:
            res["temporal"] = V.temporal_convergence()
            res["spatial"] = V.spatial_convergence()
            print("temporal order", res["temporal"]["order"])
            print("spatial", res["spatial"]["err"])
        res["wall_s"] = time.perf_counter() - t0
        V.write_json(res, out / "validation_2d.json")
    if allw or "backends" in what:
        rows = V.backend_agreement()
        for r in rows:
            print(f"{r['variant']:36s} {r['dim']}D {r['form']:10s} {r['dtype']}: "
                  f"{r['rel_err']:.2e}")
        V.write_json(rows, out / "validation_backends.json")
    if "tgv3d" in what:
        r = V.tgv3d(a.backend, n=a.n, dtype=a.dtype, checkpoint=a.checkpoint,
                    wall_limit=a.wall_limit)
        if not r["finished"]:
            print(f"checkpointed at t={r['t']:.3f}; rerun the same command to resume")
            return 3
        print({k: v for k, v in r.items() if not isinstance(v, list)})
        V.write_json(r, out / f"validation_tgv3d_n{a.n}.json")
    if "forced2d" in what:
        r = V.forced(a.backend, 2, n=a.n, nu=a.nu, alpha=a.alpha, klo=a.klo, khi=a.khi,
                     power=a.power, t_end=a.t_end, t_avg=a.t_avg, fit_range=(a.kfit0, a.kfit1),
                     alpha_kmax=a.alpha_kmax, checkpoint=a.checkpoint, wall_limit=a.wall_limit)
        if not r["finished"]:
            print(f"checkpointed at t={r['t']:.3f}; rerun the same command to resume")
            return 3
        print({k: v for k, v in r.items() if not isinstance(v, list)})
        V.write_json(r, out / "validation_forced2d.json")
    if "forced3d" in what:
        r = V.forced(a.backend, 3, n=a.n, nu=a.nu, alpha=0.0, klo=a.klo, khi=a.khi,
                     power=a.power, t_end=a.t_end, t_avg=a.t_avg, fit_range=(a.kfit0, a.kfit1),
                     checkpoint=a.checkpoint, wall_limit=a.wall_limit)
        if not r["finished"]:
            print(f"checkpointed at t={r['t']:.3f}; rerun the same command to resume")
            return 3
        print({k: v for k, v in r.items() if not isinstance(v, list)})
        V.write_json(r, out / "validation_forced3d.json")
    return 0


def _cmd_bench(a: argparse.Namespace) -> int:
    from pyspectral import benchmark

    return benchmark.main(["--quick"] if a.quick else ["--full"])


def main(argv: list[str] | None = None) -> int:
    """Entry point."""
    warnings.filterwarnings("ignore", message=".*CuPy.*")
    p = argparse.ArgumentParser(prog="pyspectral", description=__doc__)
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("list", help="list backends and availability").set_defaults(fn=_cmd_list)

    r = sub.add_parser("run", help="run a simulation and print diagnostics")
    r.add_argument("--backend", default="numpy")
    r.add_argument("--dim", type=int, default=2)
    r.add_argument("--n", type=int, default=128)
    r.add_argument("--nu", type=float, default=1e-3)
    r.add_argument("--alpha", type=float, default=0.0)
    r.add_argument("--dtype", default="float64")
    r.add_argument("--form", default="auto")
    r.add_argument("--case", default="random", choices=["tgv", "smooth", "random"])
    r.add_argument("--seed", type=int, default=0)
    r.add_argument("--t-end", type=float, default=1.0)
    r.add_argument("--dt", type=float, default=None)
    r.add_argument("--cfl", type=float, default=0.5)
    r.add_argument("--out", default=None)
    r.set_defaults(fn=_cmd_run)

    v = sub.add_parser("validate", help="run validation experiments, write results/*.json")
    v.add_argument("--what", default="2d,backends",
                   help="comma list: 2d, backends, tgv3d, forced2d, forced3d, all")
    v.add_argument("--quick", action="store_true")
    v.add_argument("--out-dir", default="results")
    v.add_argument("--backend", default="cupy-cuda-c/fused-inplace")
    v.add_argument("--n", type=int, default=128)
    v.add_argument("--dtype", default="float32")
    v.add_argument("--nu", type=float, default=1e-3)
    v.add_argument("--alpha", type=float, default=0.0)
    v.add_argument("--alpha-kmax", type=float, default=0.0)
    v.add_argument("--klo", type=float, default=1.0)
    v.add_argument("--khi", type=float, default=2.5)
    v.add_argument("--power", type=float, default=0.1)
    v.add_argument("--t-end", type=float, default=20.0)
    v.add_argument("--t-avg", type=float, default=10.0)
    v.add_argument("--kfit0", type=float, default=4.0)
    v.add_argument("--kfit1", type=float, default=16.0)
    v.add_argument("--checkpoint", default=None)
    v.add_argument("--wall-limit", type=float, default=None)
    v.set_defaults(fn=_cmd_validate)

    b = sub.add_parser("bench", help="run the benchmark (same as bench/run.py)")
    g = b.add_mutually_exclusive_group()
    g.add_argument("--quick", action="store_true")
    g.add_argument("--full", action="store_true")
    b.set_defaults(fn=_cmd_bench)

    a = p.parse_args(argv)
    return int(a.fn(a))


if __name__ == "__main__":
    sys.exit(main())


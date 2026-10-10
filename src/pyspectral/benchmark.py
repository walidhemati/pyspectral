"""Benchmark harness (the logic behind ``bench/run.py``).

Measures, per backend variant / grid size / precision, the wall time of one full IF-RK4 time
step, and the hardware ceilings it is compared against:

* device copy bandwidth (``torch`` ``copy_``; read + write counted),
* the standalone cuFFT time for exactly the batched transforms one RK stage needs
  (in-place CuPy plans called from Python and replayed from a CUDA graph, and out-of-place
  ``torch.fft``; the fastest is the FFT ceiling),
* the "FFT roofline" step time ``4 * t_fft_stage + pointwise_bytes / B_copy``.

Traffic model (minimal, stated in the paper): every FFT reads its input and writes its output
exactly once; the physical-space kernel reads the inverse-FFT outputs and writes the products
once; the spectral kernel reads the forward-FFT outputs, the state and the RK accumulator and
writes the accumulator (stages 1-3), the new state (stage 4) and the next stage's FFT inputs.
"""

from __future__ import annotations

import argparse
import datetime as _dt
import gc
import json
import math
import platform
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

from pyspectral import initial
from pyspectral.backends import REGISTRY, BackendUnavailable, get
from pyspectral.config import Config, fft_split
from pyspectral.grid import spectral_shape

SPEC_COPY_BW = 361.0  # GB/s, suite-wide measured ceiling (SPEC)
SPEC_BW = 448.0  # GB/s, RTX 3070 Laptop datasheet


# ----------------------------------------------------------------------------- traffic model
def traffic_model(n: int, dim: int, form: str, dtype: str) -> dict[str, float]:
    """Minimal bytes moved per time step (4 RK stages) by the fused schedule."""
    sr = 8 if dtype == "float64" else 4
    sc = 2 * sr
    npts = float(n) ** dim
    nmodes = float(np.prod(spectral_shape(n, dim)))
    ni, nf = fft_split(dim, form)
    ncomp = 1 if dim == 2 else 3
    fft = 4 * (ni + nf) * (nmodes * sc + npts * sr)
    phys = 4 * (ni + nf) * npts * sr
    post = (4 * (nf + ncomp + ni) + 7 * ncomp) * nmodes * sc
    return {"fft_bytes": fft, "pointwise_bytes": phys + post, "total_bytes": fft + phys + post,
            "points": npts, "ffts_per_step": 4 * (ni + nf)}


# ----------------------------------------------------------------------------- machine
def machine_info() -> dict[str, Any]:
    """Versions and device name."""
    info: dict[str, Any] = {"python": platform.python_version(),
                            "date": _dt.datetime.now().astimezone().isoformat(timespec="seconds"),
                            "cpu": platform.processor() or platform.machine(),
                            "numpy": np.__version__}
    for mod in ("torch", "triton", "warp", "numba", "scipy"):
        try:
            m = __import__(mod)
            info[mod] = getattr(m, "__version__", "?")
        except Exception:  # noqa: BLE001
            info[mod] = None
    try:
        from pyspectral.backends._cuda_env import import_cufft, import_cupy

        cp = import_cupy()
        info["cupy"] = cp.__version__
        info["cufft"] = import_cufft().getVersion()
        info["driver"] = cp.cuda.runtime.driverGetVersion()
    except Exception:  # noqa: BLE001
        info["cupy"] = None
    try:
        import torch

        if torch.cuda.is_available():
            info["gpu"] = torch.cuda.get_device_name()
            info["cuda_runtime_torch"] = torch.version.cuda
        try:
            import subprocess

            cmd = ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"]
            out = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
            info["driver"] = out.stdout.strip() or info.get("driver")
        except Exception:  # noqa: BLE001
            pass
    except Exception:  # noqa: BLE001
        pass
    return info


def copy_bandwidth(nbytes: int = 1 << 30, repeats: int = 10) -> float:
    """Device-to-device copy bandwidth in GB/s (read + write counted), median of repeats."""
    import torch

    a = torch.empty(nbytes // 4, dtype=torch.float32, device="cuda")
    b = torch.empty_like(a)
    for _ in range(3):
        b.copy_(a)
    torch.cuda.synchronize()
    ts = []
    for _ in range(repeats):
        t = time.perf_counter()
        b.copy_(a)
        torch.cuda.synchronize()
        ts.append(time.perf_counter() - t)
    del a, b
    torch.cuda.empty_cache()
    return 2 * nbytes / float(np.median(ts)) / 1e9


# ----------------------------------------------------------------------------- timing
def _time_fn(fn: Any, sync: Any, min_time: float, repeats: int) -> tuple[list[float], int]:
    """Calibrate an inner count so each repeat lasts >= min_time; return per-call times."""
    fn()
    sync()
    k = 1
    while True:
        t = time.perf_counter()
        for _ in range(k):
            fn()
        sync()
        el = time.perf_counter() - t
        if el >= min_time * 0.5 or k >= 1 << 16:
            break
        k *= 2
    k = max(1, int(math.ceil(k * min_time / max(el, 1e-9))))
    out = []
    for _ in range(repeats):
        t = time.perf_counter()
        for _ in range(k):
            fn()
        sync()
        out.append((time.perf_counter() - t) / k)
    return out, k


def fft_ceiling(n: int, dim: int, form: str, dtype: str, min_time: float, repeats: int
                ) -> dict[str, Any]:
    """Standalone cuFFT time for one RK stage's batched transforms (CuPy in-place, torch)."""
    ni, nf = fft_split(dim, form)
    nw = max(ni, nf)
    spec = spectral_shape(n, dim)
    out: dict[str, Any] = {}
    try:
        from pyspectral.backends._cuda_env import import_cufft, import_cupy

        cp, cf = import_cupy(), import_cufft()
        fp64 = dtype == "float64"
        W = cp.zeros((nw,) + spec, cp.complex128 if fp64 else cp.complex64)
        fsc = int(np.prod(spec))
        shape = [n] * dim
        cemb = [n] * (dim - 1) + [n // 2 + 1]
        remb = [n] * (dim - 1) + [2 * (n // 2 + 1)]
        c2r = cf.CUFFT_Z2D if fp64 else cf.CUFFT_C2R
        r2c = cf.CUFFT_D2Z if fp64 else cf.CUFFT_R2C
        inv = cf.PlanNd(shape, cemb, 1, fsc, remb, 1, 2 * fsc, c2r, ni, "C", -1, None)
        fwd = cf.PlanNd(shape, remb, 1, 2 * fsc, cemb, 1, fsc, r2c, nf, "C", -1, None)

        def stage_cp() -> None:
            inv.fft(W, W, cf.CUFFT_INVERSE)
            fwd.fft(W, W, cf.CUFFT_FORWARD)

        sync = cp.cuda.get_current_stream().synchronize
        ts, k = _time_fn(stage_cp, sync, min_time, repeats)
        out["cupy_inplace_stage_s"] = float(np.median(ts))
        # same transforms replayed from a CUDA graph (10 stages per graph): removes Python and
        # launch overhead, which otherwise dominates small grids
        try:
            stream = cp.cuda.Stream(non_blocking=True)
            with stream:
                stream.begin_capture()
                for _ in range(10):
                    stage_cp()
                graph = stream.end_capture()
            ts_g, _ = _time_fn(lambda: graph.launch(stream), stream.synchronize, min_time,
                               repeats)
            out["cupy_inplace_graph_stage_s"] = float(np.median(ts_g)) / 10
        except Exception as e:  # noqa: BLE001
            out["cupy_graph_error"] = f"{type(e).__name__}: {e}"
        ts_i, _ = _time_fn(lambda: inv.fft(W, W, cf.CUFFT_INVERSE), sync, min_time / 4, 3)
        ts_f, _ = _time_fn(lambda: fwd.fft(W, W, cf.CUFFT_FORWARD), sync, min_time / 4, 3)
        out["cupy_c2r_batch_s"] = float(np.median(ts_i))
        out["cupy_r2c_batch_s"] = float(np.median(ts_f))
    except Exception as e:  # noqa: BLE001
        out["cupy_error"] = f"{type(e).__name__}: {e}"
    try:
        import torch

        rdt = torch.float64 if dtype == "float64" else torch.float32
        Q = torch.zeros((nw,) + spec, dtype=torch.complex128 if dtype == "float64"
                        else torch.complex64, device="cuda")
        R = torch.zeros((nw,) + (n,) * dim, dtype=rdt, device="cuda")
        dims = tuple(range(-dim, 0))

        def stage_t() -> None:
            torch.fft.irfftn(Q[:ni], s=(n,) * dim, dim=dims, out=R[:ni])
            torch.fft.rfftn(R[:nf], dim=dims, out=Q[:nf])

        ts, _ = _time_fn(stage_t, torch.cuda.synchronize, min_time, repeats)
        out["torch_stage_s"] = float(np.median(ts))
    except Exception as e:  # noqa: BLE001
        out["torch_error"] = f"{type(e).__name__}: {e}"
    keys = ("cupy_inplace_stage_s", "cupy_inplace_graph_stage_s", "torch_stage_s")
    cands = [out[k] for k in keys if k in out]
    out["best_stage_s"] = min(cands) if cands else None
    return out


def _free_gpu() -> None:
    gc.collect()
    try:
        import torch

        torch.cuda.empty_cache()
        # each job compiles its own graphs; a long sweep must not hit dynamo's recompile limit
        torch._dynamo.reset()
    except Exception:  # noqa: BLE001
        pass
    try:
        from pyspectral.backends._cuda_env import import_cupy

        import_cupy().get_default_memory_pool().free_all_blocks()
    except Exception:  # noqa: BLE001
        pass


def _peak_mem_reset() -> None:
    try:
        import torch

        torch.cuda.reset_peak_memory_stats()
    except Exception:  # noqa: BLE001
        pass


def _peak_mem_mb() -> float | None:
    """Peak device memory of this process (torch allocator + CuPy pool), MB."""
    tot = 0.0
    try:
        import torch

        tot += torch.cuda.max_memory_allocated() / 1e6
    except Exception:  # noqa: BLE001
        pass
    try:
        from pyspectral.backends._cuda_env import import_cupy

        tot += import_cupy().get_default_memory_pool().total_bytes() / 1e6
    except Exception:  # noqa: BLE001
        pass
    return tot


def time_solver(name: str, n: int, dim: int, dtype: str, form: str, min_time: float,
                repeats: int) -> dict[str, Any]:
    """Median/min/max wall time per time step of one variant (warm-up incl. JIT excluded)."""
    v = get(name)
    cfg = Config(n=n, dim=dim, nu=1e-4, dtype=dtype, form=form)
    _peak_mem_reset()
    s = v.make(cfg)
    ic = (initial.random_2d(n, k0=min(16, n / 8), seed=1) if dim == 2
          else initial.random_3d(n, k0=min(4, n / 8), seed=1))
    s.set_state(initial.to_spectral(ic, dim))
    dt = 0.2 * (2 * math.pi / n) / 4.0
    t0 = time.perf_counter()
    for _ in range(2):  # warm-up: JIT/compile, plan creation, autotuning
        s.step(dt)
    s.synchronize()
    warm = time.perf_counter() - t0
    ts, k = _time_fn(lambda: s.step(dt), s.synchronize, min_time, repeats)
    st = s.get_state()
    finite = bool(np.isfinite(st).all())
    out = {"times_s": ts, "inner_steps": k, "warmup_s": warm, "finite": finite,
           "form": s.form, "ffts_per_step": s.fft_per_step, "peak_mem_MB": _peak_mem_mb()}
    return out


# ----------------------------------------------------------------------------- plans
def plan(mode: str) -> dict[str, Any]:
    """Size lists, variants and timing parameters for quick/dev/full runs."""
    gpu_all = [v.name for v in REGISTRY if v.gpu]
    gpu_2d = [x for x in gpu_all if x != "cupy-cuda-c/fused-inplace-lean"]
    if mode == "quick":
        return {"sizes2d": [256, 1024], "sizes3d": [32, 64], "min_time": 0.06, "repeats": 5,
                "budget_s": 90, "fp64": ["cupy-cuda-c/fused-inplace"], "variants2d": gpu_2d,
                "variants3d": gpu_all, "rot": ["cupy-cuda-c/fused-inplace"],
                "numpy2d": [256], "numpy3d": [32], "copy_bytes": 1 << 28}
    if mode == "dev":  # shared GPU: keep peak memory < ~1 GB
        return {"sizes2d": [256, 512, 1024, 2048], "sizes3d": [32, 64, 96, 128],
                "min_time": 0.3, "repeats": 5, "budget_s": 900,
                "fp64": ["cupy-cuda-c/fused-inplace", "triton/fused", "torch-eager/r2c-minfft",
                         "cupy/elementwise"],
                "variants2d": gpu_2d, "variants3d": gpu_all,
                "rot": ["cupy-cuda-c/fused-inplace", "triton/fused", "cupy/elementwise",
                        "torch-compile/inductor"],
                "numpy2d": [256, 512, 1024], "numpy3d": [32, 64], "copy_bytes": 1 << 28,
                # larger grids only for the leanest variant (keeps the shared GPU < ~1 GB)
                "big2d": [4096], "big3d": [192],
                "big_variants": ["cupy-cuda-c/fused-inplace-lean"]}
    # full: idle GPU, ~25 min. Priority order = job order; a time budget skips the tail.
    return {"sizes2d": [256, 1024, 2048, 4096, 8192], "sizes3d": [64, 128, 192, 256, 384],
            "min_time": 0.5, "repeats": 5, "budget_s": 1800,
            "fp64": ["cupy-cuda-c/fused-inplace"],
            "variants2d": gpu_2d, "variants3d": [x for x in gpu_all if not x.endswith("-lean")],
            "rot": ["cupy-cuda-c/fused-inplace"],
            # skip combinations that only measure an out-of-memory error
            "max_n": {("torch-eager/c2c-textbook", 2): 4096, ("torch-eager/c2c-textbook", 3): 192},
            "big2d": [], "big3d": [384], "big_variants": ["cupy-cuda-c/fused-inplace-lean"],
            "numpy2d": [256, 1024, 2048], "numpy3d": [64, 128], "numpy_dtypes": ["float32"],
            "copy_bytes": 1 << 30}


def _jobs(p: dict[str, Any]) -> list[tuple[str, int, int, str, str]]:
    """(variant, dim, n, dtype, form) in priority order (most important first)."""
    jobs: list[tuple[str, int, int, str, str]] = []
    for dim, sizes, variants in ((2, p["sizes2d"], p["variants2d"]),
                                 (3, p["sizes3d"], p["variants3d"])):
        for n in sizes:
            for v in variants:
                if n > p.get("max_n", {}).get((v, dim), 1 << 30):
                    continue
                jobs.append((v, dim, n, "float32", "basdevant" if "textbook" not in v else "auto"))
    for dim, sizes in ((2, p.get("big2d", [])), (3, p.get("big3d", []))):
        for n in sizes:
            for v in p["big_variants"]:
                jobs.append((v, dim, n, "float32", "basdevant"))
    for dim, sizes in ((2, p["sizes2d"]), (3, p["sizes3d"])):
        for n in sizes:
            for v in p["fp64"]:
                jobs.append((v, dim, n, "float64", "basdevant"))
    for n in p["sizes3d"]:
        for v in p["rot"]:
            jobs.append((v, 3, n, "float32", "rotational"))
    for dim, sizes in ((2, p["numpy2d"]), (3, p["numpy3d"])):
        for n in sizes:
            for dt in p.get("numpy_dtypes", ["float32", "float64"]):
                jobs.append(("numpy/scipy-fft", dim, n, dt, "basdevant"))
    return jobs


def run_benchmark(mode: str, out: Path | None) -> dict[str, Any]:
    """Run the benchmark plan and return the results document."""
    p = plan(mode)
    t_start = time.perf_counter()
    doc: dict[str, Any] = {"machine": machine_info(), "mode": mode,
                           "provisional": mode != "full", "runs": [], "fft_ceiling": [],
                           "skipped": []}
    if mode != "full":
        doc["provisional_reason"] = ("dev/quick run on a shared GPU; not publishable numbers")
    try:
        bw = copy_bandwidth(p["copy_bytes"])
    except Exception as e:  # noqa: BLE001
        bw = None
        doc["skipped"].append({"what": "copy_bandwidth", "why": str(e)})
    doc["ceiling"] = {"copy_bandwidth_GBs": SPEC_COPY_BW, "spec_bandwidth_GBs": SPEC_BW,
                      "copy_bandwidth_measured_GBs": bw, "fp32_tflops": None}
    print(f"copy bandwidth measured: {bw} GB/s (ceiling used: {SPEC_COPY_BW})", flush=True)
    ceil_cache: dict[tuple[int, int, str, str], dict[str, Any]] = {}
    jobs = _jobs(p)
    for i, (name, dim, n, dtype, form) in enumerate(jobs):
        elapsed = time.perf_counter() - t_start
        if elapsed > p["budget_s"]:
            doc["skipped"].append({"variant": name, "dim": dim, "n": n, "dtype": dtype,
                                   "form": form, "why": "time budget exhausted"})
            continue
        v = get(name)
        if v.gpu and not v.available()[0]:
            doc["skipped"].append({"variant": name, "why": v.available()[1]})
            continue
        try:
            r = time_solver(name, n, dim, dtype, form, p["min_time"], p["repeats"])
            _free_gpu()
        except BackendUnavailable as e:
            doc["skipped"].append({"variant": name, "dim": dim, "n": n, "dtype": dtype,
                                   "form": form, "why": str(e)})
            continue
        except Exception as e:  # noqa: BLE001  (OOM etc.)
            _free_gpu()
            doc["skipped"].append({"variant": name, "dim": dim, "n": n, "dtype": dtype,
                                   "form": form, "why": f"{type(e).__name__}: {str(e)[:200]}"})
            print(f"[{i + 1}/{len(jobs)}] {name} {dim}D n={n} {dtype} {form}: FAILED {e!s:.120}",
                  flush=True)
            continue
        fform = r["form"]
        key = (dim, n, dtype, fform)
        if key not in ceil_cache and v.gpu:
            ceil_cache[key] = fft_ceiling(n, dim, fform, dtype, p["min_time"], p["repeats"])
            ceil_cache[key].update({"dim": dim, "n": n, "dtype": dtype, "form": fform})
            doc["fft_ceiling"].append(ceil_cache[key])
            _free_gpu()
        doc["runs"].append(_row(name, dim, n, dtype, fform, r, ceil_cache.get(key)))
        row = doc["runs"][-1]
        print(f"[{i + 1}/{len(jobs)}] {name:34s} {dim}D n={n:5d} {dtype} {fform:10s} "
              f"{row['step_ms']:9.3f} ms/step  {row['value']:9.1f} Mpts/s  "
              f"roofline {row.get('pct_fft_roofline') or 0:5.1f}%  "
              f"mem {r['peak_mem_MB'] or 0:.0f} MB", flush=True)
        if out is not None:
            out.write_text(json.dumps(doc, indent=1) + "\n")
    doc["wall_s"] = time.perf_counter() - t_start
    if out is not None:
        out.write_text(json.dumps(doc, indent=1) + "\n")
    return doc


def _row(name: str, dim: int, n: int, dtype: str, form: str, r: dict[str, Any],
         ceil: dict[str, Any] | None) -> dict[str, Any]:
    v = get(name)
    ts = np.array(r["times_s"])
    tm = float(np.median(ts))
    tm_model = traffic_model(n, dim, form, dtype)
    pts = tm_model["points"]
    rate = pts / ts / 1e6  # Mpts/s  (grid points advanced one full RK4 step per second)
    bytes_per_pt = tm_model["total_bytes"] / pts
    eff = tm_model["total_bytes"] / tm / 1e9
    row: dict[str, Any] = {
        "backend": v.backend, "variant": v.variant, "size": [n] * dim, "dim": dim,
        "dtype": dtype, "form": form, "metric": "Mpts/s", "value": float(np.median(rate)),
        "repeats": int(len(ts)), "median": float(np.median(rate)), "min": float(rate.min()),
        "max": float(rate.max()), "step_ms": tm * 1e3, "step_ms_min": float(ts.min() * 1e3),
        "step_ms_max": float(ts.max() * 1e3), "inner_steps": r["inner_steps"],
        "bytes_per_unit": bytes_per_pt, "effective_GBs": eff,
        "pct_copy_bw": 100 * eff / SPEC_COPY_BW, "pct_spec_bw": 100 * eff / SPEC_BW,
        "flops_per_unit": None, "pct_peak_flops": None,
        "ffts_per_step": tm_model["ffts_per_step"],
        "fft_Gpts_per_s": tm_model["ffts_per_step"] * pts / tm / 1e9,
        "finite": r["finite"], "warmup_s": r["warmup_s"], "peak_mem_MB": r["peak_mem_MB"],
        "notes": v.description,
    }
    if v.gpu and ceil and ceil.get("best_stage_s"):
        t_fft = 4 * ceil["best_stage_s"]
        t_pw = tm_model["pointwise_bytes"] / (SPEC_COPY_BW * 1e9)
        row.update({"fft_ceiling_step_ms": t_fft * 1e3,
                    "roofline_step_ms": (t_fft + t_pw) * 1e3,
                    "pct_fft_roofline": 100 * (t_fft + t_pw) / tm,
                    "fft_share_of_step": t_fft / tm})
    return row


def main(argv: list[str] | None = None) -> int:
    """CLI for the benchmark."""
    ap = argparse.ArgumentParser(description="pyspectral benchmark (see bench/run.py)")
    g = ap.add_mutually_exclusive_group()
    g.add_argument("--quick", action="store_true", help="< 2 min smoke run")
    g.add_argument("--dev", action="store_true", help="shared-GPU dev run (< 1 GB, provisional)")
    g.add_argument("--full", action="store_true", help="publishable run (~25 min, idle GPU)")
    ap.add_argument("--out", default=None, help="output JSON path")
    a = ap.parse_args(argv)
    mode = "full" if a.full else "dev" if a.dev else "quick"
    root = Path(__file__).resolve().parents[2]
    default = {"full": "rtx3070-laptop.json", "dev": "rtx3070-laptop.json",
               "quick": "quick.json"}[mode]
    out = Path(a.out) if a.out else root / "results" / default
    out.parent.mkdir(parents=True, exist_ok=True)
    doc = run_benchmark(mode, out)
    print(f"wrote {out} ({len(doc['runs'])} runs, {len(doc['skipped'])} skipped, "
          f"{doc['wall_s']:.0f} s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())

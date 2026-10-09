"""Read results/*.json -> figures/*.pdf + *.png and paper/results.tex (all paper numbers).

    python figures/make_figures.py
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results"
FIG = ROOT / "figures"
sys.path.insert(0, str(ROOT / "src"))

plt.rcParams.update({"font.family": "serif", "font.size": 9, "axes.spines.top": False,
                     "axes.spines.right": False, "figure.dpi": 150, "savefig.bbox": "tight"})
COL_W, FULL_W = 3.4, 7.0
COLORS = {"numpy": "#7f7f7f", "torch-eager": "#9467bd", "torch-compile": "#8c564b",
          "triton": "#1f77b4", "warp": "#2ca02c", "numba": "#ff7f0e", "cupy-cuda-c": "#d62728",
          "cupy": "#e377c2"}
LADDER = ["torch-eager/c2c-textbook", "torch-eager/r2c-textbook", "torch-eager/r2c-minfft",
          "torch-compile/inductor", "cupy/elementwise", "warp/fused", "triton/fused",
          "cupy-cuda-c/fused-inplace", "cupy-cuda-c/fused-inplace-graph"]
MARK = {"torch-eager/c2c-textbook": "x", "torch-eager/r2c-textbook": "+",
        "torch-eager/r2c-minfft": "v", "torch-compile/inductor": "s", "cupy/elementwise": "D",
        "warp/fused": "^", "triton/fused": "o", "cupy-cuda-c/fused-inplace": "*",
        "cupy-cuda-c/fused-inplace-graph": "P", "cupy-cuda-c/fused-inplace-lean": "1",
        "numpy/scipy-fft": "."}


def load(name: str) -> dict | list | None:
    p = RES / name
    return json.loads(p.read_text()) if p.exists() else None


def save(fig: plt.Figure, name: str) -> None:
    fig.savefig(FIG / f"{name}.pdf")
    fig.savefig(FIG / f"{name}.png", dpi=200)
    plt.close(fig)
    print("wrote", name)


def vname(r: dict) -> str:
    return f"{r['backend']}/{r['variant']}"


def runs(doc: dict, dim: int, dtype: str = "float32", form: str | None = None) -> list[dict]:
    out = [r for r in doc["runs"] if r["dim"] == dim and r["dtype"] == dtype]
    if form:
        out = [r for r in out if r["form"] == form]
    return out


# ------------------------------------------------------------------------------ performance
def fig_scaling(doc: dict) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(FULL_W, 2.9))
    for ax, dim in zip(axes, (2, 3), strict=True):
        rs = [r for r in runs(doc, dim) if "textbook" not in r["variant"] or True]
        names = [v for v in LADDER + ["numpy/scipy-fft"] if any(vname(r) == v for r in rs)]
        for v in names:
            pts = sorted([(r["size"][0], r["step_ms"]) for r in rs if vname(r) == v
                          and (r["form"] == "basdevant" or "textbook" in v)])
            if not pts:
                continue
            x, y = zip(*pts, strict=True)
            ax.loglog(x, y, marker=MARK.get(v, "o"), ms=4, lw=1,
                      color=COLORS[v.split("/")[0]], label=v)
        ce = sorted({(r["size"][0], r["roofline_step_ms"]) for r in rs
                     if r.get("roofline_step_ms") and r["form"] == "basdevant"})
        if ce:
            x, y = zip(*ce, strict=True)
            ax.loglog(x, y, "k--", lw=1.2, label="FFT roofline")
        ax.set_xlabel(f"$N$ (grid ${'N^2' if dim == 2 else 'N^3'}$)")
        ax.set_ylabel("wall time per RK4 step [ms]")
        ax.set_title(f"{dim}D, fp32")
        ax.set_xticks([x for x in sorted({r['size'][0] for r in rs})])
        ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
        ax.minorticks_off()
    h, lab = axes[1].get_legend_handles_labels()
    h0, l0 = axes[0].get_legend_handles_labels()
    for hh, ll in zip(h0, l0, strict=True):
        if ll not in lab:
            h.append(hh)
            lab.append(ll)
    fig.legend(h, lab, loc="lower center", ncol=4, fontsize=6.5, frameon=False,
               bbox_to_anchor=(0.5, -0.12))
    fig.tight_layout()
    save(fig, "fig_scaling")


def _largest(doc: dict, dim: int, dtype: str = "float32") -> int | None:
    """Largest size at which the fastest fused backend ran."""
    ns = [r["size"][0] for r in runs(doc, dim, dtype) if r["backend"] == "cupy-cuda-c"]
    return max(ns) if ns else None


def fig_roofline(doc: dict) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(FULL_W, 2.8), sharey=False)
    for ax, dim in zip(axes, (2, 3), strict=True):
        n = _largest(doc, dim)
        rs = [r for r in runs(doc, dim) if r["size"][0] == n and r.get("pct_fft_roofline")
              and (r["form"] == "basdevant" or "textbook" in r["variant"])]
        rs = sorted(rs, key=lambda r: LADDER.index(vname(r)) if vname(r) in LADDER else 99)
        y = np.arange(len(rs))
        ax.barh(y, [r["pct_fft_roofline"] for r in rs],
                color=[COLORS[r["backend"]] for r in rs], height=0.65)
        ax.barh(y, [r["pct_copy_bw"] for r in rs], color="none", edgecolor="k", hatch="////",
                lw=0.5, height=0.65, label="% of copy BW (ideal-FFT traffic)")
        ax.axvline(100, color="k", ls="--", lw=1.2, label="FFT roofline (100%)")
        ax.set_yticks(y, [vname(r) for r in rs], fontsize=6.5)
        ax.invert_yaxis()
        ax.set_xlabel("% of FFT roofline step time")
        ax.set_title(f"{dim}D, $N={n}$, fp32")
        for yi, r in zip(y, rs, strict=True):
            ax.text(r["pct_fft_roofline"] + 1, yi, f"{r['pct_fft_roofline']:.0f}%",
                    va="center", fontsize=6)
    axes[0].legend(fontsize=6, frameon=False, loc="lower right")
    fig.tight_layout()
    save(fig, "fig_roofline")


def fig_ladder(doc: dict) -> None:
    """Step time at the largest 2D/3D size, against the FFT-only and FFT-roofline times."""
    fig, axes = plt.subplots(1, 2, figsize=(FULL_W, 2.6))
    for ax, dim in zip(axes, (2, 3), strict=True):
        n = _largest(doc, dim)
        rs = [r for r in runs(doc, dim) if r["size"][0] == n
              and (r["form"] == "basdevant" or "textbook" in r["variant"])]
        rs = sorted(rs, key=lambda r: LADDER.index(vname(r)) if vname(r) in LADDER else 99)
        y = np.arange(len(rs))
        ax.barh(y, [r["step_ms"] for r in rs], color=[COLORS[r["backend"]] for r in rs],
                height=0.65)
        roof = [r.get("roofline_step_ms") for r in rs if r.get("roofline_step_ms")]
        fft = [r.get("fft_ceiling_step_ms") for r in rs if r.get("fft_ceiling_step_ms")]
        if roof:
            ax.axvline(min(roof), color="k", ls="--", lw=1.2, label="FFT roofline")
            ax.axvline(min(fft), color="k", ls=":", lw=1.0, label="FFTs alone")
        ax.set_yticks(y, [f"{vname(r)} ({r['ffts_per_step']})" for r in rs], fontsize=6.5)
        ax.invert_yaxis()
        ax.set_xscale("log")
        ax.set_xlabel("ms per RK4 step (log)")
        ax.set_title(f"{dim}D, $N={n}$, fp32 (FFTs/step)")
    axes[0].legend(fontsize=6.5, frameon=False, loc="lower right")
    fig.tight_layout()
    save(fig, "fig_ladder")


def fig_precision(doc: dict) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(FULL_W, 2.4))
    for ax, dim in zip(axes, (2, 3), strict=True):
        for dtype, ls in (("float32", "-"), ("float64", "--")):
            for v in ("cupy-cuda-c/fused-inplace", "triton/fused", "torch-eager/r2c-minfft"):
                pts = sorted([(r["size"][0], r["value"]) for r in runs(doc, dim, dtype, "basdevant")
                              if vname(r) == v])
                if pts:
                    x, yv = zip(*pts, strict=True)
                    ax.loglog(x, yv, ls=ls, marker=MARK[v], ms=3, lw=1,
                              color=COLORS[v.split("/")[0]], label=f"{v} {dtype}")
        ax.set_xlabel("$N$")
        ax.set_ylabel("Mpts/s")
        ax.set_title(f"{dim}D: fp32 (solid) vs fp64 (dashed)")
    axes[0].legend(fontsize=5.5, frameon=False)
    fig.tight_layout()
    save(fig, "fig_precision")


# ------------------------------------------------------------------------------ validation
def fig_tgv3d() -> list[dict]:
    files = sorted(RES.glob("validation_tgv3d_n*.json"), key=lambda p: int(p.stem.split("n")[-1]))
    docs = [json.loads(p.read_text()) for p in files]
    if not docs:
        return []
    from pyspectral.validation import load_tgv_reference

    ref = load_tgv_reference()
    fig, ax = plt.subplots(figsize=(COL_W, 2.5))
    ax.plot(ref["t"], ref["dissipation"], "k-", lw=2.2, alpha=0.35,
            label=r"reference, $512^3$ spectral")
    cols = plt.cm.Reds(np.linspace(0.45, 1.0, len(docs)))
    for d, c in zip(docs, cols, strict=True):
        ax.plot(d["t"], d["dissipation_2nuZ"], color=c, lw=1,
                label=rf"this work, ${d['n']}^3$")
    ax.set_xlabel("$t$")
    ax.set_ylabel(r"$\varepsilon = -\mathrm{d}E/\mathrm{d}t$")
    ax.set_xlim(0, 20)
    ax.legend(fontsize=7, frameon=False)
    fig.tight_layout()
    save(fig, "fig_tgv3d")
    return docs


def fig_conv2d(v2: dict) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(FULL_W, 2.4))
    sp = v2.get("spatial")
    if sp:
        axes[0].semilogy(sp["n"], sp["err"], "o-", color=COLORS["numpy"], ms=4)
        axes[0].set_xlabel("$N$")
        axes[0].set_ylabel(r"max. rel. error of $\omega$")
        axes[0].set_title("spatial: exponential convergence")
    tc = v2.get("temporal")
    ro = v2.get("rk4_order")
    if tc:
        dt = np.array(tc["dt"])
        axes[1].loglog(dt, tc["err"], "o-", color=COLORS["cupy-cuda-c"], ms=4,
                       label=f"IF-RK4, nonlinear flow (order {tc['order']:.2f})")
    if ro:
        axes[1].loglog(ro["dt"], ro["err_rk4"], "s-", color=COLORS["numpy"], ms=4,
                       label=f"RK4, exact TGV (order {ro['order_rk4']:.2f})")
        axes[1].loglog(ro["dt"], np.maximum(ro["err_ifrk4"], 1e-17), "^:", color="k", ms=4,
                       label="IF-RK4, exact TGV (round-off)")
    x = np.array([0.003, 0.03])
    axes[1].loglog(x, 2e-3 * x**4 / x[0] ** 4 * 1e-9, "k--", lw=0.8, label=r"$\propto\Delta t^4$")
    axes[1].set_xlabel(r"$\Delta t$")
    axes[1].set_ylabel("max. rel. error")
    axes[1].set_title("temporal")
    axes[1].legend(fontsize=6, frameon=False)
    fig.tight_layout()
    save(fig, "fig_conv2d")


def fig_spectra(f2: dict | None, f3: dict | None) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(FULL_W, 2.6))
    if f2:
        k = np.array(f2["k"], float)
        E = np.array(f2["E_k"])
        m = (k > 0) & (k < f2["n"] / 3)
        axes[0].loglog(k[m], E[m], color=COLORS["cupy-cuda-c"], lw=1.2, label=f"$N={f2['n']}$")
        k0, k1 = f2["fit_range"]
        kk = np.array([k0, k1])
        e0 = np.interp(k0, k, E)
        axes[0].loglog(kk, 2 * e0 * (kk / k0) ** -3, "k--", lw=1, label="$k^{-3}$")
        axes[0].loglog(kk, 0.5 * e0 * (kk / k0) ** f2["slope"], "k:", lw=1,
                       label=f"fit $k^{{{f2['slope']:.2f}}}$")
        axes[0].axvspan(*f2["band"], color="0.85", lw=0)
        axes[0].set_xlabel("$k$")
        axes[0].set_ylabel("$E(k)$")
        axes[0].set_title("2D forced: enstrophy cascade")
        axes[0].legend(fontsize=7, frameon=False)
    if f3:
        k = np.array(f3["k"], float)
        E = np.array(f3["E_k"])
        m = (k > 0) & (k < f3["n"] / 3)
        eps = f3["epsilon"]
        eta = f3["eta"]
        comp = E[m] * eps ** (-2 / 3) * k[m] ** (5 / 3)
        axes[1].semilogx(k[m] * eta, comp, color=COLORS["cupy-cuda-c"], lw=1.2,
                         label=rf"$N={f3['n']}$, $Re_\lambda={f3['re_lambda']:.0f}$")
        axes[1].axhline(1.6, color="k", ls="--", lw=1, label=r"$C_K \approx 1.6$")
        axes[1].set_xlabel(r"$k\eta$")
        axes[1].set_ylabel(r"$\varepsilon^{-2/3}k^{5/3}E(k)$")
        axes[1].set_title("3D forced: compensated spectrum")
        axes[1].legend(fontsize=7, frameon=False)
    fig.tight_layout()
    save(fig, "fig_spectra")


# ------------------------------------------------------------------------------ results.tex
def fmt(x: float, nd: int = 2) -> str:
    return f"{x:.{nd}f}"


def sci(x: float) -> str:
    if x == 0 or not math.isfinite(x):
        return "0"
    e = int(math.floor(math.log10(abs(x))))
    m = x / 10**e
    return rf"\num{{{m:.1f}e{e}}}"


def write_results_tex(doc: dict | None, v2: dict | None, vb: list | None, tgv: list[dict],
                      f2: dict | None, f3: dict | None) -> None:
    L = ["% GENERATED by figures/make_figures.py from results/*.json -- do not edit"]

    def m(name: str, val: str) -> None:
        L.append(rf"\newcommand{{\{name}}}{{{val}}}")

    prov = bool(doc.get("provisional", True)) if doc else True
    L.append(r"\newif\ifprovisional")
    L.append(r"\provisionaltrue" if prov else r"\provisionalfalse")
    if doc:
        m("benchMode", doc.get("mode", "?"))
        m("gpuName", doc["machine"].get("gpu", "unknown GPU"))
        m("copyBW", fmt(doc["ceiling"]["copy_bandwidth_GBs"], 0))
        for dim, tag in ((2, "Two"), (3, "Three")):
            n = _largest(doc, dim)
            rs = [r for r in runs(doc, dim) if r["size"][0] == n and r["form"] == "basdevant"
                  and r["backend"] != "numpy"]
            if not rs:
                continue
            best = min(rs, key=lambda r: r["step_ms"])
            m(f"best{tag}Name", vname(best).replace("_", r"\_"))
            m(f"best{tag}N", str(n))
            m(f"best{tag}Ms", fmt(best["step_ms"], 2))
            m(f"best{tag}Mpts", fmt(best["value"], 0))
            m(f"best{tag}Roof", fmt(best.get("pct_fft_roofline") or 0, 0))
            m(f"best{tag}Copy", fmt(best["pct_copy_bw"], 0))
            m(f"best{tag}Spec", fmt(best["pct_spec_bw"], 0))
            m(f"best{tag}FFTGpts", fmt(best["fft_Gpts_per_s"], 1))
            m(f"best{tag}FFTshare", fmt(100 * (best.get("fft_share_of_step") or 0), 0))
            m(f"best{tag}RoofMs", fmt(best.get("roofline_step_ms") or 0, 2))
            m(f"best{tag}FFTMs", fmt(best.get("fft_ceiling_step_ms") or 0, 2))
            eag = [r for r in rs if vname(r) == "torch-eager/r2c-minfft"]
            if eag:
                m(f"speedup{tag}Eager", fmt(eag[0]["step_ms"] / best["step_ms"], 1))
            tb = [r for r in runs(doc, dim) if r["size"][0] == n
                  and vname(r) == "torch-eager/c2c-textbook"]
            if tb:
                m(f"speedup{tag}Textbook", fmt(tb[0]["step_ms"] / best["step_ms"], 1))
            nps = [r for r in runs(doc, dim) if r["backend"] == "numpy"]
            if nps:
                nmax = max(r["size"][0] for r in nps)
                cpu = [r for r in nps if r["size"][0] == nmax][0]
                gpu = [r for r in runs(doc, dim) if r["size"][0] == nmax
                       and vname(r) == vname(best)]
                if gpu:
                    m(f"cpu{tag}N", str(nmax))
                    m(f"speedup{tag}CPU", fmt(cpu["step_ms"] / gpu[0]["step_ms"], 0))
            rot = [r for r in runs(doc, 3, "float32", "rotational") if r["size"][0] == n
                   and r["backend"] == "cupy-cuda-c" and r["variant"] == "fused-inplace"]
            bas = [r for r in rs if vname(r) == "cupy-cuda-c/fused-inplace"]
            if dim == 3 and rot and bas:
                m("rotOverBasd", fmt(rot[0]["step_ms"] / bas[0]["step_ms"], 2))
            f64 = [r for r in runs(doc, dim, "float64", "basdevant") if r["size"][0] == n
                   and vname(r) == "cupy-cuda-c/fused-inplace"]
            if f64 and bas:
                m(f"fpRatio{tag}", fmt(f64[0]["step_ms"] / bas[0]["step_ms"], 1))
        # performance table
        rows = []
        for dim in (2, 3):
            n = _largest(doc, dim)
            for r in sorted([r for r in runs(doc, dim) if r["size"][0] == n
                             and (r["form"] == "basdevant" or "textbook" in r["variant"])],
                            key=lambda r: r["step_ms"], reverse=True):
                roof = r.get("pct_fft_roofline")
                rows.append(f"{dim}D ${n}^{dim}$ & \\texttt{{{vname(r)}}} & {r['ffts_per_step']} & "
                            f"{r['step_ms']:.2f} & {r['value']:.0f} & {r['effective_GBs']:.0f} & "
                            f"{r['pct_copy_bw']:.0f} & "
                            f"{'--' if roof is None else f'{roof:.0f}'} \\\\")
            rows.append(r"\midrule")
        L.append(r"\newcommand{\perfTableRows}{" + "\n".join(rows[:-1]) + "}")
    if v2:
        t = v2["tgv2d"]
        e64 = max(r["rel_err"] for r in t if r["dtype"] == "float64")
        e32 = max(r["rel_err"] for r in t if r["dtype"] == "float32")
        m("tgvTwoErrSixtyFour", sci(e64))
        m("tgvTwoErrThirtyTwo", sci(e32))
        m("tgvTwoNVariants", str(len({r["variant"] for r in t})))
        m("rkOrder", fmt(v2["rk4_order"]["order_rk4"], 2))
        if "temporal" in v2:
            m("ifrkOrder", fmt(v2["temporal"]["order"], 2))
        if "spatial" in v2:
            sp = v2["spatial"]
            m("spatialErrMin", sci(min(sp["err"])))
            m("spatialNMax", str(max(sp["n"])))
            m("spatialNRef", str(sp["n_ref"]))
    if vb:
        m("aggErrSixtyFour", sci(max(r["rel_err"] for r in vb if r["dtype"] == "float64")))
        m("aggErrThirtyTwo", sci(max(r["rel_err"] for r in vb if r["dtype"] == "float32")))
        m("aggN", str(len(vb)))
    if tgv:
        best = max(tgv, key=lambda d: d["n"])
        m("tgvN", str(best["n"]))
        m("tgvPeak", fmt(best["peak_value"], 5))
        m("tgvPeakTime", fmt(best["peak_time"], 2))
        m("tgvRefPeak", fmt(best["ref_peak_value"], 5))
        m("tgvRefPeakTime", fmt(best["ref_peak_time"], 2))
        m("tgvPeakErr", fmt(100 * abs(best["peak_rel_err"]), 2))
        m("tgvMaxErr", fmt(100 * best["max_abs_err_over_peak"], 2))
        m("tgvSteps", str(best["steps"]))
        m("tgvWall", fmt(best["wall_s"], 0))
        rows = [f"${d['n']}^3$ & {d['peak_value']:.5f} & {d['peak_time']:.2f} & "
                f"{100 * d['peak_rel_err']:+.2f} & {100 * d['max_abs_err_over_peak']:.2f} & "
                f"{d['steps']} \\\\" for d in tgv]
        L.append(r"\newcommand{\tgvTableRows}{" + "\n".join(rows) + "}")
    if f2:
        m("twoSlope", fmt(f2["slope"], 2))
        m("twoN", str(f2["n"]))
        m("twoFit", f"{f2['fit_range'][0]:.0f}\\le k\\le {f2['fit_range'][1]:.0f}")
    if f3:
        m("threeSlope", fmt(f3["slope"], 2))
        m("threeN", str(f3["n"]))
        m("threeRe", fmt(f3["re_lambda"], 0))
        m("threeKeta", fmt(f3["kmax_eta"], 2))
        m("threeFit", f"{f3['fit_range'][0]:.0f}\\le k\\le {f3['fit_range'][1]:.0f}")
    (ROOT / "paper" / "results.tex").write_text("\n".join(L) + "\n")
    print("wrote paper/results.tex")


def main() -> int:
    doc = load("rtx3070-laptop.json")
    v2 = load("validation_2d.json")
    vb = load("validation_backends.json")
    f2 = load("validation_forced2d.json")
    f3 = load("validation_forced3d.json")
    if doc:
        fig_scaling(doc)
        fig_roofline(doc)
        fig_ladder(doc)
        fig_precision(doc)
    tgv = fig_tgv3d()
    if v2:
        fig_conv2d(v2)
    if f2 or f3:
        fig_spectra(f2, f3)
    write_results_tex(doc, v2, vb, tgv, f2, f3)
    return 0


if __name__ == "__main__":
    sys.exit(main())

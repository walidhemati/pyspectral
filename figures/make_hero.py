"""Hero animation: vorticity of freely decaying 2D turbulence (docs/media/hero.gif).

    python figures/make_hero.py [--backend cupy-cuda-c/fused-inplace] [--n 768]
"""

from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
warnings.filterwarnings("ignore", message=".*CuPy.*")


def main() -> int:
    import matplotlib
    from PIL import Image

    from pyspectral import initial
    from pyspectral.backends import make_solver
    from pyspectral.config import Config
    from pyspectral.simulate import run
    from pyspectral.validation import pad_spectrum

    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", default="cupy-cuda-c/fused-inplace")
    ap.add_argument("--n", type=int, default=768)
    ap.add_argument("--nu", type=float, default=3e-5)
    ap.add_argument("--t-end", type=float, default=40.0)
    ap.add_argument("--frames", type=int, default=40)
    ap.add_argument("--px", type=int, default=300)
    ap.add_argument("--out", default=str(ROOT / "docs" / "media" / "hero.gif"))
    a = ap.parse_args()

    s = make_solver(a.backend, Config(n=a.n, dim=2, nu=a.nu, dtype="float32"))
    s.set_state(initial.to_spectral(initial.random_2d(a.n, k0=24, seed=7, energy=0.5), 2))
    cmap = matplotlib.colormaps["RdBu_r"]
    frames: list[Image.Image] = []

    def grab(sol: object) -> None:
        w_hat = pad_spectrum(sol.get_state(), a.px, 2)  # spectral down-sampling (exact band)
        w = np.fft.irfftn(w_hat, s=(a.px, a.px), axes=(0, 1))
        lim = 2.5 * float(np.sqrt(np.mean(w * w)))
        rgb = (cmap(np.clip(0.5 + 0.5 * w / lim, 0, 1))[..., :3] * 255).astype(np.uint8)
        frames.append(Image.fromarray(rgb).quantize(colors=64, method=Image.Quantize.MEDIANCUT))

    grab(s)
    # frames uniformly spaced in log-time-ish: early dynamics are fast
    res = run(s, a.t_end, cfl=0.6, dt_max=0.02, callback=grab,
              callback_dt=a.t_end / (a.frames - 1))
    print(f"{res.steps} steps, {res.wall:.1f} s, {len(frames)} frames")
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    frames[0].save(out, save_all=True, append_images=frames[1:], duration=120, loop=0,
                   optimize=True)
    print(f"wrote {out} ({out.stat().st_size / 1e6:.2f} MB)")
    return 0


if __name__ == "__main__":
    sys.exit(main())

"""Measure the hardware ceilings used by the roofline: copy bandwidth and standalone cuFFT.

    python bench/roofline.py [--n2d 4096] [--n3d 256] [--dtype float32]
"""

import argparse
import json
import sys
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
warnings.filterwarnings("ignore", message=".*CuPy.*")

from pyspectral.benchmark import (  # noqa: E402
    SPEC_BW,
    SPEC_COPY_BW,
    copy_bandwidth,
    fft_ceiling,
    traffic_model,
)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n2d", type=int, default=2048)
    ap.add_argument("--n3d", type=int, default=128)
    ap.add_argument("--dtype", default="float32")
    ap.add_argument("--copy-mib", type=int, default=256)
    a = ap.parse_args()
    out = {"copy_bandwidth_measured_GBs": copy_bandwidth(a.copy_mib << 20),
           "copy_bandwidth_spec_suite_GBs": SPEC_COPY_BW, "spec_bandwidth_GBs": SPEC_BW}
    for dim, n in ((2, a.n2d), (3, a.n3d)):
        c = fft_ceiling(n, dim, "basdevant", a.dtype, 0.3, 5)
        tm = traffic_model(n, dim, "basdevant", a.dtype)
        c["fft_ideal_GBs_cupy"] = (tm["fft_bytes"] / 4 / c["cupy_inplace_stage_s"] / 1e9
                                   if "cupy_inplace_stage_s" in c else None)
        out[f"{dim}d_n{n}"] = c
    print(json.dumps(out, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())

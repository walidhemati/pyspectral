"""The only benchmark entry point.

    python bench/run.py --quick   # < 2 min smoke run       -> results/quick.json
    python bench/run.py --dev     # shared-GPU dev run      -> results/rtx3070-laptop.json
                                  #   (marked provisional)
    python bench/run.py --full    # publishable, idle GPU   -> results/rtx3070-laptop.json
"""

import sys
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
warnings.filterwarnings("ignore", message=".*CuPy.*")

from pyspectral.benchmark import main  # noqa: E402

if __name__ == "__main__":
    sys.exit(main())

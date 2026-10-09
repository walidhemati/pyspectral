#!/usr/bin/env bash
# Reproduce every validation result in results/validation_*.json.
# Long runs are split into <= 50 s chunks with on-disk checkpoints (exit code 3 = "resume me"),
# which keeps each GPU job short on a shared GPU.
set -euo pipefail
cd "$(dirname "$0")/.."
PY=${PY:-python}
CK=${CK:-/tmp/pyspectral-ck}
mkdir -p "$CK"

chunked() {  # run a validate command until it finishes
  local rc=3
  while [ $rc -eq 3 ]; do
    set +e
    PYTHONPATH=src $PY -m pyspectral.cli validate "$@" --wall-limit 50
    rc=$?
    set -e
  done
  return $rc
}

# 2D TGV (all backends), RK4 order, IF-RK4 self-convergence, spectral convergence (CPU)
PYTHONPATH=src $PY -m pyspectral.cli validate --what 2d
# every GPU variant vs the NumPy fp64 oracle
PYTHONPATH=src $PY -m pyspectral.cli validate --what backends
# 3D Taylor-Green vortex, Re = 1600
for n in 64 128 192; do
  chunked --what tgv3d --n $n --backend cupy-cuda-c/fused-inplace --checkpoint "$CK/tgv$n.npz"
done
chunked --what tgv3d --n 256 --backend cupy-cuda-c/fused-inplace-lean --checkpoint "$CK/tgv256.npz"
# forced 2D (enstrophy cascade) and 3D turbulence
chunked --what forced2d --n 512 --nu 5e-5 --alpha 0.2 --alpha-kmax 2.5 --klo 3.5 --khi 4.5 \
  --power 0.01 --t-end 60 --t-avg 30 --kfit0 15 --kfit1 50 --checkpoint "$CK/f2.npz"
chunked --what forced3d --n 192 --nu 0.003 --klo 0.5 --khi 2.5 --power 0.1 --t-end 20 \
  --t-avg 8 --kfit0 3 --kfit1 8 --checkpoint "$CK/f3.npz"

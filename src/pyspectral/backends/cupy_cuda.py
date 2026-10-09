"""Hand-written CUDA C backend (CuPy RawModule / NVRTC) with in-place batched cuFFT plans.

Per RK4 stage: one batched in-place C2R, one fused physical-space kernel, one batched in-place
R2C and one fused spectral kernel (nonlinear term + dealiasing + projection + IF-RK4 update +
next-stage FFT inputs). The only memory is the state, the RK accumulator and the FFT work
buffers (2 + n_work spectral fields), no temporaries are allocated inside a step.
"""

from __future__ import annotations

from importlib import resources
from typing import Any

import numpy as np

from pyspectral.backends._cuda_env import import_cufft, import_cupy
from pyspectral.backends._fused_common import if_scalars, if_table, n_work
from pyspectral.backends.base import BackendUnavailable, SpectralSolver
from pyspectral.config import Config, fft_split
from pyspectral.grid import spectral_shape
from pyspectral.ops import CupyOps

_BLOCK = 256


def _load_module(cp: Any, fp64: bool) -> Any:
    src = resources.files("pyspectral.kernels").joinpath("spectral.cu").read_text()
    opts = ("-DUSE_DOUBLE",) if fp64 else ()
    return cp.RawModule(code=src, options=("--std=c++14",) + opts)


class CupyCudaSolver(SpectralSolver):
    """Fused CUDA C kernels + in-place cuFFT (the hand-written reference backend)."""

    backend = "cupy-cuda-c"
    variant = "fused-inplace"
    default_form = "basdevant"
    forms = ("basdevant", "rotational")

    def __init__(self, cfg: Config, graph: bool = False, batched: bool = True) -> None:
        super().__init__(cfg)
        if self.dim == 2 and self.form != "basdevant":
            raise BackendUnavailable("cupy-cuda-c: 2D supports the basdevant form only")
        try:
            cp = import_cupy()
            cufft = import_cufft()
        except Exception as e:  # noqa: BLE001
            raise BackendUnavailable(str(e)) from e
        self.cp, self.cufft = cp, cufft
        fp64 = cfg.is_fp64
        self.rdt = np.float64 if fp64 else np.float32
        self.cdt = np.complex128 if fp64 else np.complex64
        n, d = self.n, self.dim
        self.nh = n // 2 + 1
        spec = spectral_shape(n, d)
        self.fsc = int(np.prod(spec))
        self.fsr = 2 * self.fsc
        self.rot = int(self.form == "rotational")
        self.n_inv, self.n_fwd = fft_split(d, self.form)
        self.nw = n_work(d, self.form)
        ncomp = 1 if d == 2 else 3
        self.u0 = cp.zeros((ncomp,) + spec if d == 3 else spec, dtype=self.cdt)
        self.acc = cp.zeros_like(self.u0)
        self.W = cp.zeros((self.nw,) + spec, dtype=self.cdt)
        self.tab = cp.zeros(2 * self.nh, dtype=self.rdt)
        self._tab_key: tuple[float, float] | None = None
        self.mod = _load_module(cp, fp64)
        k = self.mod.get_function
        if d == 2:
            self.k_pre, self.k_phys, self.k_post = k("pre2d"), k("phys2d"), k("post2d")
        else:
            self.k_pre = k("pre3d")
            self.k_phys = k("phys3d_rot" if self.rot else "phys3d_basd")
            self.k_post = k("post3d")
        # batched plans need a cuFFT work area as large as the whole batch; ``batched=False``
        # loops over single-field plans (work area of one field) for memory-lean big runs
        self.batched = batched
        self.plan_inv, self.plan_fwd = self._make_plans()
        self.scale = self.rdt(1.0 / float(n) ** d)
        self._ops = CupyOps()
        self.graph = graph
        self._graphs: dict[tuple[float, float], Any] = {}

    @classmethod
    def available(cls) -> tuple[bool, str]:
        try:
            cp = import_cupy()
            import_cufft()
            if cp.cuda.runtime.getDeviceCount() < 1:
                return False, "no CUDA device"
        except Exception as e:  # noqa: BLE001
            return False, str(e)
        return True, ""

    @property
    def ops(self) -> Any:
        return self._ops

    # ------------------------------------------------------------------ cuFFT plans
    def _make_plans(self) -> tuple[Any, Any]:
        cf = self.cufft
        n, d = self.n, self.dim
        shape = [n] * d
        cemb = [n] * (d - 1) + [self.nh]
        remb = [n] * (d - 1) + [2 * self.nh]
        fp64 = self.cfg.is_fp64
        c2r = cf.CUFFT_Z2D if fp64 else cf.CUFFT_C2R
        r2c = cf.CUFFT_D2Z if fp64 else cf.CUFFT_R2C
        bi, bf = (self.n_inv, self.n_fwd) if self.batched else (1, 1)
        inv = cf.PlanNd(shape, cemb, 1, self.fsc, remb, 1, self.fsr, c2r, bi, "C", -1, None)
        fwd = cf.PlanNd(shape, remb, 1, self.fsr, cemb, 1, self.fsc, r2c, bf, "C", -1, None)
        return inv, fwd

    def fft_inverse(self) -> None:
        """Batched in-place C2R on the work buffer."""
        if self.batched:
            self.plan_inv.fft(self.W, self.W, self.cufft.CUFFT_INVERSE)
        else:
            for i in range(self.n_inv):
                self.plan_inv.fft(self.W[i], self.W[i], self.cufft.CUFFT_INVERSE)

    def fft_forward(self) -> None:
        """Batched in-place R2C on the work buffer."""
        if self.batched:
            self.plan_fwd.fft(self.W, self.W, self.cufft.CUFFT_FORWARD)
        else:
            for i in range(self.n_fwd):
                self.plan_fwd.fft(self.W[i], self.W[i], self.cufft.CUFFT_FORWARD)

    # ------------------------------------------------------------------ state
    def _grid(self, count: int) -> tuple[int]:
        return ((count + _BLOCK - 1) // _BLOCK,)

    def set_state(self, state_hat: np.ndarray) -> None:
        a = self.check_state(state_hat).astype(self.cdt)
        self.u0[...] = self.cp.asarray(a.reshape(self.u0.shape))
        self._pre()

    def _pre(self) -> None:
        n, nh, fsc = np.int32(self.n), np.int32(self.nh), np.int64(self.fsc)
        if self.dim == 2:
            args = (self.u0, self.W, n, nh, fsc, self.scale)
        else:
            args = (self.u0, self.W, n, nh, fsc, np.int32(self.rot), self.scale)
        self.k_pre(self._grid(self.fsc), (_BLOCK,), args)

    def state_native(self) -> Any:
        return self.u0

    # ------------------------------------------------------------------ stepping
    def _update_table(self, dt: float) -> tuple[Any, ...]:
        key = (dt, self.cfg.nu)
        if self._tab_key != key:
            self.tab.set(if_table(self.n, self.cfg.nu, dt).astype(self.rdt))
            self._tab_key = key
        ea1, eah, eb1, ebh = if_scalars(self.cfg.alpha, self.force_amp, dt)
        r = self.rdt
        klo2 = r(self.cfg.force_klo**2) if self.force_amp != 0.0 else r(1.0)
        khi2 = r(self.cfg.force_khi**2) if self.force_amp != 0.0 else r(0.0)
        return (r(dt), r(ea1), r(eah), r(eb1), r(ebh), klo2, khi2, self.scale)

    def _stage(self, stage: int, sc: tuple[Any, ...]) -> None:
        self.fft_inverse()
        self.k_phys(self._grid(self.fsr), (_BLOCK,), (self.W, np.int64(self.fsr)))
        self.fft_forward()
        n, nh, fsc = np.int32(self.n), np.int32(self.nh), np.int64(self.fsc)
        if self.dim == 2:
            args = (self.W, self.u0, self.acc, self.tab, n, nh, fsc, np.int32(stage)) + sc
        else:
            args = (self.W, self.u0, self.acc, self.tab, n, nh, fsc, np.int32(self.rot),
                    np.int32(stage)) + sc
        self.k_post(self._grid(self.fsc), (_BLOCK,), args)

    def _step(self, dt: float) -> None:
        sc = self._update_table(dt)
        if not self.graph:
            for s in range(4):
                self._stage(s, sc)
            return
        key = (float(dt), float(self.force_amp))
        g = self._graphs.get(key)
        if g is None:
            if len(self._graphs) > 8:
                self._graphs.clear()
            cur = self.cp.cuda.get_current_stream()
            cur.synchronize()
            stream = self.cp.cuda.Stream(non_blocking=True)
            with stream:
                stream.begin_capture()
                for s in range(4):
                    self._stage(s, sc)
                g = stream.end_capture()
            self._graphs[key] = g
        g.launch(self.cp.cuda.get_current_stream())


class CupyCudaLeanSolver(CupyCudaSolver):
    """Single-field cuFFT plans looped over the batch: smallest memory footprint."""

    variant = "fused-inplace-lean"

    def __init__(self, cfg: Config) -> None:
        super().__init__(cfg, batched=False)


class CupyCudaGraphSolver(CupyCudaSolver):
    """Same kernels, one whole time step replayed as a CUDA graph (removes launch overhead)."""

    variant = "fused-inplace-graph"

    def __init__(self, cfg: Config) -> None:
        super().__init__(cfg, graph=True)

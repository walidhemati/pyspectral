"""Time-stepping driver: CFL-adaptive dt, diagnostics history, constant-power forcing."""

from __future__ import annotations

import math
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from pyspectral.backends.base import SpectralSolver


@dataclass
class RunResult:
    """Time history recorded by :func:`run`."""

    t: list[float] = field(default_factory=list)
    energy: list[float] = field(default_factory=list)
    enstrophy: list[float] = field(default_factory=list)
    dt: list[float] = field(default_factory=list)
    steps: int = 0
    wall: float = 0.0
    finished: bool = True

    def as_dict(self) -> dict[str, Any]:
        return {"t": self.t, "energy": self.energy, "enstrophy": self.enstrophy, "dt": self.dt,
                "steps": self.steps, "wall_s": self.wall}


def cfl_dt(solver: SpectralSolver, cfl: float, dt_max: float) -> float:
    """``dt = cfl * dx / sum_i max|u_i|`` capped at ``dt_max``."""
    umax = solver.diag.max_velocity_sum(solver.state_native())
    dx = 2.0 * math.pi / solver.n
    if umax <= 0.0 or not math.isfinite(umax):
        return dt_max
    return min(dt_max, cfl * dx / umax)


def run(solver: SpectralSolver, t_end: float, *, dt: float | None = None, cfl: float = 0.5,
        dt_max: float = 0.05, cfl_every: int = 10, diag_dt: float | None = None,
        target_power: float | None = None,
        callback: Callable[[SpectralSolver], None] | None = None,
        callback_dt: float | None = None, wall_limit: float | None = None,
        result: RunResult | None = None) -> RunResult:
    """Integrate to ``t_end``.

    ``dt`` fixed, or CFL-adaptive (re-evaluated every ``cfl_every`` steps; the final step is
    shortened to land exactly on ``t_end`` and on every diagnostics time). ``target_power``
    enables constant-power band forcing: the band growth rate is reset every ``cfl_every``
    steps to ``A = P / (2 E_band)``.
    """
    res = result if result is not None else RunResult()
    d = solver.diag
    cfg = solver.cfg

    def record() -> None:
        st = solver.state_native()
        res.t.append(solver.t)
        res.energy.append(d.energy(st))
        res.enstrophy.append(d.enstrophy(st))

    if not res.t:
        record()
    next_diag = (res.t[-1] + diag_dt) if diag_dt else math.inf
    next_cb = solver.t + callback_dt if callback_dt else math.inf
    cur_dt = dt if dt is not None else cfl_dt(solver, cfl, dt_max)
    t0 = time.perf_counter()
    k = 0
    eps = 1e-12 * max(1.0, t_end)
    while solver.t < t_end - eps:
        if k % cfl_every == 0:
            if dt is None and k > 0:
                cur_dt = cfl_dt(solver, cfl, dt_max)
            if target_power is not None:
                eb = d.band_energy(solver.state_native(), cfg.force_klo, cfg.force_khi)
                solver.set_forcing_amplitude(target_power / (2.0 * eb) if eb > 0 else 0.0)
        h = min(cur_dt, t_end - solver.t, next_diag - solver.t, next_cb - solver.t)
        solver.step(h)
        res.dt.append(h)
        k += 1
        if solver.t >= next_diag - eps:
            record()
            next_diag += diag_dt  # type: ignore[operator]
        if callback is not None and solver.t >= next_cb - eps:
            callback(solver)
            next_cb += callback_dt  # type: ignore[operator]
        if wall_limit is not None and time.perf_counter() - t0 > wall_limit:
            break
    solver.synchronize()
    res.finished = solver.t >= t_end - eps
    res.wall += time.perf_counter() - t0
    res.steps += k
    if res.finished and res.t[-1] < solver.t - eps:
        record()
    return res

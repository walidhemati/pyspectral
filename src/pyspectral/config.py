"""Simulation configuration shared by all backends."""

from __future__ import annotations

from dataclasses import dataclass, replace

FORMS_2D = ("basdevant", "advective")
FORMS_3D = ("basdevant", "rotational")
DEALIAS = ("2/3", "phase-shift")


@dataclass(frozen=True)
class Config:
    """Physical and numerical parameters of a periodic-box run (box side ``2*pi``).

    The linear operator, integrated exactly by the integrating factor, is
    ``L(k) = -(nu*k^2 + alpha*[|k| <= alpha_kmax]) + force_amp*[force_klo <= |k| <= force_khi]``:
    viscosity, linear drag (uniform Ekman friction if ``alpha_kmax == 0``, otherwise large-scale
    friction on ``|k| <= alpha_kmax`` only) and band-limited linear forcing (negative viscosity
    in a wavenumber band, Lundgren-style forcing in spectral space).

    ``form`` selects how the nonlinear term is evaluated (it fixes the FFT count):
    2D ``basdevant`` (4 FFTs/stage) or ``advective`` (5); 3D ``rotational`` u x omega (9)
    or ``basdevant`` traceless-stress divergence form (8). ``"auto"`` picks the backend's
    default.
    """

    n: int
    dim: int = 2
    nu: float = 1e-3
    alpha: float = 0.0
    alpha_kmax: float = 0.0
    force_amp: float = 0.0
    force_klo: float = 0.0
    force_khi: float = 0.0
    dtype: str = "float32"
    form: str = "auto"
    dealias: str = "2/3"

    def __post_init__(self) -> None:
        if self.dim not in (2, 3):
            raise ValueError("dim must be 2 or 3")
        if self.n < 8 or self.n % 2:
            raise ValueError("n must be even and >= 8")
        if self.dtype not in ("float32", "float64"):
            raise ValueError("dtype must be float32 or float64")
        forms = FORMS_2D if self.dim == 2 else FORMS_3D
        if self.form != "auto" and self.form not in forms:
            raise ValueError(f"form must be one of {forms} for dim={self.dim}")
        if self.dealias not in DEALIAS:
            raise ValueError(f"dealias must be one of {DEALIAS}")

    def resolved_form(self, default: str | None = None) -> str:
        """Return the concrete nonlinear-term form (``auto`` -> ``default`` or basdevant)."""
        if self.form != "auto":
            return self.form
        return default or "basdevant"

    @property
    def drag_k2(self) -> float:
        """Upper k^2 of the drag band (huge if the drag is uniform)."""
        return self.alpha_kmax**2 if self.alpha_kmax > 0 else 1e30

    def with_(self, **kw: object) -> Config:
        """Return a copy with some fields replaced."""
        return replace(self, **kw)  # type: ignore[arg-type]

    @property
    def is_fp64(self) -> bool:
        return self.dtype == "float64"


def ffts_per_stage(dim: int, form: str) -> int:
    """Number of real-to-complex/complex-to-real FFTs per right-hand-side evaluation."""
    table = {
        (2, "basdevant"): 4,  # 2 inverse (u, v) + 2 forward (v^2-u^2, uv)
        (2, "advective"): 5,  # 4 inverse (u, v, w_x, w_y) + 1 forward (u.grad w)
        (3, "basdevant"): 8,  # 3 inverse (u) + 5 forward (traceless u_i u_j)
        (3, "rotational"): 9,  # 6 inverse (u, omega) + 3 forward (u x omega)
    }
    return table[(dim, form)]


def fft_split(dim: int, form: str) -> tuple[int, int]:
    """(inverse, forward) FFT counts per stage."""
    table = {
        (2, "basdevant"): (2, 2),
        (2, "advective"): (4, 1),
        (3, "basdevant"): (3, 5),
        (3, "rotational"): (6, 3),
    }
    return table[(dim, form)]

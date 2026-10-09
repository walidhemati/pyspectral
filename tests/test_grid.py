import numpy as np

from pyspectral.config import Config, ffts_per_stage
from pyspectral.grid import dealias_mask, hermitian_weights, k_squared, spectral_shape


def test_shapes_and_mask():
    assert spectral_shape(16, 2) == (16, 9)
    assert spectral_shape(16, 3) == (16, 16, 9)
    m = dealias_mask(12, 2)
    # 3|k| < 12  ->  |k| <= 3
    assert m[3, 3] and not m[4, 0] and not m[0, 4]
    assert m.sum() == 7 * 4


def test_hermitian_weights_parseval():
    rng = np.random.default_rng(0)
    f = rng.standard_normal((16, 16))
    fh = np.fft.rfftn(f)
    w = hermitian_weights(16, 2)
    assert np.isclose(np.sum(w * np.abs(fh) ** 2) / 16**4, np.mean(f**2))


def test_k2_and_fft_counts():
    assert k_squared(8, 3)[1, 2, 3] == 1 + 4 + 9
    assert ffts_per_stage(2, "basdevant") == 4
    assert ffts_per_stage(3, "rotational") == 9
    assert Config(n=16, dim=3).resolved_form() == "basdevant"

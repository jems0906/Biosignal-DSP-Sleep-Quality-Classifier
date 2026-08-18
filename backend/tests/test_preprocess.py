import numpy as np
from app.ml.features import extract_bandpowers
from app.ml.preprocess import preprocess


def test_preprocess_preserves_shape_and_removes_dc():
    values = np.ones(1000) * 3 + np.sin(np.linspace(0, 20, 1000))
    result = preprocess(values, 100)
    assert result.shape == values.shape
    assert abs(result.mean()) < 0.2


def test_bandpowers_are_normalized():
    values = np.sin(2 * np.pi * 2 * np.arange(3000) / 100)
    _, _, powers = extract_bandpowers(values, 100)
    np.testing.assert_allclose(sum(powers.values()), 1.0, atol=0.01)
    assert powers['delta'] > powers['beta']

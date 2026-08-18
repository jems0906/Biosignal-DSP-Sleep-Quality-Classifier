import numpy as np

from app.ml.features import bandpower, welch_psd


def test_welch_returns_frequency_axis():
    values = np.sin(2 * np.pi * 10 * np.arange(3000) / 100)
    frequencies, psd = welch_psd(values, 100)
    assert frequencies.shape == psd.shape
    assert frequencies[0] == 0
    assert frequencies[-1] <= 50


def test_bandpower_ignores_out_of_band_values():
    frequencies = np.array([0, 2, 4, 6, 8, 12, 30, 40], dtype=float)
    psd = np.ones_like(frequencies)
    assert bandpower(frequencies, psd, 4, 8) > 0
    assert bandpower(frequencies, psd, 60, 80) == 0

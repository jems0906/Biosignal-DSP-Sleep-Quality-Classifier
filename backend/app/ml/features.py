import numpy as np
from scipy import signal

BANDS = {"delta": (0.5, 4), "theta": (4, 8), "alpha": (8, 12), "beta": (12, 30)}


def welch_psd(values: np.ndarray, sampling_rate: float) -> tuple[np.ndarray, np.ndarray]:
    return signal.welch(values, fs=sampling_rate, nperseg=min(1024, len(values)))


def bandpower(frequencies: np.ndarray, psd: np.ndarray, low: float, high: float) -> float:
    mask = (frequencies >= low) & (frequencies < high)
    if mask.sum() < 2:
        return 0.0
    return float(np.trapezoid(psd[mask], frequencies[mask]))


def extract_bandpowers(values: np.ndarray, sampling_rate: float) -> tuple[np.ndarray, np.ndarray, dict[str, float]]:
    frequencies, psd = welch_psd(values, sampling_rate)
    powers = {name: bandpower(frequencies, psd, low, high) for name, (low, high) in BANDS.items()}
    total = sum(powers.values()) or 1.0
    return frequencies, psd, {name: value / total for name, value in powers.items()}

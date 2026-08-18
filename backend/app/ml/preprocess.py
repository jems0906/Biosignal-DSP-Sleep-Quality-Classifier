import numpy as np
from scipy import signal


def bandpass_filter(values: np.ndarray, sampling_rate: float, low: float = 0.5, high: float = 30.0) -> np.ndarray:
    if values.size < 32:
        return values.astype(float)
    nyquist = sampling_rate / 2
    upper = min(high, nyquist * 0.95)
    if upper <= low:
        return values.astype(float)
    coefficients = signal.butter(4, [low / nyquist, upper / nyquist], btype="band")
    return signal.sosfiltfilt(signal.tf2sos(*coefficients), values)


def notch_filter(values: np.ndarray, sampling_rate: float, frequency: float = 60.0) -> np.ndarray:
    if values.size < 16 or frequency >= sampling_rate / 2:
        return values.astype(float)
    b, a = signal.iirnotch(frequency, 30, sampling_rate)
    return signal.filtfilt(b, a, values)


def preprocess(values: np.ndarray, sampling_rate: float) -> np.ndarray:
    return notch_filter(bandpass_filter(values, sampling_rate), sampling_rate)

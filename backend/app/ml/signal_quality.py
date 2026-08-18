import numpy as np


def assess_quality(values: np.ndarray, sampling_rate: float) -> dict[str, object]:
    amplitude = float(np.max(np.abs(values))) if values.size else 0.0
    flatline = bool(values.size > 1 and np.std(values) < 1e-8)
    high_amplitude = amplitude > 500.0
    reasons = []
    if flatline:
        reasons.append("flatline")
    if high_amplitude:
        reasons.append("amplitude threshold")
    return {"usable": not reasons, "artifact": bool(reasons), "amplitude": amplitude, "reasons": reasons, "sampling_rate": sampling_rate}

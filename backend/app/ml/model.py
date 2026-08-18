import numpy as np
from pathlib import Path

import joblib

LABELS = ["Wake", "N1", "N2", "N3", "REM"]
MODEL_PATH = Path(__file__).resolve().parents[2] / "models" / "model.pkl"


def _load_model():
    if not MODEL_PATH.exists():
        return None
    return joblib.load(MODEL_PATH)


def predict(bandpowers: dict[str, float], quality: dict[str, object]) -> dict[str, object]:
    if quality["artifact"]:
        return {"label": "Unusable", "confidence": 0.98, "probabilities": {label: 0.0 for label in LABELS}}
    features = np.array([[bandpowers[key] for key in ("delta", "theta", "alpha", "beta")]])
    trained_model = _load_model()
    if trained_model is not None:
        probabilities = trained_model.predict_proba(features)[0]
        classes = list(trained_model.classes_)
        probability_map = {label: round(float(probabilities[classes.index(label)]), 4) if label in classes else 0.0 for label in LABELS}
        label = str(trained_model.predict(features)[0])
        return {"label": label, "confidence": probability_map.get(label, 0.0), "probabilities": probability_map}
    delta, theta, alpha, beta = (bandpowers[key] for key in ("delta", "theta", "alpha", "beta"))
    scores = np.array([alpha + beta, theta * 1.4, theta + delta * 0.2, delta * 1.6, theta + alpha * 0.8])
    probabilities = np.exp(scores - scores.max())
    probabilities /= probabilities.sum()
    index = int(probabilities.argmax())
    return {"label": LABELS[index], "confidence": float(probabilities[index]), "probabilities": {label: round(float(probabilities[i]), 4) for i, label in enumerate(LABELS)}}

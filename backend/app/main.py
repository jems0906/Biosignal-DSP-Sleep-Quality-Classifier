import csv
import io
import json
from pathlib import Path

import numpy as np
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware

from .ml.features import extract_bandpowers
from .ml.model import predict
from .ml.preprocess import preprocess
from .ml.signal_quality import assess_quality

app = FastAPI(title="Somnus Signal Lab", version="1.0.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])
METRICS_PATH = Path(__file__).resolve().parents[1] / "models" / "metrics.json"


def demo_signal() -> tuple[np.ndarray, float]:
    sampling_rate = 100.0
    time = np.arange(0, 30, 1 / sampling_rate)
    values = 34 * np.sin(2 * np.pi * 2.2 * time) + 10 * np.sin(2 * np.pi * 8 * time) + 4 * np.random.default_rng(7).normal(size=time.size)
    return values, sampling_rate


def parse_csv(content: bytes) -> tuple[np.ndarray, float]:
    try:
        rows = list(csv.reader(io.StringIO(content.decode("utf-8"))))
        values = np.array([float(row[-1]) for row in rows if row and row[-1].strip()], dtype=float)
        if not values.size:
            raise ValueError("CSV contains no numeric signal values")
        return values, 100.0
    except (UnicodeDecodeError, ValueError) as error:
        raise HTTPException(status_code=400, detail=f"Could not parse CSV signal: {error}") from error


def analyze(values: np.ndarray, sampling_rate: float, sample_name: str) -> dict:
    values = values[: min(values.size, 30000)]
    filtered = preprocess(values, sampling_rate)
    frequencies, psd, powers = extract_bandpowers(filtered, sampling_rate)
    quality = assess_quality(values, sampling_rate)
    _, _, stft = __import__("scipy").signal.stft(filtered, fs=sampling_rate, nperseg=min(256, values.size))
    downsample = max(1, values.size // 600)
    return {"sample_name": sample_name, "sampling_rate": sampling_rate, "duration_seconds": round(values.size / sampling_rate, 2), "channels": ["EEG Fpz-Cz"], "raw_signal": values[::downsample].round(3).tolist(), "filtered_signal": filtered[::downsample].round(3).tolist(), "frequencies": frequencies[:300].round(2).tolist(), "psd": psd[:300].round(6).tolist(), "bandpowers": {key: round(value, 4) for key, value in powers.items()}, "quality": quality, "prediction": predict(powers, quality), "spectrogram": np.abs(stft[:80, :80]).round(5).tolist()}


@app.get("/api/health")
def health() -> dict[str, str]:
    return {"status": "ok", "service": "somnus-signal-lab"}


@app.get("/api/samples")
def samples() -> list[dict[str, str]]:
    return [{"id": "demo-n2", "name": "Demo / N2-like epoch", "description": "Synthetic 30 s EEG with dominant delta and theta activity"}, {"id": "demo-wake", "name": "Demo / Wake-like epoch", "description": "Synthetic epoch with alpha and beta activity"}]


@app.get("/api/model-card")
def model_card() -> dict:
    if not METRICS_PATH.exists():
        return {"status": "not_trained", "message": "Run backend/scripts/train_model.py to create held-out subject metrics."}
    return json.loads(METRICS_PATH.read_text(encoding="utf-8"))


@app.get("/api/demo/{sample_id}")
def demo(sample_id: str) -> dict:
    if sample_id not in {"demo-n2", "demo-wake"}:
        raise HTTPException(status_code=404, detail="Sample not found")
    values, rate = demo_signal()
    if sample_id == "demo-wake":
        time = np.arange(0, 30, 1 / rate)
        values = 24 * np.sin(2 * np.pi * 10 * time) + 8 * np.sin(2 * np.pi * 18 * time) + 4 * np.random.default_rng(9).normal(size=time.size)
    return analyze(values, rate, sample_id)


@app.post("/api/analyze")
async def analyze_upload(file: UploadFile = File(...)) -> dict:
    content = await file.read()
    if Path(file.filename or "").suffix.lower() == ".csv":
        values, rate = parse_csv(content)
    else:
        try:
            from .mne_compat import mne
            raw = mne.io.read_raw_edf(io.BytesIO(content), preload=True, verbose=False)
            values = raw.get_data(picks="eeg")[0]
            rate = float(raw.info["sfreq"])
        except Exception as error:
            raise HTTPException(status_code=400, detail=f"EDF could not be read: {error}") from error
    return analyze(values, rate, file.filename or "uploaded-signal")

from pydantic import BaseModel, Field


class Prediction(BaseModel):
    label: str
    confidence: float = Field(ge=0, le=1)
    probabilities: dict[str, float]


class AnalysisResponse(BaseModel):
    sample_name: str
    sampling_rate: float
    duration_seconds: float
    channels: list[str]
    raw_signal: list[float]
    filtered_signal: list[float]
    frequencies: list[float]
    psd: list[float]
    bandpowers: dict[str, float]
    quality: dict[str, object]
    prediction: Prediction
    spectrogram: list[list[float]]

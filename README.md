# Somnus Signal Lab

Production-style biosignal algorithm lifecycle demo: curation, DSP preprocessing, quality control, interpretable sleep-stage inference, validation, failure analysis, and deployment.

## Run locally

### Backend

```powershell
python -m pip install -r backend/requirements.txt
python -m uvicorn app.main:app --app-dir backend --reload --port 8000
```

The API exposes interactive docs at `http://localhost:8000/docs`.

### Frontend

```powershell
Set-Location frontend
npm install
npm run dev
```

Set `VITE_API_URL` when the API is not at `http://localhost:8000`.

## What is included

- FastAPI endpoints for demo epochs and EDF/CSV uploads.
- MNE EDF ingestion, Butterworth 0.5-30 Hz bandpass, 60 Hz notch, Welch PSD, normalized bandpower, and artifact gates.
- React/Vite dashboard with raw vs filtered signal, frequency-band chart, quality gate, confidence, upload control, and model card view.
- Subject-level split plan in `notebooks/train_model.ipynb`; PhysioNet Sleep-EDF is the intended source dataset.
- Unit/API tests, model card, failure analysis, Dockerfiles, Railway config, and GitHub Actions CI.

The built-in demo predictor is intentionally transparent and deterministic. Replace `backend/app/ml/model.py` with a trained `joblib` artifact after curating Sleep-EDF epochs and evaluating on held-out subjects. This project is not a medical device.

## Train on Sleep-EDF

After installing the backend dependencies, run a small subject subset first:

```powershell
python backend/scripts/train_model.py --subjects 0 1 2 3 4
```

The command downloads Sleep-EDF through MNE, extracts EEG epochs and bandpower features, performs a subject-level 70/15/15 split, trains a balanced RandomForest, and writes `backend/models/model.pkl` plus `backend/models/metrics.json`. The API loads the artifact automatically, and `/api/model-card` exposes the held-out metrics and confusion matrix to the dashboard.

## Test

```powershell
python -m pytest backend/tests -q
```

## Deployment

`railway.json` defines separate backend and frontend Docker services. Configure the frontend service variable `VITE_API_URL` with the public backend URL before building.

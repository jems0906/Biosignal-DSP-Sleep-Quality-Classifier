# Somnus Signal Lab Model Card

## Intended use
This project demonstrates an interpretable sleep-stage screening workflow for 30-second EEG epochs. It is intended for engineering evaluation, algorithm prototyping, and quality-control review. It is not a medical device and must not be used for diagnosis or treatment decisions.

## Data and split
The target dataset is PhysioNet Sleep-EDF Expanded: 197 whole-night recordings with expert-scored 30-second epochs. Training, validation, and test sets are split by subject (70/15/15) to avoid epoch-level leakage. A production training run should record subject IDs, recording provenance, channel montage, and rejected epochs.

## Inputs and method
EDF or single-column CSV signals are ingested with MNE, bandpass filtered from 0.5 to 30 Hz, notch filtered at 60 Hz when supported by the sampling rate, and transformed with Welch PSD. Relative delta, theta, alpha, and beta bandpower are the features. The demo fallback uses a small rule-informed probability model; a serialized Logistic Regression or Random Forest artifact can replace it in `app/ml/model.py`.

## Limitations
N1 is transitional and difficult to separate from Wake/N2. Movement and electrode artifacts can dominate bandpower. Performance may shift with device, montage, sampling rate, and population. Confidence is model confidence, not clinical certainty.

## Validation workflow
Run `python backend/scripts/train_model.py --subjects 0 1 2 3 4` to download a manageable Sleep-EDF subset and generate `backend/models/metrics.json`. The workflow reports macro F1, balanced accuracy, per-class recall, confusion matrix, and subject counts from the held-out test partition. A production review should additionally report calibration error, confidence intervals, and signal-quality stratification before deployment.

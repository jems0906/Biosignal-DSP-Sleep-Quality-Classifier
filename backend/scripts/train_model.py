"""Train a subject-level RandomForest on PhysioNet Sleep-EDF.

Run from the repository root after installing backend/requirements.txt:
    python backend/scripts/train_model.py --subjects 0 1 2 3 4
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import balanced_accuracy_score, classification_report, confusion_matrix, f1_score

ROOT = Path(__file__).resolve().parents[2]
BACKEND = ROOT / "backend"
sys.path.insert(0, str(BACKEND))
from app.ml.features import extract_bandpowers
from app.mne_compat import mne
from app.ml.preprocess import preprocess

STAGES = {f"Sleep stage {stage}": stage for stage in ("W", "1", "2", "3", "4", "R")}
LABELS = ["Wake", "N1", "N2", "N3", "REM"]
STAGE_LABELS = {"W": "Wake", "1": "N1", "2": "N2", "3": "N3", "4": "N3", "R": "REM"}


def epoch_recording(psg_path: str, hypnogram_path: str) -> tuple[list[list[float]], list[str], list[str]]:
    raw = mne.io.read_raw_edf(psg_path, preload=True, verbose=False)
    annotations = mne.read_annotations(hypnogram_path)
    raw.set_annotations(annotations)
    eeg_picks = mne.pick_types(raw.info, eeg=True, exclude="bads")
    if not len(eeg_picks):
        return [], [], []
    signal_values = raw.get_data(picks=[eeg_picks[0]])[0]
    sampling_rate = float(raw.info["sfreq"])
    features, labels, subjects = [], [], []
    subject = Path(psg_path).stem.split("-")[0]
    for annotation in annotations:
        stage = STAGE_LABELS.get(annotation["description"].replace("Sleep stage ", ""))
        if stage is None:
            continue
        start = max(0, int(annotation["onset"] * sampling_rate))
        stop = start + int(30 * sampling_rate)
        epoch = signal_values[start:stop]
        if epoch.size < int(30 * sampling_rate):
            continue
        filtered = preprocess(epoch, sampling_rate)
        _, _, powers = extract_bandpowers(filtered, sampling_rate)
        features.append([powers[name] for name in ("delta", "theta", "alpha", "beta")])
        labels.append(stage)
        subjects.append(subject)
    return features, labels, subjects


def split_subjects(features: np.ndarray, labels: np.ndarray, subjects: np.ndarray) -> tuple[np.ndarray, ...]:
    unique_subjects = np.unique(subjects)
    rng = np.random.default_rng(42)
    rng.shuffle(unique_subjects)
    if len(unique_subjects) < 3:
        raise ValueError("At least three subjects are required for train/validation/test splitting")
    if len(unique_subjects) <= 4:
        train_subjects = unique_subjects[:-2]
        validation_subjects = unique_subjects[-2:-1]
        test_subjects = unique_subjects[-1:]
    else:
        train_count = max(1, int(round(len(unique_subjects) * 0.70)))
        validation_count = max(1, int(round(len(unique_subjects) * 0.15)))
        train_subjects = unique_subjects[:train_count]
        validation_subjects = unique_subjects[train_count:train_count + validation_count]
        test_subjects = unique_subjects[train_count + validation_count:]
    train_index = np.flatnonzero(np.isin(subjects, train_subjects))
    validation_index = np.flatnonzero(np.isin(subjects, validation_subjects))
    test_index = np.flatnonzero(np.isin(subjects, test_subjects))
    return train_index, validation_index, test_index


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--subjects", nargs="+", type=int, default=list(range(10)))
    args = parser.parse_args()
    print(f"Fetching Sleep-EDF subjects: {args.subjects}")
    recordings = mne.datasets.sleep_physionet.age.fetch_data(subjects=args.subjects, recording=[1], on_missing="warn")
    all_features, all_labels, all_subjects = [], [], []
    for psg_path, hypnogram_path in recordings:
        features, labels, subjects = epoch_recording(psg_path, hypnogram_path)
        all_features.extend(features)
        all_labels.extend(labels)
        all_subjects.extend(subjects)
    if len(set(all_subjects)) < 3:
        raise RuntimeError("At least three subjects are required for train/validation/test splitting")
    features = np.asarray(all_features, dtype=float)
    labels = np.asarray(all_labels)
    subjects = np.asarray(all_subjects)
    train_index, validation_index, test_index = split_subjects(features, labels, subjects)
    model = RandomForestClassifier(n_estimators=300, class_weight="balanced", random_state=42, n_jobs=-1)
    model.fit(features[train_index], labels[train_index])
    test_predictions = model.predict(features[test_index])
    report = classification_report(labels[test_index], test_predictions, labels=LABELS, output_dict=True, zero_division=0)
    metrics = {"status": "trained", "dataset": "PhysioNet Sleep-EDF Expanded", "subjects": int(len(set(subjects))), "epochs": int(len(labels)), "split": {"train_subjects": int(len(set(subjects[train_index]))), "validation_subjects": int(len(set(subjects[validation_index]))), "test_subjects": int(len(set(subjects[test_index])))}, "balanced_accuracy": float(balanced_accuracy_score(labels[test_index], test_predictions)), "macro_f1": float(f1_score(labels[test_index], test_predictions, labels=LABELS, average="macro")), "classification_report": report, "confusion_matrix": confusion_matrix(labels[test_index], test_predictions, labels=LABELS).tolist(), "labels": LABELS}
    models_dir = BACKEND / "models"
    models_dir.mkdir(exist_ok=True)
    joblib.dump(model, models_dir / "model.pkl")
    (models_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()

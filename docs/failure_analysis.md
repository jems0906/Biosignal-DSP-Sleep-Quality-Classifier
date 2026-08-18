# Failure Analysis

The lifecycle review focuses on errors that are actionable rather than only aggregate accuracy.

| Failure mode | Observable symptom | Likely cause | Mitigation |
| --- | --- | --- | --- |
| N1 predicted as Wake/N2 | Low-confidence transition epochs | N1 has no stable spectral signature | Add temporal context and report class-specific recall |
| Movement predicted as N3 | Excess low-frequency energy | EMG and electrode motion contaminate PSD | Quality gate, artifact rejection, and channel review |
| REM predicted as Wake | Similar broadband energy | EOG/EMG context is absent from a single EEG channel | Add EOG and chin EMG features |
| Flatline accepted | Near-zero PSD and unstable probabilities | Disconnected electrode or export issue | Flatline detector blocks inference |
| Device shift | Consistent per-site drift | Montage/amplifier differences | Site-level calibration and subgroup reports |

## Review protocol
For each false prediction, retain the subject, epoch index, channel, raw/filtered trace, quality flags, bandpower vector, predicted probabilities, and expert label. Review the highest-confidence errors first, then stratify by stage and artifact status. Never tune thresholds on the held-out subject set.

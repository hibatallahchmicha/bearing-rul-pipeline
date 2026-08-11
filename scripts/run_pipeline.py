"""End-to-end pipeline: wire all 10 bearing_pdm modules together on real
IMS vibration data, evaluated with leave-one-bearing-out cross-validation.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from bearing_pdm.cost_threshold import best_threshold, sweep_thresholds
from bearing_pdm.evaluation import leave_one_unit_out
from bearing_pdm.fault_frequencies import compute_fault_frequencies
from bearing_pdm.features_freq import frequency_domain_features
from bearing_pdm.features_time import time_domain_features
from bearing_pdm.health_index import fit_healthy_baseline, mahalanobis_health_index, stack_features
from bearing_pdm.ingestion import list_snapshot_files, read_snapshot
from bearing_pdm.rul_model import fit_stage_thresholds, fraction_of_life_remaining, predict_stage, true_stage
from bearing_pdm.segmentation import window_signal

DATA_ROOT = Path("data/raw/4. Bearings")
SHAFT_SPEED_RPM = 2000
FS_HZ = 20000
HEALTHY_FRACTION = 0.2   # first 20% of a bearing's life used for its baseline contribution

# Read every Nth snapshot file instead of all of them -- development-speed
# tradeoff, not a hidden shortcut. Set to 1 for a final full-fidelity run.
SNAPSHOT_STRIDE = 10

# (unit_id, snapshot_dir, channel_index) -- channel per NASA's documented
# bearing->column mapping. Test 1 has 2 channels/bearing; the first of each
# pair is used here for simplicity (both are valid, this is a scope choice).
UNITS = [
    ("1st_test_b3", DATA_ROOT / "1st_test" / "1st_test", 4),
    ("1st_test_b4", DATA_ROOT / "1st_test" / "1st_test", 6),
    ("2nd_test_b1", DATA_ROOT / "2nd_test" / "2nd_test", 0),
    ("3rd_test_b3", DATA_ROOT / "3rd_test" / "4th_test" / "txt", 2),
]


def build_unit_features(snapshot_dir: Path, channel: int, faults) -> dict[str, np.ndarray]:
    files = list_snapshot_files(snapshot_dir)[::SNAPSHOT_STRIDE]
    rows = []
    for i, path in enumerate(files):
        signal = read_snapshot(path)[:, channel]
        windows = window_signal(signal)
        merged = {**time_domain_features(windows), **frequency_domain_features(windows, fs=FS_HZ, faults=faults)}
        rows.append(merged)
        if (i + 1) % 25 == 0 or (i + 1) == len(files):
            print(f"    {i + 1}/{len(files)} snapshot files processed")

    keys = rows[0].keys()
    return {k: np.concatenate([r[k] for r in rows]) for k in keys}


def evaluate_fixed_threshold(scores: np.ndarray, is_critical: np.ndarray, threshold: float) -> tuple[float, float]:
    """False-alarm and missed-failure rate for one FIXED threshold (not swept).

    Used to score a threshold chosen from training data against a held-out
    unit, instead of re-optimizing the threshold on the unit being tested.
    """
    predicted_critical = scores >= threshold
    false_alarms = np.sum(predicted_critical & ~is_critical)
    missed_failures = np.sum(~predicted_critical & is_critical)
    return (
        false_alarms / max(np.sum(~is_critical), 1),
        missed_failures / max(np.sum(is_critical), 1),
    )


def normalize_to_own_baseline(matrix: np.ndarray, healthy_fraction: float = HEALTHY_FRACTION) -> np.ndarray:
    """Z-score a unit's own feature matrix against its own early-life mean/std.

    Different physical bearings -- even nominally identical ones -- have
    different noise floors, mounting stiffness, and sensor gain, so raw
    feature values aren't directly comparable across bearings (this is the
    mismatch flagged in the health_index.py checkpoint). This is
    legitimate, deployable per-asset calibration -- every new bearing gets
    a short healthy break-in period to calibrate against -- not leakage:
    it never uses knowledge of when or whether the bearing fails.
    """
    n_healthy = max(int(len(matrix) * healthy_fraction), 1)
    baseline = matrix[:n_healthy]
    mean = baseline.mean(axis=0)
    std = baseline.std(axis=0)
    std[std == 0] = 1.0
    return (matrix - mean) / std


def run() -> None:
    faults = compute_fault_frequencies(shaft_speed_rpm=SHAFT_SPEED_RPM)

    print("Extracting features per unit (reads real vibration data)...")
    unit_matrices, unit_fraction = {}, {}
    for unit_id, snapshot_dir, channel in UNITS:
        print(f"  {unit_id} ...")
        feats = build_unit_features(snapshot_dir, channel, faults)
        matrix, _ = stack_features(feats)
        unit_matrices[unit_id] = normalize_to_own_baseline(matrix)
        unit_fraction[unit_id] = fraction_of_life_remaining(len(matrix))
        print(f"    {len(matrix)} windows")

    unit_ids = [u[0] for u in UNITS]
    accuracies = []

    for train_units, test_unit in leave_one_unit_out(unit_ids):
        healthy_chunks, train_matrices, train_fractions = [], [], []
        for u in train_units:
            matrix = unit_matrices[u]
            healthy_chunks.append(matrix[: int(len(matrix) * HEALTHY_FRACTION)])
            train_matrices.append(matrix)
            train_fractions.append(unit_fraction[u])

        mean, inv_cov = fit_healthy_baseline(np.concatenate(healthy_chunks))
        train_scores = [mahalanobis_health_index(m, mean, inv_cov) for m in train_matrices]
        thresholds = fit_stage_thresholds(train_scores, train_fractions)

        test_matrix = unit_matrices[test_unit]
        test_scores = mahalanobis_health_index(test_matrix, mean, inv_cov)
        actual = true_stage(unit_fraction[test_unit])
        predicted = predict_stage(test_scores, thresholds)

        accuracy = (predicted == actual).mean()
        accuracies.append(accuracy)

        train_critical_labels = [true_stage(f) == "critical" for f in train_fractions]
        pooled_train_scores = np.concatenate(train_scores)
        pooled_train_critical = np.concatenate(train_critical_labels)
        cost_result = best_threshold(sweep_thresholds(pooled_train_scores, pooled_train_critical))

        test_is_critical = actual == "critical"
        false_alarm_rate, missed_failure_rate = evaluate_fixed_threshold(
            test_scores, test_is_critical, cost_result.threshold
        )

        print(
            f"fold test={test_unit}: stage accuracy={accuracy:.3f} | "
            f"train-chosen alarm threshold={cost_result.threshold:.2f} "
            f"(false_alarm_rate={false_alarm_rate:.3f}, "
            f"missed_failure_rate={missed_failure_rate:.3f})"
        )

    print(f"\nmean stage accuracy across {len(unit_ids)} leave-one-bearing-out folds: {np.mean(accuracies):.3f}")


if __name__ == "__main__":
    run()
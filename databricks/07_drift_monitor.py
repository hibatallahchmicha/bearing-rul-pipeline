# Databricks notebook source
# MAGIC %md
# MAGIC # Drift monitoring — change that is not wear
# MAGIC
# MAGIC In predictive maintenance, feature distributions are *supposed* to change:
# MAGIC that is what degradation looks like. A naive drift monitor comparing all
# MAGIC incoming data to training data fires whenever a bearing starts to fail —
# MAGIC exactly when the model matters most.
# MAGIC
# MAGIC So the comparison is conditioned on health:
# MAGIC
# MAGIC - **Reference:** training bearings while truly healthy (true RUL at the cap).
# MAGIC - **Current:** bearings never used for training, at snapshots the champion
# MAGIC   predicts are healthy.
# MAGIC
# MAGIC A difference between those cannot be wear — the model says these bearings
# MAGIC are not wearing. It is something else: a recalibrated sensor, a different
# MAGIC bearing type, a different operating run.
# MAGIC
# MAGIC **Effect size, not significance.** Drift is measured by Population
# MAGIC Stability Index, with the conventional 0.25 threshold. An earlier
# MAGIC version used KS p-values and flagged 16 of 17 features on data with no
# MAGIC drift: with thousands of snapshots, a significance test detects
# MAGIC differences far too small to matter.
# MAGIC
# MAGIC **Positive control.** A simulated mid-life sensor recalibration (+30%
# MAGIC gain) shows whether the monitor detects drift, and whether it detects the
# MAGIC right features.
# MAGIC
# MAGIC **Known limitation.** The reference is read from current Gold, not frozen
# MAGIC at training time. The proper fix is to save a reference sample with the
# MAGIC model at registration.

# COMMAND ----------

import os
import sys

REPO_ROOT = os.path.dirname(os.getcwd())
if REPO_ROOT not in sys.path:
    sys.path.append(REPO_ROOT)

import json
from datetime import UTC, datetime

import mlflow
import pandas as pd
from mlflow.tracking import MlflowClient

from bearing_pdm.drift import (
    DRIFTED_FEATURE_FRACTION_ALARM,
    PSI_DRIFT_THRESHOLD,
    DriftReport,
    detect_drift,
    should_trigger_retrain,
)
from bearing_pdm.feature_matrix import build_sensor_matrix

CATALOG = "bearing"
GOLD_TABLE = f"{CATALOG}.gold.features"
PREDICTIONS_TABLE = f"{CATALOG}.gold.rul_predictions"
DRIFT_TABLE = f"{CATALOG}.gold.drift_reports"
MODEL_NAME = f"{CATALOG}.ml.rul_regressor"

# A snapshot counts as "model says healthy" when its predicted RUL is at
# least 90% of the cap -- far above any alert. A stated choice, not tuned.
HEALTHY_PREDICTION_FRACTION = 0.9

# Positive control: a sensor recalibrated mid-life to read 30% higher.
SIMULATED_GAIN = 1.3

mlflow.set_tracking_uri("databricks")
mlflow.set_registry_uri("databricks-uc")
client = MlflowClient()

# COMMAND ----------

champion = client.get_model_version_by_alias(MODEL_NAME, "champion")
model_uri = f"models:/{MODEL_NAME}@champion"
feature_columns = mlflow.models.get_model_info(model_uri).signature.inputs.input_names()

training_params = client.get_run(champion.run_id).data.params
healthy_fraction = float(training_params["healthy_fraction"])
rul_cap_hours = float(training_params["rul_cap_hours"])
training_gold_version = training_params["gold_table_version"]

gold_version = spark.sql(f"DESCRIBE HISTORY {GOLD_TABLE} LIMIT 1").first()["version"]
if str(gold_version) != training_gold_version:
    print(
        f"WARNING: Gold is at version {gold_version}, but the champion trained on {training_gold_version}. "
        "The reference below reflects current Gold, not the exact training data."
    )
print(f"champion v{champion.version}, {len(feature_columns)} features, Gold version {gold_version}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Reference and current populations

# COMMAND ----------

gold = spark.sql(f"SELECT * FROM {GOLD_TABLE} VERSION AS OF {gold_version}").toPandas()

latest_predictions = spark.sql(
    f"""
    SELECT bearing_id, snapshot_ts, predicted_rul_hours, model_version
    FROM {PREDICTIONS_TABLE}
    WHERE scored_at = (SELECT MAX(scored_at) FROM {PREDICTIONS_TABLE})
    """
).toPandas()

scored_version = int(latest_predictions["model_version"].iloc[0])
assert scored_version == int(champion.version), (
    f"latest predictions came from v{scored_version}, champion is v{champion.version} -- run 06_batch_score first"
)


def health_conditioned_populations(gold_frame):
    """Normalize, join the latest predictions, and mark both populations."""
    ordered, matrix = build_sensor_matrix(gold_frame, feature_columns, healthy_fraction=healthy_fraction)
    joined = ordered.merge(latest_predictions, on=["bearing_id", "snapshot_ts"], how="left")
    assert len(joined) == len(ordered), "prediction join changed the row count"

    reference_rows = (joined["runs_to_failure"] & (joined["rul_hours_capped"] >= rul_cap_hours)).to_numpy()
    current_rows = (
        ~joined["runs_to_failure"]
        & (joined["predicted_rul_hours"] >= HEALTHY_PREDICTION_FRACTION * rul_cap_hours)
    ).to_numpy()
    return joined, matrix, reference_rows, current_rows


def as_feature_dict(matrix, rows):
    return {name: matrix[rows, i] for i, name in enumerate(feature_columns)}


joined, matrix, reference_rows, current_rows = health_conditioned_populations(gold)
reference = as_feature_dict(matrix, reference_rows)
reference_bearings = sorted(joined.loc[reference_rows, "bearing_id"].unique())
current_bearings = sorted(joined.loc[current_rows, "bearing_id"].unique())
print(f"reference: {reference_rows.sum()} healthy training snapshots from {reference_bearings}")
print(f"current:   {current_rows.sum()} model-says-healthy new snapshots from {current_bearings}")
# COMMAND ----------

# MAGIC %md
# MAGIC ## Observed drift — all new bearings, and per test set
# MAGIC
# MAGIC The reference holds test-1 bearings only. Test 2 was a separate run on a
# MAGIC different date; if its bearings drift and test 1's do not, the monitor is
# MAGIC detecting a real change in operating conditions.

# COMMAND ----------

observed = detect_drift(reference, as_feature_dict(matrix, current_rows))

by_test_set = {}
for test_set in sorted(joined.loc[current_rows, "test_set"].unique()):
    rows = current_rows & (joined["test_set"] == test_set).to_numpy()
    by_test_set[test_set] = detect_drift(reference, as_feature_dict(matrix, rows))

print(f"PSI threshold {PSI_DRIFT_THRESHOLD}; alarm when >= {DRIFTED_FEATURE_FRACTION_ALARM:.0%} of features drift")
n_obs = f"{len(observed.drifted_features)}/{observed.n_features}"
print(f"observed, all new bearings: {n_obs} drifted -> {observed.drift_detected}")
for test_set, report in by_test_set.items():
    n_set = f"{len(report.drifted_features)}/{report.n_features}"
    print(f"observed, {test_set} only:    {n_set} drifted -> {report.drift_detected}")
# COMMAND ----------

# MAGIC %md
# MAGIC ## Positive control: simulated sensor recalibration
# MAGIC
# MAGIC From halfway through their life, the new bearings' sensor gain rises 30%.
# MAGIC Amplitude features scale by the gain and energies by its square, but
# MAGIC shape features (kurtosis, skewness, crest factor) and frequencies are
# MAGIC scale-invariant. A correct monitor shows a jump in PSI for the first
# MAGIC group relative to the observed run, and little change for the second.

# COMMAND ----------

amplitude_cols = [c for c in ["rms", "std", "peak_to_peak"] if c in feature_columns]
energy_cols = [c for c in feature_columns if c.endswith("energy")]
scale_dependent = set(amplitude_cols) | set(energy_cols)

simulated_gold = gold.copy()
life_fraction = simulated_gold["elapsed_hours"] / simulated_gold.groupby("bearing_id")["elapsed_hours"].transform("max")
affected = (~simulated_gold["runs_to_failure"]) & (life_fraction > 0.5)
simulated_gold.loc[affected, amplitude_cols] *= SIMULATED_GAIN
simulated_gold.loc[affected, energy_cols] *= SIMULATED_GAIN**2

_, sim_matrix, sim_reference_rows, sim_current_rows = health_conditioned_populations(simulated_gold)
simulated = detect_drift(
    as_feature_dict(sim_matrix, sim_reference_rows),
    as_feature_dict(sim_matrix, sim_current_rows),
)
print(f"simulated: {len(simulated.drifted_features)}/{simulated.n_features} drifted -> {simulated.drift_detected}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## PSI per feature
# MAGIC
# MAGIC Above 0.25 is a significant shift. Compare `simulated_gain` with
# MAGIC `observed_all`: the scale-dependent rows should rise, the
# MAGIC scale-invariant rows should not.

# COMMAND ----------

psi_table = pd.DataFrame(
    {
        "feature": feature_columns,
        "scale_dependent": [f in scale_dependent for f in feature_columns],
        "observed_all": [observed.psi[f] for f in feature_columns],
        **{f"observed_{ts}": [r.psi[f] for f in feature_columns] for ts, r in by_test_set.items()},
        "simulated_gain": [simulated.psi[f] for f in feature_columns],
    }
)
psi_table["simulated_minus_observed"] = psi_table["simulated_gain"] - psi_table["observed_all"]
display(psi_table.sort_values(["scale_dependent", "feature"]).round(3))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Record the reports
# MAGIC
# MAGIC Only the `observed` scenario feeds the retrain trigger. Per-test-set and
# MAGIC simulated reports are kept for diagnosis.

# COMMAND ----------

checked_at = datetime.now(UTC)
scenarios = [("observed", observed)]
scenarios += [(f"observed_{ts}", r) for ts, r in by_test_set.items()]
scenarios += [(f"simulated_gain_{SIMULATED_GAIN}", simulated)]

reports = pd.DataFrame(
    [
        {
            "checked_at": checked_at,
            "scenario": scenario,
            "model_version": int(champion.version),
            "gold_table_version": int(gold_version),
            "n_features": report.n_features,
            "drifted_features": report.drifted_features,
            "drift_detected": report.drift_detected,
            "psi_json": json.dumps({k: round(v, 4) for k, v in report.psi.items()}),
            "psi_threshold": PSI_DRIFT_THRESHOLD,
            "drifted_fraction_alarm": DRIFTED_FEATURE_FRACTION_ALARM,
        }
        for scenario, report in scenarios
    ]
)
spark.createDataFrame(reports).write.format("delta").mode("append").saveAsTable(DRIFT_TABLE)
print(f"appended {len(reports)} reports to {DRIFT_TABLE}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Retrain trigger
# MAGIC
# MAGIC Recommends a retrain only after three consecutive observed drift reports,
# MAGIC and never retrains by itself: retraining redefines "healthy", which
# MAGIC needs a human to confirm the drift is real and not a fault.

# COMMAND ----------

history = (
    spark.table(DRIFT_TABLE)
    .filter("scenario = 'observed'")
    .orderBy("checked_at")
    .toPandas()
)
recent_reports = [
    DriftReport(
        drifted_features=list(row.drifted_features),
        n_features=int(row.n_features),
        drift_detected=bool(row.drift_detected),
    )
    for row in history.itertuples()
]
print(f"{len(recent_reports)} observed reports on record")
print(f"retrain recommended: {should_trigger_retrain(recent_reports)}")
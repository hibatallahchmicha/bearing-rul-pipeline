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
# MAGIC A difference between those cannot be wear — the model says these
# MAGIC bearings are not wearing. It is something else: a recalibrated sensor, a
# MAGIC different bearing type, a pipeline change.
# MAGIC
# MAGIC **Positive control.** No drift is expected in a single-lab dataset, and
# MAGIC "no drift" proves nothing unless the monitor is shown to detect drift. A
# MAGIC simulated mid-life sensor recalibration (+30% gain) is run through the
# MAGIC same test.
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

from datetime import datetime, timezone

import mlflow
import numpy as np
import pandas as pd
from mlflow.tracking import MlflowClient

from bearing_pdm.drift import (
    DRIFTED_FEATURE_FRACTION_ALARM,
    P_VALUE_THRESHOLD,
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
    """Normalize, then split into the reference and current populations."""
    ordered, matrix = build_sensor_matrix(gold_frame, feature_columns, healthy_fraction=healthy_fraction)
    joined = ordered.merge(latest_predictions, on=["bearing_id", "snapshot_ts"], how="left")
    assert len(joined) == len(ordered), "prediction join changed the row count"

    reference_rows = (joined["runs_to_failure"] & (joined["rul_hours_capped"] >= rul_cap_hours)).to_numpy()
    current_rows = (
        ~joined["runs_to_failure"]
        & (joined["predicted_rul_hours"] >= HEALTHY_PREDICTION_FRACTION * rul_cap_hours)
    ).to_numpy()

    reference = {name: matrix[reference_rows, i] for i, name in enumerate(feature_columns)}
    current = {name: matrix[current_rows, i] for i, name in enumerate(feature_columns)}
    return reference, current, int(reference_rows.sum()), int(current_rows.sum())


reference, current, n_reference, n_current = health_conditioned_populations(gold)
print(f"reference: {n_reference} healthy training snapshots; current: {n_current} model-says-healthy new snapshots")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Observed drift

# COMMAND ----------

observed = detect_drift(reference, current)
print(f"observed: {len(observed.drifted_features)}/{observed.n_features} features drifted -> drift_detected={observed.drift_detected}")
print(f"drifted: {observed.drifted_features}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Positive control: simulated sensor recalibration
# MAGIC
# MAGIC From halfway through their life, the new bearings' sensor gain rises 30%.
# MAGIC Amplitude features scale by the gain and energies by its square, but
# MAGIC shape features (kurtosis, skewness, crest factor) and frequencies are
# MAGIC scale-invariant. A correct monitor flags the first group and not the
# MAGIC second. Calibration uses the first 20% of life, before the change, so
# MAGIC normalization cannot absorb it.

# COMMAND ----------

amplitude_cols = [c for c in ["rms", "std", "peak_to_peak"] if c in feature_columns]
energy_cols = [c for c in feature_columns if c.endswith("energy")]

simulated_gold = gold.copy()
life_fraction = simulated_gold["elapsed_hours"] / simulated_gold.groupby("bearing_id")["elapsed_hours"].transform("max")
affected = (~simulated_gold["runs_to_failure"]) & (life_fraction > 0.5)
simulated_gold.loc[affected, amplitude_cols] *= SIMULATED_GAIN
simulated_gold.loc[affected, energy_cols] *= SIMULATED_GAIN**2

sim_reference, sim_current, _, _ = health_conditioned_populations(simulated_gold)
simulated = detect_drift(sim_reference, sim_current)

expected_changed = set(amplitude_cols) | set(energy_cols)
print(f"simulated: {len(simulated.drifted_features)}/{simulated.n_features} features drifted -> drift_detected={simulated.drift_detected}")
print(f"scale-dependent features flagged:   {sorted(expected_changed & set(simulated.drifted_features))}")
print(f"scale-invariant features flagged:   {sorted(set(simulated.drifted_features) - expected_changed)}")
print(f"scale-dependent features missed:    {sorted(expected_changed - set(simulated.drifted_features))}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Record both reports
# MAGIC
# MAGIC Appended to a Delta table so drift history is queryable, not just
# MAGIC printed. Simulated reports are labelled and never feed the retrain
# MAGIC trigger.

# COMMAND ----------

checked_at = datetime.now(timezone.utc)
reports = pd.DataFrame(
    [
        {
            "checked_at": checked_at,
            "scenario": scenario,
            "model_version": int(champion.version),
            "gold_table_version": int(gold_version),
            "reference_rows": n_reference,
            "current_rows": n_current,
            "n_features": report.n_features,
            "drifted_features": report.drifted_features,
            "drift_detected": report.drift_detected,
            "p_value_threshold": P_VALUE_THRESHOLD,
            "drifted_fraction_alarm": DRIFTED_FEATURE_FRACTION_ALARM,
        }
        for scenario, report in [("observed", observed), (f"simulated_gain_{SIMULATED_GAIN}", simulated)]
    ]
)
spark.createDataFrame(reports).write.format("delta").mode("append").saveAsTable(DRIFT_TABLE)
print(f"appended {len(reports)} reports to {DRIFT_TABLE}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Retrain trigger
# MAGIC
# MAGIC Recommends a retrain only after three consecutive observed drift
# MAGIC reports, and never retrains by itself: retraining redefines "healthy",
# MAGIC which needs a human to confirm the drift is real and not a fault.

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
# Databricks notebook source
# MAGIC %md
# MAGIC # Batch scoring — the champion model scores every bearing
# MAGIC
# MAGIC Loads whichever model version currently holds the `champion` alias and
# MAGIC predicts remaining useful life for every bearing in Gold, including
# MAGIC those that never failed — in a real plant, which bearing will fail is
# MAGIC unknown. Batch rather than real-time serving because snapshots arrive
# MAGIC roughly every ten minutes; there is nothing to gain from millisecond
# MAGIC latency.
# MAGIC
# MAGIC **Contracts enforced here.** Feature columns come from the model's
# MAGIC signature, not from Gold. Normalization settings come from the
# MAGIC champion's own training run. Both mean scoring prepares data exactly as
# MAGIC training did, even if defaults or tables change later.

# COMMAND ----------

import os
import sys

REPO_ROOT = os.path.dirname(os.getcwd())
if REPO_ROOT not in sys.path:
    sys.path.append(REPO_ROOT)

from datetime import UTC, datetime

import mlflow
import pandas as pd
from mlflow.tracking import MlflowClient

from bearing_pdm.feature_matrix import build_sensor_matrix
from bearing_pdm.metrics import DEFAULT_ALERT_HORIZON_HOURS
from bearing_pdm.rul_regressor import predict_clipped

CATALOG = "bearing"
GOLD_TABLE = f"{CATALOG}.gold.features"
PREDICTIONS_TABLE = f"{CATALOG}.gold.rul_predictions"
MODEL_NAME = f"{CATALOG}.ml.rul_regressor"



mlflow.set_tracking_uri("databricks")
mlflow.set_registry_uri("databricks-uc")
client = MlflowClient()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Load the champion and its training contract

# COMMAND ----------

champion = client.get_model_version_by_alias(MODEL_NAME, "champion")
model_uri = f"models:/{MODEL_NAME}@champion"
model = mlflow.sklearn.load_model(model_uri)

feature_columns = mlflow.models.get_model_info(model_uri).signature.inputs.input_names()

training_params = client.get_run(champion.run_id).data.params
healthy_fraction = float(training_params["healthy_fraction"])
rul_cap_hours = float(training_params["rul_cap_hours"])

print(f"champion: {MODEL_NAME} version {champion.version}")
print(f"{len(feature_columns)} input features; healthy_fraction={healthy_fraction}; cap={rul_cap_hours} h")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Score every bearing

# COMMAND ----------

gold_version = spark.sql(f"DESCRIBE HISTORY {GOLD_TABLE} LIMIT 1").first()["version"]
gold = spark.sql(f"SELECT * FROM {GOLD_TABLE} VERSION AS OF {gold_version}").toPandas()

missing = sorted(set(feature_columns) - set(gold.columns))
assert not missing, f"Gold lacks columns the champion was trained on: {missing}"

ordered, matrix = build_sensor_matrix(gold, feature_columns, healthy_fraction=healthy_fraction)
predicted = predict_clipped(model, pd.DataFrame(matrix, columns=feature_columns), rul_cap_hours)

scored_at = datetime.now(UTC)
predictions = ordered[["bearing_id", "test_set", "snapshot_ts", "elapsed_hours", "runs_to_failure"]].copy()
predictions["predicted_rul_hours"] = predicted
predictions["maintenance_alert"] = predicted <= DEFAULT_ALERT_HORIZON_HOURS
predictions["model_name"] = MODEL_NAME
predictions["model_version"] = int(champion.version)
predictions["gold_table_version"] = int(gold_version)
predictions["scored_at"] = scored_at

assert len(predictions) == len(gold), "scoring dropped or duplicated rows"
assert predictions["predicted_rul_hours"].between(0, rul_cap_hours).all(), "prediction outside [0, cap]"

# COMMAND ----------

# MAGIC %md
# MAGIC ## Append to the prediction log
# MAGIC
# MAGIC Appended, never overwritten: every row records which model version
# MAGIC scored which Gold version, and when. That history is the audit trail,
# MAGIC and the input a prediction-drift monitor would read later.

# COMMAND ----------

(
    spark.createDataFrame(predictions)
    .write.format("delta")
    .mode("append")
    .saveAsTable(PREDICTIONS_TABLE)
)
print(f"appended {len(predictions)} predictions to {PREDICTIONS_TABLE}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Fleet status — each bearing at its latest snapshot
# MAGIC
# MAGIC Sanity check, not an evaluation: the three failing bearings were in the
# MAGIC champion's training data, so their predictions here are in-sample. The
# MAGIC honest performance numbers are the leave-one-bearing-out results from
# MAGIC `04_train_evaluate`. What to look for: failing bearings near 0 hours at
# MAGIC their last snapshot, healthy ones well above the alert horizon.

# COMMAND ----------

latest = predictions.loc[predictions.groupby("bearing_id")["snapshot_ts"].idxmax()]
display(
    latest[["bearing_id", "runs_to_failure", "elapsed_hours", "predicted_rul_hours", "maintenance_alert"]]
    .round(1)
    .sort_values("bearing_id")
)
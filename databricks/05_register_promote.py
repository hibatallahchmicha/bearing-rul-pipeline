# Databricks notebook source
# MAGIC %md
# MAGIC # Register & promote — only a model that beats the clock reaches production
# MAGIC
# MAGIC Reads the evaluation runs from `04_train_evaluate`, retrains the best
# MAGIC vibration model on all run-to-failure bearings, registers it in Unity
# MAGIC Catalog with its input signature, and applies a promotion gate.
# MAGIC
# MAGIC **Aliases, not stages.** Unity Catalog's registry replaced the old
# MAGIC Staging/Production stages with aliases — named pointers moved between
# MAGIC versions. Every new version gets `challenger`; it gets `champion` only if
# MAGIC it passes the gate. Scoring always loads `@champion`, so rollback is
# MAGIC moving one pointer back, with no code change.

# COMMAND ----------

import os
import sys

REPO_ROOT = os.path.dirname(os.getcwd())
if REPO_ROOT not in sys.path:
    sys.path.append(REPO_ROOT)

import mlflow
import pandas as pd
from mlflow.exceptions import MlflowException
from mlflow.models import infer_signature
from mlflow.tracking import MlflowClient

from bearing_pdm.feature_matrix import build_sensor_matrix
from bearing_pdm.normalization import DEFAULT_HEALTHY_FRACTION
from bearing_pdm.rul_regressor import build_gradient_boosting_regressor, build_ridge_regressor
from bearing_pdm.rul_target import DEFAULT_RUL_CAP_HOURS

CATALOG = "bearing"
GOLD_TABLE = f"{CATALOG}.gold.features"
MODEL_SCHEMA = f"{CATALOG}.ml"
MODEL_NAME = f"{MODEL_SCHEMA}.rul_regressor"

LABEL_COLUMNS = [
    "bearing_id", "test_set", "channel", "snapshot_ts",
    "elapsed_hours", "rul_hours", "rul_hours_capped", "runs_to_failure",
]
SCORE = "metrics.pooled_asymmetric_score_per_prediction"
CLOCK_BASELINE = "baseline_linear_degradation"

current_user = spark.sql("SELECT current_user()").first()[0]
EVAL_EXPERIMENT = f"/Users/{current_user}/bearing-rul-lobo"
REGISTRY_EXPERIMENT = f"/Users/{current_user}/bearing-rul-registry"

mlflow.set_tracking_uri("databricks")
mlflow.set_registry_uri("databricks-uc")
spark.sql(f"CREATE SCHEMA IF NOT EXISTS {MODEL_SCHEMA}")
client = MlflowClient()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Evaluation results for the current Gold version
# MAGIC
# MAGIC Only runs made on the Gold version that exists now count. If Gold was
# MAGIC rebuilt since `04` ran, the old scores describe different data, and the
# MAGIC notebook stops rather than promote a model judged on something else.

# COMMAND ----------

gold_version = spark.sql(f"DESCRIBE HISTORY {GOLD_TABLE} LIMIT 1").first()["version"]

runs = mlflow.search_runs(
    experiment_names=[EVAL_EXPERIMENT],
    filter_string=f"params.gold_table_version = '{gold_version}'",
    order_by=["start_time DESC"],
)
if "tags.mlflow.parentRunId" in runs.columns:
    runs = runs[runs["tags.mlflow.parentRunId"].isna()]

latest = runs.drop_duplicates("params.predictor")
assert not latest.empty, (
    f"no evaluation runs for Gold version {gold_version} -- run 04_train_evaluate first"
)
display(latest[["params.predictor", "params.inputs", SCORE, "metrics.pooled_mae_hours"]])

clock_score = latest.loc[latest["params.predictor"] == CLOCK_BASELINE, SCORE].item()
winner = latest[latest["params.inputs"] == "sensor"].sort_values(SCORE).iloc[0]
winner_name = winner["params.predictor"]
winner_score = float(winner[SCORE])
print(f"best vibration model: {winner_name} (score {winner_score:.1f}); clock baseline: {clock_score:.1f}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Retrain on all run-to-failure bearings
# MAGIC
# MAGIC Evaluation trained on two bearings at a time to measure generalization.
# MAGIC The model that ships is trained on all three, with the same
# MAGIC hyperparameters and the same `build_sensor_matrix` preparation — the
# MAGIC shared function is what keeps training and scoring inputs identical.

# COMMAND ----------

gold = spark.sql(
    f"SELECT * FROM {GOLD_TABLE} VERSION AS OF {gold_version} WHERE runs_to_failure"
).toPandas()
FEATURE_COLUMNS = [c for c in gold.columns if c not in LABEL_COLUMNS]

ordered, matrix = build_sensor_matrix(gold, FEATURE_COLUMNS, healthy_fraction=DEFAULT_HEALTHY_FRACTION)
X = pd.DataFrame(matrix, columns=FEATURE_COLUMNS)
y = ordered["rul_hours_capped"].to_numpy(dtype=float)
training_bearings = sorted(ordered["bearing_id"].unique())

if winner_name == "gradient_boosting":
    hyperparams = {
        "n_estimators": int(winner["params.n_estimators"]),
        "max_depth": int(winner["params.max_depth"]),
    }
    model = build_gradient_boosting_regressor(**hyperparams)
elif winner_name == "ridge":
    hyperparams = {"alpha": float(winner["params.alpha"])}
    model = build_ridge_regressor(**hyperparams)
else:
    raise ValueError(f"no builder for predictor {winner_name!r}")

model.fit(X, y)
print(f"trained {winner_name} on {len(X)} rows from {training_bearings}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Register with an explicit signature
# MAGIC
# MAGIC The signature records the 17 input columns and their types. Unity
# MAGIC Catalog requires one, and it is what makes scoring with wrong-shaped data
# MAGIC fail immediately instead of returning plausible nonsense.
# MAGIC
# MAGIC The registered model returns raw predictions; scoring clips them to
# MAGIC [0, cap] with the same `predict_clipped` used in evaluation.

# COMMAND ----------

mlflow.set_experiment(REGISTRY_EXPERIMENT)

with mlflow.start_run(run_name=f"register_{winner_name}") as run:
    mlflow.log_params(
        {
            "predictor": winner_name,
            "gold_table": GOLD_TABLE,
            "gold_table_version": gold_version,
            "training_bearings": ",".join(training_bearings),
            "healthy_fraction": DEFAULT_HEALTHY_FRACTION,
            "rul_cap_hours": DEFAULT_RUL_CAP_HOURS,
            "evaluation_run_id": winner["run_id"],
            **hyperparams,
        }
    )
    mlflow.log_dict({"feature_columns": FEATURE_COLUMNS}, "feature_columns.json")

    signature = infer_signature(X, model.predict(X))
    model_info = mlflow.sklearn.log_model(
        model,
        artifact_path="model",
        signature=signature,
        input_example=X.head(5),
        registered_model_name=MODEL_NAME,
    )

# log_model reports the version it just registered -- no need to search the
# registry for it (searching Unity Catalog by run_id is not reliably supported).
new_version = model_info.registered_model_version
print(f"registered {MODEL_NAME} version {new_version}")

for key, value in {
    "eval_asym_score_per_pred": winner_score,
    "clock_baseline_score": clock_score,
    "gold_table_version": gold_version,
    "evaluation_run_id": winner["run_id"],
}.items():
    client.set_model_version_tag(MODEL_NAME, new_version, key, str(value))

client.set_registered_model_alias(MODEL_NAME, "challenger", new_version)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Promotion gate
# MAGIC
# MAGIC `champion` moves to the new version only if it beats the clock baseline
# MAGIC AND beats the current champion's recorded evaluation score. A model that
# MAGIC cannot beat elapsed operating time has not learned anything the
# MAGIC calendar does not already know, and does not reach production.

# COMMAND ----------

try:
    current = client.get_model_version_by_alias(MODEL_NAME, "champion")
    current_score = float(current.tags["eval_asym_score_per_pred"])
except MlflowException:
    current, current_score = None, float("inf")

beats_clock = winner_score < clock_score
beats_champion = winner_score < current_score

if beats_clock and beats_champion:
    client.set_registered_model_alias(MODEL_NAME, "champion", new_version)
    previous = f"version {current.version}" if current else "none"
    print(f"PROMOTED: version {new_version} is now champion (previous champion: {previous})")
else:
    reasons = []
    if not beats_clock:
        reasons.append(f"does not beat the clock baseline ({winner_score:.1f} vs {clock_score:.1f})")
    if not beats_champion:
        reasons.append(f"does not beat champion v{current.version} ({winner_score:.1f} vs {current_score:.1f})")
    print(f"NOT PROMOTED: version {new_version} stays challenger -- " + "; ".join(reasons))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Rollback, if a champion misbehaves in production
# MAGIC
# MAGIC Scoring loads `models:/bearing.ml.rul_regressor@champion`, never a
# MAGIC version number. Rolling back is one call, with no redeploy:
# MAGIC
# MAGIC ```python
# MAGIC client.set_registered_model_alias(MODEL_NAME, "champion", "<previous version>")
# MAGIC ```

# COMMAND ----------

versions = client.search_model_versions(f"name = '{MODEL_NAME}'")
display(
    pd.DataFrame(
        [
            {
                "version": v.version,
                "aliases": ", ".join(v.aliases),
                "eval_score": v.tags.get("eval_asym_score_per_pred"),
                "gold_version": v.tags.get("gold_table_version"),
            }
            for v in versions
        ]
    ).sort_values("version")
)
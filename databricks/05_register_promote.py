# Databricks notebook source
# MAGIC %md
# MAGIC # Register & promote — the model whose alerts cost least
# MAGIC
# MAGIC Reads the evaluation runs from `04_train_evaluate`, retrains the best
# MAGIC vibration model on all run-to-failure bearings, registers it in Unity
# MAGIC Catalog with its input signature, and applies a promotion gate.
# MAGIC
# MAGIC **Chosen by alarm cost.** What the champion does in production is raise
# MAGIC `maintenance_alert`. So it is picked by the cost of that decision —
# MAGIC false alarms cost 1, missed alarms 10, the ratio in `cost_threshold`.
# MAGIC RUL accuracy stays as a floor: it must still beat the clock.
# MAGIC
# MAGIC **Aliases, not stages.** Every new version gets `challenger`; it gets
# MAGIC `champion` only if it passes the gate. Scoring loads `@champion`, so
# MAGIC rollback is moving one pointer, with no code change.

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
from bearing_pdm.metrics import DEFAULT_ALERT_HORIZON_HOURS
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
ALARM = "metrics.alarm_cost"
FALSE_ALARMS = "metrics.alarm_false_alarms"
MISSED_ALARMS = "metrics.alarm_missed_alarms"
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
# MAGIC Only runs on the Gold version that exists now count, and only the most
# MAGIC recent run per predictor.

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
assert not latest.empty, f"no evaluation runs for Gold version {gold_version} -- run 04_train_evaluate first"
assert ALARM in latest.columns and latest[ALARM].notna().all(), (
    "evaluation runs lack alarm cost -- re-run 04_train_evaluate"
)
display(latest[["params.predictor", "params.inputs", ALARM, FALSE_ALARMS, MISSED_ALARMS, SCORE]])

clock = latest[latest["params.predictor"] == CLOCK_BASELINE].iloc[0]
winner = latest[latest["params.inputs"] == "sensor"].sort_values([ALARM, SCORE]).iloc[0]
winner_name = winner["params.predictor"]
print(f"lowest alarm cost among vibration models: {winner_name} ({winner[ALARM]:.0f})")
print(f"clock baseline alarm cost: {clock[ALARM]:.0f}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Retrain on all run-to-failure bearings
# MAGIC
# MAGIC Same hyperparameters, same `build_sensor_matrix` preparation as
# MAGIC evaluation. The shipped model is trained on all three failing bearings.

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
            "alert_horizon_hours": DEFAULT_ALERT_HORIZON_HOURS,
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
    "alarm_cost": float(winner[ALARM]),
    "alarm_false_alarms": int(winner[FALSE_ALARMS]),
    "alarm_missed_alarms": int(winner[MISSED_ALARMS]),
    "eval_asym_score_per_pred": float(winner[SCORE]),
    "clock_alarm_cost": float(clock[ALARM]),
    "gold_table_version": gold_version,
    "evaluation_run_id": winner["run_id"],
}.items():
    client.set_model_version_tag(MODEL_NAME, new_version, key, str(value))

client.set_registered_model_alias(MODEL_NAME, "challenger", new_version)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Promotion gate
# MAGIC
# MAGIC `champion` moves to the new version only if every check passes:
# MAGIC
# MAGIC 1. It beats the clock on RUL accuracy — otherwise it has learned nothing
# MAGIC    the calendar does not already know.
# MAGIC 2. Its alerts cost less than the clock's.
# MAGIC 3. Its alerts cost less than the current champion's — **if** that
# MAGIC    champion was judged by alarm cost on the same Gold version. Costs on
# MAGIC    different data are not comparable, and a champion promoted by an older
# MAGIC    gate that never measured alarm cost does not block one that was.
# MAGIC
# MAGIC An earlier version of this gate capped the false-alarm rate at 5%. It
# MAGIC was removed: it judged one kind of mistake in isolation, and a model
# MAGIC that never alerts passes it perfectly.

# COMMAND ----------

try:
    alias_version = client.get_model_version_by_alias(MODEL_NAME, "champion")
    current = client.get_model_version(MODEL_NAME, alias_version.version)
except MlflowException:
    current = None

winner_cost = float(winner[ALARM])
checks = {
    f"beats the clock on RUL accuracy ({winner[SCORE]:.1f} < {clock[SCORE]:.1f})": winner[SCORE] < clock[SCORE],
    f"alerts cost less than the clock's ({winner_cost:.0f} < {clock[ALARM]:.0f})": winner_cost < clock[ALARM],
}

if current is None:
    print("note: no current champion")
elif "alarm_cost" not in current.tags:
    print(f"note: champion v{current.version} predates alarm-cost evaluation, so it does not block promotion")
elif current.tags.get("gold_table_version") != str(gold_version):
    print(f"note: champion v{current.version} was judged on different Gold data, so costs are not comparable")
else:
    current_cost = float(current.tags["alarm_cost"])
    checks[f"alerts cost less than champion v{current.version}'s ({winner_cost:.0f} < {current_cost:.0f})"] = (
        winner_cost < current_cost
    )

for description, passed in checks.items():
    print(f"{'PASS' if passed else 'FAIL'}  {description}")

if all(checks.values()):
    client.set_registered_model_alias(MODEL_NAME, "champion", new_version)
    previous = f"v{current.version}" if current else "none"
    print(f"\nPROMOTED: version {new_version} is now champion (previous: {previous})")
else:
    print(f"\nNOT PROMOTED: version {new_version} stays challenger")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Rollback, if a champion misbehaves in production
# MAGIC
# MAGIC ```python
# MAGIC client.set_registered_model_alias(MODEL_NAME, "champion", "<previous version>")
# MAGIC ```

# COMMAND ----------

# search_model_versions returns summaries; tags and aliases are only
# reliably populated when each version is fetched individually.
versions = [
    client.get_model_version(MODEL_NAME, v.version)
    for v in client.search_model_versions(f"name = '{MODEL_NAME}'")
]
display(
    pd.DataFrame(
        [
            {
                "version": int(v.version),
                "aliases": ", ".join(v.aliases or []),
                "alarm_cost": v.tags.get("alarm_cost"),
                "false_alarms": v.tags.get("alarm_false_alarms"),
                "missed_alarms": v.tags.get("alarm_missed_alarms"),
                "eval_score": v.tags.get("eval_asym_score_per_pred"),
                "gold_version": v.tags.get("gold_table_version"),
            }
            for v in versions
        ]
    ).sort_values("version")
)
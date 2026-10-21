# Databricks notebook source
# MAGIC %md
# MAGIC # Train & evaluate — does vibration beat a clock, without crying wolf?
# MAGIC
# MAGIC Four predictors, one leave-one-bearing-out split, one metric suite, all
# MAGIC logged to MLflow. The comparison that matters is the vibration-feature
# MAGIC models against `baseline_linear_degradation`, which uses elapsed
# MAGIC operating hours and nothing else.
# MAGIC
# MAGIC **False alarms are measured too.** Accuracy on failing bearings says
# MAGIC nothing about alerts on healthy ones. Two out-of-sample rates are logged:
# MAGIC alerts on held-out failing bearings while they provably had more than the
# MAGIC alert horizon left, and alerts on the bearings that never failed — which
# MAGIC are never used for training, so every model sees them for the first time.
# MAGIC
# MAGIC **Data versioning.** Gold is read at a pinned Delta version, logged on
# MAGIC every run. Time travel only reaches back as far as file retention (7
# MAGIC days by default), so the MLflow run is the long-term record.

# COMMAND ----------

import os
import sys

REPO_ROOT = os.path.dirname(os.getcwd())
if REPO_ROOT not in sys.path:
    sys.path.append(REPO_ROOT)

import mlflow
import numpy as np
import pandas as pd

from bearing_pdm.baselines import LinearDegradationBaseline, MeanRULBaseline
from bearing_pdm.evaluation import split_by_unit
from bearing_pdm.feature_matrix import build_sensor_matrix
from bearing_pdm.metrics import DEFAULT_ALERT_HORIZON_HOURS, provable_false_alarm_rate, rul_metrics
from bearing_pdm.normalization import DEFAULT_HEALTHY_FRACTION
from bearing_pdm.rul_regressor import (
    DEFAULT_MAX_DEPTH,
    DEFAULT_N_ESTIMATORS,
    DEFAULT_RIDGE_ALPHA,
    build_gradient_boosting_regressor,
    build_ridge_regressor,
    predict_clipped,
)
from bearing_pdm.rul_target import DEFAULT_RUL_CAP_HOURS
from bearing_pdm.tracking import configure_tracking, fold_run, log_rul_metrics, parent_run

CATALOG = "bearing"
GOLD_TABLE = f"{CATALOG}.gold.features"

LABEL_COLUMNS = [
    "bearing_id", "test_set", "channel", "snapshot_ts",
    "elapsed_hours", "rul_hours", "rul_hours_capped", "runs_to_failure",
]

current_user = spark.sql("SELECT current_user()").first()[0]
EXPERIMENT = f"/Users/{current_user}/bearing-rul-lobo"
configure_tracking(tracking_uri="databricks", experiment=EXPERIMENT)
print(f"logging to experiment: {EXPERIMENT}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Load every bearing from Gold, at a pinned version
# MAGIC
# MAGIC Failing bearings are used for training and evaluation; healthy ones
# MAGIC only for measuring false alarms. `hours_to_end` is the time from each
# MAGIC snapshot to the end of that bearing's recording: exact remaining life
# MAGIC for a failing bearing, a lower bound for a healthy one.

# COMMAND ----------

gold_version = spark.sql(f"DESCRIBE HISTORY {GOLD_TABLE} LIMIT 1").first()["version"]
gold = spark.sql(f"SELECT * FROM {GOLD_TABLE} VERSION AS OF {gold_version}").toPandas()
FEATURE_COLUMNS = [c for c in gold.columns if c not in LABEL_COLUMNS]

ordered, X_sensor_all = build_sensor_matrix(gold, FEATURE_COLUMNS, healthy_fraction=DEFAULT_HEALTHY_FRACTION)
ordered["hours_to_end"] = (
    ordered.groupby("bearing_id")["elapsed_hours"].transform("max") - ordered["elapsed_hours"]
)

X_clock_all = ordered[["elapsed_hours"]].to_numpy(dtype=float)
y_all = ordered["rul_hours_capped"].to_numpy(dtype=float)
units_all = ordered["bearing_id"].to_numpy()
hours_to_end_all = ordered["hours_to_end"].to_numpy(dtype=float)
failing = ordered["runs_to_failure"].to_numpy(dtype=bool)
healthy = ~failing

assert np.all(np.isfinite(X_sensor_all)), "normalized features contain NaN or inf"
print(f"Gold version {gold_version}: {failing.sum()} failing-bearing rows, {healthy.sum()} healthy-bearing rows")
print(ordered.groupby(["runs_to_failure", "bearing_id"]).size())

# COMMAND ----------

# MAGIC %md
# MAGIC ## Evaluation
# MAGIC
# MAGIC Per predictor: leave-one-bearing-out over the failing bearings (accuracy
# MAGIC plus false alarms on each held-out bearing), then one model trained on all
# MAGIC failing bearings scores the healthy ones. Every prediction is clipped to
# MAGIC [0, cap] by the same rule, so no predictor gets an unfair advantage.

# COMMAND ----------

PREDICTORS = [
    ("baseline_mean", MeanRULBaseline, "clock", {}),
    ("baseline_linear_degradation", lambda: LinearDegradationBaseline(elapsed_col=0), "clock", {}),
    ("ridge", lambda: build_ridge_regressor(DEFAULT_RIDGE_ALPHA), "sensor", {"alpha": DEFAULT_RIDGE_ALPHA}),
    (
        "gradient_boosting",
        lambda: build_gradient_boosting_regressor(DEFAULT_N_ESTIMATORS, DEFAULT_MAX_DEPTH),
        "sensor",
        {"n_estimators": DEFAULT_N_ESTIMATORS, "max_depth": DEFAULT_MAX_DEPTH},
    ),
]

summary = []

for name, factory, inputs, hyperparams in PREDICTORS:
    X_all = X_sensor_all if inputs == "sensor" else X_clock_all
    X_f, y_f, units_f, hte_f = X_all[failing], y_all[failing], units_all[failing], hours_to_end_all[failing]
    fold_true, fold_pred, fold_hte = [], [], []

    params = {
        "predictor": name,
        "inputs": inputs,
        "gold_table": GOLD_TABLE,
        "gold_table_version": gold_version,
        "rul_cap_hours": DEFAULT_RUL_CAP_HOURS,
        "healthy_fraction": DEFAULT_HEALTHY_FRACTION,
        "alert_horizon_hours": DEFAULT_ALERT_HORIZON_HOURS,
        "n_input_columns": X_all.shape[1],
        **hyperparams,
    }

    with parent_run(run_name=name, params=params):
        for train_mask, test_mask in split_by_unit(units_f):
            test_unit = units_f[test_mask][0]
            model = factory().fit(X_f[train_mask], y_f[train_mask])
            pred = predict_clipped(model, X_f[test_mask], DEFAULT_RUL_CAP_HOURS)

            with fold_run(test_unit):
                log_rul_metrics(rul_metrics(y_f[test_mask], pred))
                mlflow.log_metric("provable_false_alarm_rate", provable_false_alarm_rate(pred, hte_f[test_mask]))

            fold_true.append(y_f[test_mask])
            fold_pred.append(pred)
            fold_hte.append(hte_f[test_mask])

        pooled = rul_metrics(np.concatenate(fold_true), np.concatenate(fold_pred))
        log_rul_metrics(pooled, prefix="pooled_")
        pooled_far = provable_false_alarm_rate(np.concatenate(fold_pred), np.concatenate(fold_hte))

        final_model = factory().fit(X_f, y_f)
        healthy_pred = predict_clipped(final_model, X_all[healthy], DEFAULT_RUL_CAP_HOURS)
        healthy_far = provable_false_alarm_rate(healthy_pred, hours_to_end_all[healthy])

        mlflow.log_metrics(
            {
                "pooled_provable_false_alarm_rate": pooled_far,
                "healthy_provable_false_alarm_rate": healthy_far,
            }
        )

    summary.append(
        {
            "predictor": name,
            "inputs": inputs,
            "asym_score_per_pred": pooled.asymmetric_score / pooled.n,
            "mae_h": pooled.mae,
            "late_fraction": pooled.late_fraction,
            "false_alarms_failing": pooled_far,
            "false_alarms_healthy": healthy_far,
        }
    )
    print(f"{name}: done")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Results
# MAGIC
# MAGIC Ranked by asymmetric score per prediction — lower is better. The two
# MAGIC false-alarm columns are shares of snapshots where the model alerted
# MAGIC while the bearing provably had more than the alert horizon left.

# COMMAND ----------

results = pd.DataFrame(summary).sort_values("asym_score_per_pred").reset_index(drop=True)
display(results.round(3))
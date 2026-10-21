# Databricks notebook source
# MAGIC %md
# MAGIC # Train & evaluate — accuracy, and the cost of the alerts it would raise
# MAGIC
# MAGIC Four predictors, one leave-one-bearing-out split, all logged to MLflow.
# MAGIC Two questions per predictor:
# MAGIC
# MAGIC 1. **Does it beat a clock?** RUL accuracy against
# MAGIC    `baseline_linear_degradation`, which uses elapsed hours only.
# MAGIC 2. **What do its alerts cost?** An alert fires when predicted RUL is at or
# MAGIC    under the horizon. False alarms cost 1, missed alarms cost 10 — the
# MAGIC    ratio from `cost_threshold`. Judged on held-out failing bearings
# MAGIC    (both mistakes) and on healthy bearings, which never train anything
# MAGIC    (false alarms only: they were never seen failing, so silence on them
# MAGIC    cannot be proven wrong).
# MAGIC
# MAGIC **Data versioning.** Gold is read at a pinned Delta version, logged on
# MAGIC every run. Time travel only reaches back as far as file retention, so
# MAGIC the MLflow run is the long-term record.

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
from bearing_pdm.metrics import (
    DEFAULT_ALERT_HORIZON_HOURS,
    alarm_cost,
    provable_false_alarm_rate,
    rul_metrics,
)
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
# MAGIC `hours_to_end` is the time from each snapshot to the end of that
# MAGIC bearing's recording: exact remaining life for a failing bearing, a lower
# MAGIC bound for a healthy one (right-censored).

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

# COMMAND ----------

# MAGIC %md
# MAGIC ## Evaluation
# MAGIC
# MAGIC Per predictor: leave-one-bearing-out over the failing bearings, then one
# MAGIC model trained on all failing bearings scores the healthy ones. Both sets
# MAGIC of predictions are out-of-sample, and together they give the alarm cost.

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
            fold_alarms = alarm_cost(pred, hte_f[test_mask], censored=np.zeros(len(pred), dtype=bool))

            with fold_run(test_unit):
                log_rul_metrics(rul_metrics(y_f[test_mask], pred))
                mlflow.log_metrics(
                    {
                        "provable_false_alarm_rate": provable_false_alarm_rate(pred, hte_f[test_mask]),
                        "alarm_cost": fold_alarms.cost,
                        "false_alarms": fold_alarms.false_alarms,
                        "missed_alarms": fold_alarms.missed_alarms,
                    }
                )

            fold_true.append(y_f[test_mask])
            fold_pred.append(pred)
            fold_hte.append(hte_f[test_mask])

        pooled = rul_metrics(np.concatenate(fold_true), np.concatenate(fold_pred))
        log_rul_metrics(pooled, prefix="pooled_")

        final_model = factory().fit(X_f, y_f)
        healthy_pred = predict_clipped(final_model, X_all[healthy], DEFAULT_RUL_CAP_HOURS)

        failing_pred = np.concatenate(fold_pred)
        failing_hte = np.concatenate(fold_hte)
        alarms = alarm_cost(
            np.concatenate([failing_pred, healthy_pred]),
            np.concatenate([failing_hte, hours_to_end_all[healthy]]),
            censored=np.concatenate([np.zeros(len(failing_pred), dtype=bool), np.ones(len(healthy_pred), dtype=bool)]),
        )
        far_failing = provable_false_alarm_rate(failing_pred, failing_hte)
        far_healthy = provable_false_alarm_rate(healthy_pred, hours_to_end_all[healthy])

        mlflow.log_metrics(
            {
                "alarm_cost": alarms.cost,
                "alarm_false_alarms": alarms.false_alarms,
                "alarm_missed_alarms": alarms.missed_alarms,
                "pooled_provable_false_alarm_rate": far_failing,
                "healthy_provable_false_alarm_rate": far_healthy,
            }
        )

    summary.append(
        {
            "predictor": name,
            "inputs": inputs,
            "alarm_cost": alarms.cost,
            "false_alarms": alarms.false_alarms,
            "missed_alarms": alarms.missed_alarms,
            "asym_score_per_pred": pooled.asymmetric_score / pooled.n,
            "mae_h": pooled.mae,
            "false_alarm_rate_failing": far_failing,
            "false_alarm_rate_healthy": far_healthy,
        }
    )
    print(f"{name}: done")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Results
# MAGIC
# MAGIC Sorted by alarm cost — the cost of the decision the model actually makes
# MAGIC in production. `baseline_mean` predicts one constant, so it almost never
# MAGIC alerts: its alarm cost is the price of doing nothing.

# COMMAND ----------

results = pd.DataFrame(summary).sort_values("alarm_cost").reset_index(drop=True)
display(results.round(3))
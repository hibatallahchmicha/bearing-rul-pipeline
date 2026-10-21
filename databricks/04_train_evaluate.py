# Databricks notebook source
# MAGIC %md
# MAGIC # Train & evaluate — does vibration beat a clock?
# MAGIC
# MAGIC Four predictors, one leave-one-bearing-out split, one metric suite, all
# MAGIC logged to MLflow. The comparison that matters is the vibration-feature
# MAGIC models against `baseline_linear_degradation`, which uses elapsed
# MAGIC operating hours and nothing else. The margin between them is the
# MAGIC measured value of the entire signal-processing stage.
# MAGIC
# MAGIC **Data versioning.** Gold is read at a pinned Delta version and that
# MAGIC version is logged on every run. Delta time travel only reaches back as
# MAGIC far as the file-retention window (7 days by default), so it cannot be
# MAGIC the long-term record of what a model was trained on — the MLflow run is.

# COMMAND ----------

import os
import sys

REPO_ROOT = os.path.dirname(os.getcwd())
if REPO_ROOT not in sys.path:
    sys.path.append(REPO_ROOT)

import numpy as np
import pandas as pd

from bearing_pdm.baselines import LinearDegradationBaseline, MeanRULBaseline
from bearing_pdm.evaluation import split_by_unit
from bearing_pdm.metrics import rul_metrics
from bearing_pdm.normalization import DEFAULT_HEALTHY_FRACTION, normalize_to_own_baseline
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
    "bearing_id",
    "test_set",
    "channel",
    "snapshot_ts",
    "elapsed_hours",
    "rul_hours",
    "rul_hours_capped",
    "runs_to_failure",
]

current_user = spark.sql("SELECT current_user()").first()[0]
EXPERIMENT = f"/Users/{current_user}/bearing-rul-lobo"
configure_tracking(tracking_uri="databricks", experiment=EXPERIMENT)
print(f"logging to experiment: {EXPERIMENT}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Load Gold at a pinned version
# MAGIC
# MAGIC The version is captured first and the data is then read *at that
# MAGIC version*, so the number logged to MLflow is guaranteed to be the data
# MAGIC actually used — even if Gold is rewritten while this notebook runs.

# COMMAND ----------

gold_version = spark.sql(f"DESCRIBE HISTORY {GOLD_TABLE} LIMIT 1").first()["version"]

gold = (
    spark.sql(f"SELECT * FROM {GOLD_TABLE} VERSION AS OF {gold_version} WHERE runs_to_failure")
    .toPandas()
    .sort_values(["bearing_id", "snapshot_ts"])
    .reset_index(drop=True)
)

FEATURE_COLUMNS = [c for c in gold.columns if c not in LABEL_COLUMNS]
print(f"Gold version {gold_version}: {len(gold)} rows, {len(FEATURE_COLUMNS)} features")
print(gold.groupby("bearing_id").size())

# COMMAND ----------

# MAGIC %md
# MAGIC ## Inputs
# MAGIC
# MAGIC - `X_sensor` — the 17 vibration features, each bearing normalized
# MAGIC   against its own first 20% of life. Rows are already sorted by bearing
# MAGIC   and time, which normalization requires.
# MAGIC - `X_clock` — elapsed operating hours only, for the clock baseline.
# MAGIC
# MAGIC The models see vibration only. Giving them elapsed time as well would
# MAGIC blur exactly the comparison this notebook exists to make.

# COMMAND ----------

X_sensor = np.vstack(
    [
        normalize_to_own_baseline(group[FEATURE_COLUMNS].to_numpy(dtype=float), DEFAULT_HEALTHY_FRACTION)
        for _, group in gold.groupby("bearing_id", sort=True)
    ]
)
X_clock = gold[["elapsed_hours"]].to_numpy(dtype=float)
y = gold["rul_hours_capped"].to_numpy(dtype=float)
units = gold["bearing_id"].to_numpy()

assert X_sensor.shape == (len(gold), len(FEATURE_COLUMNS))
assert np.all(np.isfinite(X_sensor)), "normalized features contain NaN or inf"

# COMMAND ----------

# MAGIC %md
# MAGIC ## Leave-one-bearing-out evaluation
# MAGIC
# MAGIC One parent run per predictor, one child run per held-out bearing.
# MAGIC Every prediction is clipped to the target's valid range [0, cap], the
# MAGIC same rule for all four, so none gets an unfair advantage.

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
    X = X_sensor if inputs == "sensor" else X_clock
    fold_true, fold_pred = [], []

    params = {
        "predictor": name,
        "inputs": inputs,
        "gold_table": GOLD_TABLE,
        "gold_table_version": gold_version,
        "rul_cap_hours": DEFAULT_RUL_CAP_HOURS,
        "healthy_fraction": DEFAULT_HEALTHY_FRACTION,
        "n_input_columns": X.shape[1],
        **hyperparams,
    }

    with parent_run(run_name=name, params=params):
        for train_mask, test_mask in split_by_unit(units):
            test_unit = units[test_mask][0]
            model = factory().fit(X[train_mask], y[train_mask])
            pred = predict_clipped(model, X[test_mask], DEFAULT_RUL_CAP_HOURS)

            with fold_run(test_unit):
                log_rul_metrics(rul_metrics(y[test_mask], pred))

            fold_true.append(y[test_mask])
            fold_pred.append(pred)

        pooled = rul_metrics(np.concatenate(fold_true), np.concatenate(fold_pred))
        log_rul_metrics(pooled, prefix="pooled_")

    summary.append(
        {
            "predictor": name,
            "inputs": inputs,
            "asym_score_per_pred": pooled.asymmetric_score / pooled.n,
            "late_fraction": pooled.late_fraction,
            "mean_late_error_h": pooled.mean_late_error,
            "mae_h": pooled.mae,
            "rmse_h": pooled.rmse,
            "n": pooled.n,
        }
    )
    print(f"{name}: done")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Results
# MAGIC
# MAGIC Ranked by asymmetric score per prediction — lower is better. Late
# MAGIC predictions (the bearing fails before the model said it would) are
# MAGIC penalized harder than early ones, because that is the failure that
# MAGIC costs an unplanned breakdown.

# COMMAND ----------

results = pd.DataFrame(summary).sort_values("asym_score_per_pred").reset_index(drop=True)
display(results.round(3))

clock_score = results.loc[results["predictor"] == "baseline_linear_degradation", "asym_score_per_pred"].item()
best = results.iloc[0]
print(f"best predictor: {best['predictor']}")
print(f"beats the clock baseline: {best['asym_score_per_pred'] < clock_score and best['inputs'] == 'sensor'}")
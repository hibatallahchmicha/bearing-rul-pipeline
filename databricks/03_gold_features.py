# Databricks notebook source
# MAGIC %md
# MAGIC # Gold — model-ready features, one row per bearing per snapshot
# MAGIC
# MAGIC Runs the same tested `bearing_pdm` signal-processing code the local
# MAGIC pipeline and CI use — windowing, time-domain statistics, FFT and
# MAGIC envelope-spectrum fault-band energies — distributed with `mapInPandas`
# MAGIC rather than reimplemented in Spark. This is why `bearing_pdm` was kept
# MAGIC pure numpy/scipy: the local test suite validates the exact math that
# MAGIC runs here.
# MAGIC
# MAGIC **Grain.** One row per (bearing, snapshot): the same 12,560 rows as
# MAGIC Silver. Each snapshot's 19 windows are computed in memory and collapsed
# MAGIC with `rul_target.aggregate_windows_to_snapshot`, so the aggregation rule
# MAGIC is the tested one rather than a copy of it. Windows are never persisted.

# COMMAND ----------

import os
import sys

# A notebook in a Git folder runs with its working directory set to its own
# folder (<repo>/databricks), so the repo root is one level up. Derived
# rather than hard-coded: a fixed workspace path breaks as soon as the repo
# is cloned under a different user or folder name.
REPO_ROOT = os.path.dirname(os.getcwd())
if REPO_ROOT not in sys.path:
    sys.path.append(REPO_ROOT)
print(f"repo root: {REPO_ROOT}")

import numpy as np
from pyspark.sql import Window
from pyspark.sql import functions as F
from pyspark.sql.types import (
    BooleanType,
    DoubleType,
    IntegerType,
    StringType,
    StructField,
    StructType,
    TimestampType,
)

from bearing_pdm.fault_frequencies import compute_fault_frequencies
from bearing_pdm.features_freq import frequency_domain_features
from bearing_pdm.features_time import time_domain_features
from bearing_pdm.ingestion import SAMPLE_RATE_HZ, SAMPLES_PER_FILE
from bearing_pdm.segmentation import window_signal

CATALOG = "bearing"
SILVER_TABLE = f"{CATALOG}.silver.snapshots"
GOLD_TABLE = f"{CATALOG}.gold.features"

# Constant shaft speed of the IMS test rig, per NASA's dataset documentation.
SHAFT_SPEED_RPM = 2000

spark.sql(f"CREATE SCHEMA IF NOT EXISTS {CATALOG}.gold")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Schema, derived from the code
# MAGIC
# MAGIC The feature functions are run once on random noise to discover the
# MAGIC column names they actually return. A hand-typed list would compile fine
# MAGIC and then silently disagree with the module the day a feature is added or
# MAGIC renamed. The count is still asserted, because "17 features" is a
# MAGIC documented claim in the README.

# COMMAND ----------

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

_probe = window_signal(np.random.default_rng(0).standard_normal(SAMPLES_PER_FILE).astype(np.float32))
_faults = compute_fault_frequencies(shaft_speed_rpm=SHAFT_SPEED_RPM)
FEATURE_COLUMNS = list(
    {
        **time_domain_features(_probe),
        **frequency_domain_features(_probe, fs=SAMPLE_RATE_HZ, faults=_faults),
    }
)
assert len(FEATURE_COLUMNS) == 17, f"expected 17 features, the code returns {len(FEATURE_COLUMNS)}"
print(FEATURE_COLUMNS)

GOLD_SCHEMA = StructType(
    [
        StructField("bearing_id", StringType(), nullable=False),
        StructField("test_set", StringType(), nullable=False),
        StructField("channel", IntegerType(), nullable=False),
        StructField("snapshot_ts", TimestampType(), nullable=False),
        StructField("elapsed_hours", DoubleType(), nullable=False),
        StructField("rul_hours", DoubleType(), nullable=True),
        StructField("rul_hours_capped", DoubleType(), nullable=True),
        StructField("runs_to_failure", BooleanType(), nullable=False),
    ]
    + [StructField(name, DoubleType(), nullable=False) for name in FEATURE_COLUMNS]
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Feature extraction
# MAGIC
# MAGIC `mapInPandas` sends this function to worker processes, which do not
# MAGIC inherit the notebook's import path. So the repo root is added and
# MAGIC `bearing_pdm` is imported *inside* the function — where the code runs.

# COMMAND ----------


def extract_gold_features(batches):
    import sys as _sys

    if REPO_ROOT not in _sys.path:
        _sys.path.append(REPO_ROOT)

    import numpy as _np
    import pandas as _pd

    from bearing_pdm.fault_frequencies import compute_fault_frequencies
    from bearing_pdm.features_freq import frequency_domain_features
    from bearing_pdm.features_time import time_domain_features
    from bearing_pdm.rul_target import aggregate_windows_to_snapshot
    from bearing_pdm.segmentation import window_signal

    faults = compute_fault_frequencies(shaft_speed_rpm=SHAFT_SPEED_RPM)

    for pdf in batches:
        records = []
        for row in pdf.itertuples(index=False):
            windows = window_signal(_np.asarray(row.waveform, dtype=_np.float32))
            per_window = {
                **time_domain_features(windows),
                **frequency_domain_features(windows, fs=SAMPLE_RATE_HZ, faults=faults),
            }
            per_snapshot = aggregate_windows_to_snapshot(per_window, windows_per_snapshot=len(windows))

            record = {col: getattr(row, col) for col in LABEL_COLUMNS}
            record.update({name: float(per_snapshot[name][0]) for name in FEATURE_COLUMNS})
            records.append(record)

        yield _pd.DataFrame.from_records(records, columns=LABEL_COLUMNS + FEATURE_COLUMNS)


# COMMAND ----------

# Spark's default hands Python 10,000 rows per batch. At ~82 KB of waveform
# per row that is ~800 MB at once -- enough to crash a small serverless
# worker. 256 rows is ~21 MB per batch.
spark.conf.set("spark.sql.execution.arrow.maxRecordsPerBatch", "256")

# Silver is a handful of files after OPTIMIZE, i.e. a handful of tasks.
# Repartitioning spreads the FFT work across 32 tasks instead.
silver = spark.table(SILVER_TABLE).select(*LABEL_COLUMNS, "waveform").repartition(32)

gold = silver.mapInPandas(extract_gold_features, schema=GOLD_SCHEMA)

(
    gold.write.format("delta")
    .mode("overwrite")
    .option("overwriteSchema", "true")
    .saveAsTable(GOLD_TABLE)
)

spark.sql(
    f"COMMENT ON TABLE {GOLD_TABLE} IS "
    "'One row per bearing snapshot: 17 vibration features (median over 19 windows) "
    "plus RUL labels. Built from bearing.silver.snapshots by bearing_pdm.'"
)

print(f"gold rows: {spark.table(GOLD_TABLE).count()}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Boundary assertions
# MAGIC
# MAGIC Gold transforms rows; it never filters them. Its row count must equal
# MAGIC Silver's exactly, and no feature may be null, NaN, or infinite — any of
# MAGIC those would flow straight into model training without an error.

# COMMAND ----------

gold_tbl = spark.table(GOLD_TABLE)

silver_count = spark.table(SILVER_TABLE).count()
gold_count = gold_tbl.count()
assert gold_count == silver_count, f"Gold has {gold_count} rows, Silver has {silver_count}"

bad_value_counts = gold_tbl.select(
    [
        F.sum(
            (F.col(c).isNull() | F.isnan(c) | (F.abs(F.col(c)) == float("inf"))).cast("int")
        ).alias(c)
        for c in FEATURE_COLUMNS
    ]
).first().asDict()
broken = {c: n for c, n in bad_value_counts.items() if n}
assert not broken, f"null/NaN/inf feature values: {broken}"

n_bearings = gold_tbl.select("bearing_id").distinct().count()
assert n_bearings == 8, f"expected 8 bearings, found {n_bearings}"

print("all Gold assertions passed")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Physics sanity check
# MAGIC
# MAGIC For each bearing that ran to failure: early life (first 20%) versus late
# MAGIC life (last 10%). NASA documents the failure location of each one, so the
# MAGIC matching fault band should be the one that rises:
# MAGIC
# MAGIC | Bearing | Documented failure | Band to watch |
# MAGIC |---|---|---|
# MAGIC | 2nd_test_b1 | outer race | `env_bpfo_energy` |
# MAGIC | 1st_test_b3 | inner race | `env_bpfi_energy` |
# MAGIC | 1st_test_b4 | rolling element | `env_bsf_energy` |
# MAGIC
# MAGIC Reported as a table rather than asserted: whether the physics shows up
# MAGIC this cleanly in real data is the finding, not an assumption.

# COMMAND ----------

per_bearing = Window.partitionBy("bearing_id")

phased = (
    gold_tbl.filter(F.col("runs_to_failure"))
    .withColumn("life_fraction", F.col("elapsed_hours") / F.max("elapsed_hours").over(per_bearing))
    .withColumn(
        "phase",
        F.when(F.col("life_fraction") <= 0.2, "1_early")
        .when(F.col("life_fraction") >= 0.9, "2_late"),
    )
    .filter(F.col("phase").isNotNull())
)

display(
    phased.groupBy("bearing_id", "phase")
    .agg(
        F.round(F.avg("rms"), 4).alias("rms"),
        F.round(F.avg("kurtosis"), 2).alias("kurtosis"),
        F.round(F.avg("env_bpfo_energy"), 2).alias("env_bpfo"),
        F.round(F.avg("env_bpfi_energy"), 2).alias("env_bpfi"),
        F.round(F.avg("env_bsf_energy"), 2).alias("env_bsf"),
    )
    .orderBy("bearing_id", "phase")
)

# COMMAND ----------

spark.sql(f"OPTIMIZE {GOLD_TABLE}")
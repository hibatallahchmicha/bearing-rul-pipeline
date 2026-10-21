# Databricks notebook source
# /// script
# [tool.databricks.environment]
# environment_version = "5"
# ///
# MAGIC %md
# MAGIC # Silver — validated, labelled snapshots
# MAGIC
# MAGIC Reads Bronze, enforces data-quality rules, and attaches the labels
# MAGIC every downstream stage needs: elapsed operating hours and remaining
# MAGIC useful life.
# MAGIC
# MAGIC **Windows are deliberately not persisted here.** 883k windows of 2048
# MAGIC float32 is ~7 GB to store an intermediate that is a strided reshape
# MAGIC away from Bronze. The expensive stage is the FFT, which runs after
# MAGIC windowing regardless, so caching windows costs quota and saves no
# MAGIC compute. Windowing stays in-memory inside the Silver -> Gold transform.
# MAGIC
# MAGIC **The labelling trap this layer exists to catch.** Only some bearings
# MAGIC run to failure; the rest simply stop recording. Computing "hours until
# MAGIC the last snapshot" for a bearing that never failed would fabricate a
# MAGIC label asserting a failure that did not happen. `rul_hours` is NULL for
# MAGIC those bearings, and `runs_to_failure` says so explicitly.
# MAGIC
# MAGIC **Write mode.** Bronze is append-only and immutable; Silver is derived
# MAGIC and therefore `overwrite` -- always fully rebuildable from Bronze.

# COMMAND ----------

import os

print(os.getcwd())

# COMMAND ----------

# The repo is expected to be checked out as a Databricks Git folder, so the
# notebook can import the same tested `bearing_pdm` code that CI runs.
# Adjust if your workspace path differs.
import sys

REPO_PATH = "/Workspace/Repos/bearing-rul-pipeline"
if REPO_PATH not in sys.path:
    sys.path.append(REPO_PATH)

# COMMAND ----------
from pyspark.sql import Window
from pyspark.sql import functions as F

from bearing_pdm.ingestion import SAMPLES_PER_FILE
from bearing_pdm.rul_target import DEFAULT_RUL_CAP_HOURS

CATALOG = "bearing"
BRONZE_TABLE = f"{CATALOG}.bronze.raw_snapshots"
SILVER_TABLE = f"{CATALOG}.silver.snapshots"
REJECTED_TABLE = f"{CATALOG}.silver.rejected_snapshots"

# Bearings whose recording ends in an actual failure, per NASA's dataset
# documentation. Everything else is healthy reference data with no valid
# RUL label -- useful for the drift baseline, unusable as a training target.
FAILED_BEARINGS = {
    "1st_test_b3",  # inner race defect
    "1st_test_b4",  # rolling element defect
    "2nd_test_b1",  # outer race failure
    "3rd_test_b3",  # outer race failure
}

# Gross-corruption bound, not a calibration spec: real IMS amplitudes sit
# well under 1 g, so anything past this indicates a parsing fault or a
# damaged file rather than an unusual measurement.
MAX_PLAUSIBLE_AMPLITUDE_G = 50.0

spark.sql(f"CREATE SCHEMA IF NOT EXISTS {CATALOG}.silver")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Quality rules
# MAGIC
# MAGIC Each rule targets a failure that is *invisible downstream*: a truncated
# MAGIC or dead-channel snapshot still produces perfectly well-formed features,
# MAGIC just wrong ones. Nothing here silently drops rows -- failures are
# MAGIC quarantined to `rejected_snapshots` with the reason attached, so the
# MAGIC rejects are inspectable rather than gone.

# COMMAND ----------

bronze = spark.table(BRONZE_TABLE)

max_abs_amplitude = F.array_max(F.transform("waveform", lambda x: F.abs(x)))
has_bad_value = F.exists("waveform", lambda x: x.isNull() | F.isnan(x))

checked = (
    bronze.withColumn("max_abs_amplitude", max_abs_amplitude)
    .withColumn("qc_expected_length", F.col("n_samples") == F.lit(SAMPLES_PER_FILE))
    .withColumn("qc_all_values_finite", ~has_bad_value)
    .withColumn("qc_channel_alive", F.col("max_abs_amplitude") > F.lit(0.0))
    .withColumn(
        "qc_amplitude_plausible",
        F.col("max_abs_amplitude") < F.lit(MAX_PLAUSIBLE_AMPLITUDE_G),
    )
)

checked = checked.withColumn(
    "is_valid",
    F.col("qc_expected_length")
    & F.col("qc_all_values_finite")
    & F.col("qc_channel_alive")
    & F.col("qc_amplitude_plausible"),
).withColumn(
    "rejection_reason",
    F.when(~F.col("qc_expected_length"), F.lit("unexpected_sample_count"))
    .when(~F.col("qc_all_values_finite"), F.lit("null_or_nan_sample"))
    .when(~F.col("qc_channel_alive"), F.lit("dead_channel_all_zero"))
    .when(~F.col("qc_amplitude_plausible"), F.lit("implausible_amplitude"))
    .otherwise(F.lit(None)),
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Labels
# MAGIC
# MAGIC `elapsed_hours` is measured from each bearing's own first snapshot. It
# MAGIC is legitimately available in deployment -- you know when a bearing was
# MAGIC installed -- which is what makes the linear-degradation baseline a fair
# MAGIC comparison rather than an oracle.
# MAGIC
# MAGIC `rul_hours` is NULL wherever `runs_to_failure` is false. See the header.

# COMMAND ----------

per_bearing = Window.partitionBy("bearing_id")

labelled = (
    checked.filter(F.col("is_valid"))
    .withColumn("runs_to_failure", F.col("bearing_id").isin(list(FAILED_BEARINGS)))
    .withColumn("first_ts", F.min("snapshot_ts").over(per_bearing))
    .withColumn("last_ts", F.max("snapshot_ts").over(per_bearing))
    .withColumn(
        "elapsed_hours",
        (F.col("snapshot_ts").cast("long") - F.col("first_ts").cast("long")) / 3600.0,
    )
    .withColumn(
        "rul_hours",
        F.when(
            F.col("runs_to_failure"),
            (F.col("last_ts").cast("long") - F.col("snapshot_ts").cast("long")) / 3600.0,
        ).otherwise(F.lit(None).cast("double")),
    )
    .withColumn(
        "rul_hours_capped",
        F.when(
            F.col("runs_to_failure"),
            F.least(F.col("rul_hours"), F.lit(DEFAULT_RUL_CAP_HOURS)),
        ).otherwise(F.lit(None).cast("double")),
    )
    .withColumn("silver_processed_at", F.current_timestamp())
    .drop("first_ts", "last_ts")
)

# COMMAND ----------

(
    labelled.write.format("delta")
    .mode("overwrite")
    .option("overwriteSchema", "true")
    .saveAsTable(SILVER_TABLE)
)

rejected = checked.filter(~F.col("is_valid"))
(
    rejected.drop("waveform")
    .write.format("delta")
    .mode("overwrite")
    .option("overwriteSchema", "true")
    .saveAsTable(REJECTED_TABLE)
)

print(f"silver rows: {spark.table(SILVER_TABLE).count()}")
print(f"rejected rows: {spark.table(REJECTED_TABLE).count()}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Boundary assertions
# MAGIC
# MAGIC These guard the Silver -> Gold contract. The RUL checks matter most:
# MAGIC a mislabelled target is the one error that makes every downstream
# MAGIC metric confidently wrong rather than visibly broken.

# COMMAND ----------

silver = spark.table(SILVER_TABLE)

bad_labels = silver.filter(~F.col("runs_to_failure") & F.col("rul_hours").isNotNull()).count()
assert bad_labels == 0, f"{bad_labels} non-failing snapshots carry an RUL label"

missing_labels = silver.filter(F.col("runs_to_failure") & F.col("rul_hours").isNull()).count()
assert missing_labels == 0, f"{missing_labels} failing snapshots are missing an RUL label"

negative = silver.filter(F.col("rul_hours") < 0).count()
assert negative == 0, f"{negative} snapshots have negative RUL"

over_cap = silver.filter(F.col("rul_hours_capped") > DEFAULT_RUL_CAP_HOURS).count()
assert over_cap == 0, f"{over_cap} snapshots exceed the RUL cap"

n_failing = silver.filter(F.col("runs_to_failure")).select("bearing_id").distinct().count()
assert n_failing >= 2, (
    f"only {n_failing} run-to-failure bearing(s) in Silver -- "
    "leave-one-bearing-out needs at least 2"
)

display(
    silver.groupBy("bearing_id", "runs_to_failure")
    .agg(
        F.count("*").alias("snapshots"),
        F.round(F.max("elapsed_hours"), 1).alias("lifespan_hours"),
        F.round(F.min("rul_hours"), 1).alias("min_rul"),
        F.round(F.max("rul_hours"), 1).alias("max_rul"),
    )
    .orderBy("bearing_id")
)

# COMMAND ----------

spark.sql(f"OPTIMIZE {SILVER_TABLE}")
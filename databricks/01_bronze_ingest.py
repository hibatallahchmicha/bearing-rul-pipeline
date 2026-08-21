# Databricks notebook source
# MAGIC %md
# MAGIC # Bronze — raw vibration snapshots, ingested as-is
# MAGIC
# MAGIC One row per (snapshot, bearing). The waveform is kept as an array
# MAGIC column so that one row remains one physical measurement of one
# MAGIC bearing at one instant — the natural grain of the source data.
# MAGIC
# MAGIC **Immutability contract.** Nothing in this notebook filters, resamples,
# MAGIC or cleans. Every sample of every file in scope lands here unchanged.
# MAGIC Quality checks and windowing belong to Silver. This is what makes
# MAGIC `VERSION AS OF` meaningful: rolling back reaches complete data, not an
# MAGIC earlier lossy approximation.
# MAGIC
# MAGIC **Declared scope.** Only the four bearings observed running to failure
# MAGIC are ingested; the remaining eight never fail within their recordings and
# MAGIC carry no RUL label. That boundary is recorded per row in `ingest_scope`
# MAGIC so it is queryable rather than remembered.

# COMMAND ----------

from datetime import datetime

import numpy as np
from pyspark.sql import Row
from pyspark.sql import functions as F
from pyspark.sql.types import (
    ArrayType,
    FloatType,
    IntegerType,
    StringType,
    StructField,
    StructType,
    TimestampType,
)

CATALOG = "bearing"
BRONZE_SCHEMA = "bronze"
BRONZE_TABLE = f"{CATALOG}.{BRONZE_SCHEMA}.raw_snapshots"

# UC Volume holding the extracted NASA IMS text files.
VOLUME_ROOT = f"/Volumes/{CATALOG}/raw/ims"

SAMPLE_RATE_HZ = 20_000
SAMPLES_PER_FILE = 20_480
TIMESTAMP_FORMAT = "%Y.%m.%d.%H.%M.%S"
INGEST_SCOPE = "tests_1_2_all_bearings"

# Set to include "3rd_test" once storage quota is confirmed. Test 3 is
# 25,296 of the 37,856 total rows (~2 GB on its own), so it is opt-in
# rather than default -- a declared scope decision, recorded per row in
# `ingest_scope`, not a silent sample.
TEST_SETS = ["1st_test", "2nd_test"]

# (bearing_id, test_set, subdirectory, channel_index)
# Test 1 records two accelerometers per bearing (channels 0-7); the first
# of each pair is used. Tests 2 and 3 record one per bearing (channels 0-3).
ALL_BEARINGS = [
    ("1st_test_b1", "1st_test", "1st_test/1st_test", 0),
    ("1st_test_b2", "1st_test", "1st_test/1st_test", 2),
    ("1st_test_b3", "1st_test", "1st_test/1st_test", 4),
    ("1st_test_b4", "1st_test", "1st_test/1st_test", 6),
    ("2nd_test_b1", "2nd_test", "2nd_test/2nd_test", 0),
    ("2nd_test_b2", "2nd_test", "2nd_test/2nd_test", 1),
    ("2nd_test_b3", "2nd_test", "2nd_test/2nd_test", 2),
    ("2nd_test_b4", "2nd_test", "2nd_test/2nd_test", 3),
    ("3rd_test_b1", "3rd_test", "3rd_test/4th_test/txt", 0),
    ("3rd_test_b2", "3rd_test", "3rd_test/4th_test/txt", 1),
    ("3rd_test_b3", "3rd_test", "3rd_test/4th_test/txt", 2),
    ("3rd_test_b4", "3rd_test", "3rd_test/4th_test/txt", 3),
]

BEARINGS = [b for b in ALL_BEARINGS if b[1] in TEST_SETS]

# COMMAND ----------

spark.sql(f"CREATE CATALOG IF NOT EXISTS {CATALOG}")
spark.sql(f"CREATE SCHEMA IF NOT EXISTS {CATALOG}.{BRONZE_SCHEMA}")
spark.sql(f"CREATE SCHEMA IF NOT EXISTS {CATALOG}.raw")
spark.sql(f"CREATE VOLUME IF NOT EXISTS {CATALOG}.raw.ims")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Schema, declared explicitly
# MAGIC
# MAGIC Written out rather than inferred. Schema inference on 11,000 text files
# MAGIC costs a full extra scan and silently changes type when a file is
# MAGIC malformed — the failure mode being guarded against here is a column
# MAGIC quietly becoming `string` because one file has a stray character, which
# MAGIC would propagate through every downstream layer without erroring.

# COMMAND ----------

BRONZE_SCHEMA_SPEC = StructType(
    [
        StructField("bearing_id", StringType(), nullable=False),
        StructField("test_set", StringType(), nullable=False),
        StructField("channel", IntegerType(), nullable=False),
        StructField("snapshot_ts", TimestampType(), nullable=False),
        StructField("source_file", StringType(), nullable=False),
        StructField("sample_rate_hz", IntegerType(), nullable=False),
        StructField("n_samples", IntegerType(), nullable=False),
        # float32: the source ASCII carries three decimal places, so float64
        # would store precision the instrument never recorded, at 2x the cost.
        StructField("waveform", ArrayType(FloatType()), nullable=False),
        StructField("ingest_scope", StringType(), nullable=False),
        StructField("ingested_at", TimestampType(), nullable=False),
    ]
)

# COMMAND ----------


def read_one_snapshot(path: str, channel: int) -> np.ndarray:
    """Load a single channel from one tab-delimited snapshot file."""
    return np.loadtxt(path)[:, channel].astype(np.float32)


def snapshot_rows(bearing_id, test_set, subdir, channel, ingested_at):
    """Yield one Bronze Row per snapshot file for a single bearing."""
    directory = f"{VOLUME_ROOT}/{subdir}"
    filenames = sorted(f.name for f in dbutils.fs.ls(directory) if not f.name.endswith("/"))

    for name in filenames:
        waveform = read_one_snapshot(f"{directory}/{name}", channel)
        yield Row(
            bearing_id=bearing_id,
            test_set=test_set,
            channel=channel,
            snapshot_ts=datetime.strptime(name, TIMESTAMP_FORMAT),
            source_file=f"{subdir}/{name}",
            sample_rate_hz=SAMPLE_RATE_HZ,
            n_samples=int(waveform.shape[0]),
            waveform=waveform.tolist(),
            ingest_scope=INGEST_SCOPE,
            ingested_at=ingested_at,
        )


# COMMAND ----------

# MAGIC %md
# MAGIC ## Write, one bearing at a time
# MAGIC
# MAGIC Appending per bearing rather than building one 950 MB collection in
# MAGIC driver memory. Each append is its own Delta version, so a failure
# MAGIC partway through leaves the previous bearings intact and committed —
# MAGIC and the version history shows exactly which bearing was added when.
# MAGIC
# MAGIC `overwriteSchema` is deliberately NOT set. If a future edit changes the
# MAGIC schema, this write should fail loudly rather than silently rewriting a
# MAGIC table whose earlier versions then no longer match their own schema.

# COMMAND ----------

ingested_at = datetime.utcnow()

for bearing_id, test_set, subdir, channel in BEARINGS:
    rows = list(snapshot_rows(bearing_id, test_set, subdir, channel, ingested_at))
    df = spark.createDataFrame(rows, schema=BRONZE_SCHEMA_SPEC)
    (
        df.write.format("delta")
        .mode("append")
        .option("mergeSchema", "false")
        .saveAsTable(BRONZE_TABLE)
    )
    print(f"{bearing_id}: appended {len(rows)} snapshots")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Ingestion assertions
# MAGIC
# MAGIC These run at the Bronze boundary and are the first of the medallion
# MAGIC data-quality gates. They check the properties whose violation would be
# MAGIC invisible downstream: a truncated file still produces valid features,
# MAGIC just wrong ones.

# COMMAND ----------

bronze = spark.table(BRONZE_TABLE)

wrong_length = bronze.filter(F.col("n_samples") != SAMPLES_PER_FILE).count()
assert wrong_length == 0, f"{wrong_length} snapshots do not have {SAMPLES_PER_FILE} samples"

duplicates = (
    bronze.groupBy("bearing_id", "snapshot_ts").count().filter(F.col("count") > 1).count()
)
assert duplicates == 0, f"{duplicates} duplicate (bearing_id, snapshot_ts) pairs"

n_bearings = bronze.select("bearing_id").distinct().count()
assert n_bearings == len(BEARINGS), f"expected {len(BEARINGS)} bearings, found {n_bearings}"

display(
    bronze.groupBy("bearing_id", "test_set")
    .agg(
        F.count("*").alias("snapshots"),
        F.min("snapshot_ts").alias("first_seen"),
        F.max("snapshot_ts").alias("failed_at"),
    )
    .orderBy("bearing_id")
)

# COMMAND ----------

spark.sql(f"OPTIMIZE {BRONZE_TABLE}")
# Databricks notebook source
# MAGIC %md
# MAGIC # Bronze — raw vibration snapshots, ingested as-is
# MAGIC
# MAGIC One row per (snapshot, bearing). The waveform is kept as an array
# MAGIC column so that one row remains one physical measurement of one
# MAGIC bearing at one instant — the natural grain of the source data.
# MAGIC
# MAGIC **Immutability contract.** Nothing here filters, resamples, or cleans.
# MAGIC Every sample of every file in scope lands unchanged. Quality checks and
# MAGIC labelling belong to Silver. This is what makes `VERSION AS OF`
# MAGIC meaningful: rolling back reaches complete data, not an earlier lossy
# MAGIC approximation.
# MAGIC
# MAGIC **Declared scope vs. batch.** `DECLARED_SCOPE` is what this Bronze
# MAGIC table is intended to contain and is recorded on every row.
# MAGIC `TEST_SETS` is only what *this run* ingests, so an interrupted load can
# MAGIC be resumed one test set at a time without re-appending what is already
# MAGIC committed.

# COMMAND ----------

from datetime import UTC, datetime

import numpy as np
import pandas as pd
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

# If Free Edition would not let you create a `bearing` catalog, set this to
# the catalog your Volume actually lives in (e.g. "workspace").
CATALOG = "bearing"
BRONZE_SCHEMA = "bronze"
BRONZE_TABLE = f"{CATALOG}.{BRONZE_SCHEMA}.raw_snapshots"

VOLUME_ROOT = f"/Volumes/{CATALOG}/raw/ims"

SAMPLE_RATE_HZ = 20_000
SAMPLES_PER_FILE = 20_480
TIMESTAMP_FORMAT = "%Y.%m.%d.%H.%M.%S"

# Rows buffered before each append. Keeps driver memory bounded: one test-1
# bearing is ~2,156 rows x 82 KB ~= 177 MB if collected whole, which a small
# serverless driver may not survive. Each chunk is its own Delta version, so
# a failure partway leaves earlier chunks committed.
CHUNK_SIZE = 500

# COMMAND ----------

# MAGIC %md
# MAGIC ## Scope
# MAGIC
# MAGIC Test 3 is 25,296 of the 37,856 possible rows (~2 GB alone), so it is
# MAGIC opt-in rather than default. Tests 1 and 2 give 12,560 rows and three
# MAGIC run-to-failure bearings — enough for leave-one-bearing-out.

# COMMAND ----------

DECLARED_SCOPE = ["1st_test", "2nd_test"]
TEST_SETS = ["1st_test", "2nd_test"]

INGEST_SCOPE = "+".join(DECLARED_SCOPE)

# (bearing_id, test_set, subdirectory, channel_index)
# Test 1 records two accelerometers per bearing (channels 0-7); the first of
# each pair is used. Tests 2 and 3 record one per bearing (channels 0-3).
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
EXPECTED_BEARINGS = [b for b in ALL_BEARINGS if b[1] in DECLARED_SCOPE]

print(f"this run ingests {len(BEARINGS)} bearings from {TEST_SETS}")
print(f"declared scope is {len(EXPECTED_BEARINGS)} bearings from {DECLARED_SCOPE}")

# COMMAND ----------

spark.sql(f"CREATE CATALOG IF NOT EXISTS {CATALOG}")
spark.sql(f"CREATE SCHEMA IF NOT EXISTS {CATALOG}.{BRONZE_SCHEMA}")
spark.sql(f"CREATE SCHEMA IF NOT EXISTS {CATALOG}.raw")
spark.sql(f"CREATE VOLUME IF NOT EXISTS {CATALOG}.raw.ims")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Schema, declared explicitly
# MAGIC
# MAGIC Written out rather than inferred. Inference over 3,000+ text files
# MAGIC costs an extra full scan and silently changes type when a file is
# MAGIC malformed — the failure mode being guarded against is a column quietly
# MAGIC becoming `string` because one file has a stray character, which would
# MAGIC then propagate through every downstream layer without erroring.

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
    """Load a single channel from one tab-delimited snapshot file.

    pandas' C parser with `usecols` reads only the column we need -- test 1
    files carry 8 channels, so this parses an eighth of the bytes -- and is
    roughly an order of magnitude faster than np.loadtxt at this file size.
    Across 3,140 files that is the difference between minutes and an hour.
    """
    return (
        pd.read_csv(path, sep="\t", header=None, usecols=[channel], dtype=np.float32)
        .iloc[:, 0]
        .to_numpy()
    )


def snapshot_leaf_name(name: str) -> str:
    """Return the final path component, whichever separator was used.

    Some archives extract with Windows-style backslash separators embedded
    in the entry name, so what `dbutils.fs.ls` reports as the "filename" can
    be '2nd_test\\2nd_test\\2004.02.12.10.32.39'. Linux does not treat `\\`
    as a separator, so those become real filenames containing backslashes.
    """
    return name.replace("\\", "/").rsplit("/", 1)[-1]


def parse_snapshot_timestamp(name: str) -> datetime:
    """Extract the recording timestamp from a snapshot filename."""
    return datetime.strptime(snapshot_leaf_name(name), TIMESTAMP_FORMAT)


def snapshot_rows(bearing_id, test_set, subdir, channel, ingested_at):
    """Yield one Bronze Row per snapshot file for a single bearing."""
    directory = f"{VOLUME_ROOT}/{subdir}"
    names = [f.name for f in dbutils.fs.ls(directory) if not f.name.endswith("/")]
    names.sort(key=snapshot_leaf_name)

    for name in names:
        waveform = read_one_snapshot(f"{directory}/{name}", channel)
        yield Row(
            bearing_id=bearing_id,
            test_set=test_set,
            channel=channel,
            snapshot_ts=parse_snapshot_timestamp(name),
            source_file=f"{subdir}/{name}",
            sample_rate_hz=SAMPLE_RATE_HZ,
            n_samples=int(waveform.shape[0]),
            waveform=waveform.tolist(),
            ingest_scope=INGEST_SCOPE,
            ingested_at=ingested_at,
        )


def write_chunk(rows) -> None:
    """Append one buffered chunk to the Bronze table.

    `overwriteSchema` is deliberately absent. If a future edit changes the
    schema this write should fail loudly rather than silently rewriting a
    table whose earlier versions then no longer match their own schema.
    """
    if not rows:
        return
    (
        spark.createDataFrame(rows, schema=BRONZE_SCHEMA_SPEC)
        .write.format("delta")
        .mode("append")
        .option("mergeSchema", "false")
        .saveAsTable(BRONZE_TABLE)
    )


# COMMAND ----------

# MAGIC %md
# MAGIC ## Ingest
# MAGIC
# MAGIC Append-only. Re-running this with a `TEST_SETS` already present in the
# MAGIC table adds a second copy and the duplicate assertion below will fire —
# MAGIC correct behaviour for an immutable layer. To start over:
# MAGIC `DROP TABLE IF EXISTS bearing.bronze.raw_snapshots`

# COMMAND ----------

ingested_at = datetime.now(UTC)

for bearing_id, test_set, subdir, channel in BEARINGS:
    buffer, written = [], 0
    for row in snapshot_rows(bearing_id, test_set, subdir, channel, ingested_at):
        buffer.append(row)
        if len(buffer) >= CHUNK_SIZE:
            write_chunk(buffer)
            written += len(buffer)
            buffer = []
            print(f"  {bearing_id}: {written} snapshots written")
    write_chunk(buffer)
    written += len(buffer)
    print(f"{bearing_id}: done — {written} snapshots")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Ingestion assertions
# MAGIC
# MAGIC These check the properties whose violation would be invisible
# MAGIC downstream: a truncated file still produces perfectly well-formed
# MAGIC features, just wrong ones. They assert against `DECLARED_SCOPE`, not
# MAGIC against whatever this run happened to ingest.

# COMMAND ----------

bronze = spark.table(BRONZE_TABLE)

wrong_length = bronze.filter(F.col("n_samples") != SAMPLES_PER_FILE).count()
assert wrong_length == 0, f"{wrong_length} snapshots do not have {SAMPLES_PER_FILE} samples"

duplicates = bronze.groupBy("bearing_id", "snapshot_ts").count().filter(F.col("count") > 1).count()
assert duplicates == 0, f"{duplicates} duplicate (bearing_id, snapshot_ts) pairs"

n_bearings = bronze.select("bearing_id").distinct().count()
assert n_bearings == len(EXPECTED_BEARINGS), (
    f"expected {len(EXPECTED_BEARINGS)} bearings for scope {DECLARED_SCOPE}, found {n_bearings}"
)

print(f"total rows: {bronze.count()}")

display(
    bronze.groupBy("bearing_id", "test_set")
    .agg(
        F.count("*").alias("snapshots"),
        F.min("snapshot_ts").alias("first_seen"),
        F.max("snapshot_ts").alias("last_seen"),
    )
    .orderBy("bearing_id")
)

# COMMAND ----------

spark.sql(f"OPTIMIZE {BRONZE_TABLE}")
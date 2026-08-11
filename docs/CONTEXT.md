# Project context — bearing-rul-pipeline

A reference doc for coming back to this project cold and remembering how
the pieces fit together. Written as of the point where all 10 core
modules plus the end-to-end runner are built and have been run against
real data; see the table below for current status.

## The problem, in one paragraph

A rolling-element bearing wears out gradually — a defect forms on the
outer race, inner race, a roller, or the cage, and grows until the bearing
fails. An accelerometer bolted to the housing captures that as raw
vibration, sampled at 20 kHz. The goal of this pipeline is to turn that
raw high-frequency signal into (a) a clear indication of *which part* of
the bearing is degrading, (b) a single trend line tracking *how far*
degraded it is, and eventually (c) an estimate of *how much life is left*
— early enough that a maintenance team could act on it before failure,
not just after.

## Data flow

```
raw .rar files (NASA IMS dataset)
        |
        v
[1] ingestion.py        -- read + chronologically-sorted stream replay
        |  (one channel's 1-D signal per bearing)
        v
[2] segmentation.py      -- cut into overlapping fixed-length windows
        |  (n_windows, 2048)
        |-----------------------------------------.
        v                                          v
[3] features_time.py           [4] features_freq.py (needs fault_frequencies.py)
    RMS, kurtosis, skew,           FFT + envelope-spectrum energy at
    crest factor, p2p, std         BPFO/BPFI/BSF/FTF bands
        |                                          |
        '------------------.  .--------------------'
                             v v
                    merged feature dict (~15 features/window)
                             |
                             v
                  [5] health_index.py
                  Mahalanobis distance from a healthy baseline
                  -> one degradation score per window
                             |
                             v
                  [6] evaluation.py
                  leave-one-bearing-out folds -- defines which
                  bearings a baseline/model is allowed to see
                             |
                             v
                  [7] rul_model.py
                  healthy/degrading/critical staging (chosen over RUL
                  regression -- see Design decisions below), trained/
                  scored per evaluation.py's folds
                             |
                             v
                  [8] cost_threshold.py
                  expected-cost threshold sweep -- false-alarm vs
                  missed-failure tradeoff (currently leak-tainted in
                  scripts/run_pipeline.py -- see Known limitations)
                             |
                             v
                  [9] drift.py + retrain trigger stub
                  compares incoming feature distributions to the
                  training baseline; documents when a retrain should fire
                             |
                             v
                  scripts/run_pipeline.py
                  wires all of the above together on real IMS data,
                  leave-one-bearing-out across the 4 real failing bearings
```

Ingestion's `stream_snapshots()` also plays the role of "simulated live
sensor feed" (module 1's second job) — a `scripts/ingest_stream.py` CLI
wrapper around it is still planned but not required for the modules
above to work.

## Module status

| # | Module | File | Status |
|---|--------|------|--------|
| 1 | Ingestion + streaming replay | `bearing_pdm/ingestion.py` | Built, verified |
| 2 | Segmentation | `bearing_pdm/segmentation.py` | Built, verified |
| 3 | Time-domain features | `bearing_pdm/features_time.py` | Built, verified |
| — | Fault-frequency physics | `bearing_pdm/fault_frequencies.py` | Built, verified |
| 4 | Frequency-domain features | `bearing_pdm/features_freq.py` | Built, verified |
| 5 | Health index | `bearing_pdm/health_index.py` | Built, verified |
| 6 | Evaluation (leave-one-bearing-out) | `bearing_pdm/evaluation.py` | Built, verified |
| 7 | RUL / stage model | `bearing_pdm/rul_model.py` | Built, verified (staging chosen over regression — see below) |
| 8 | Cost-sensitive threshold | `bearing_pdm/cost_threshold.py` | Built, verified |
| 9 | Drift detection + retrain stub | `bearing_pdm/drift.py` | Built, verified |
| — | End-to-end runner | `scripts/run_pipeline.py` | Built, run on real data |
| — | Streaming CLI wrapper | `scripts/ingest_stream.py` | Not started |
| — | Tests | `tests/` | Built, 17 passing |
| — | CI | `.github/workflows/ci.yml` | Built, lint + pytest on 3.11/3.12 |

All 10 core modules were first verified against synthetic/known-signal
inputs (see `CHECKPOINTS.md` for the reasoning behind each and the
checkpoint Q&A), then run end-to-end against the real, fully-extracted
IMS dataset via `scripts/run_pipeline.py`. **Real result: mean leave-one-
bearing-out stage accuracy 0.544** across the 4 real failing bearings —
see "Real-data results" below for the full story, including a genuine
bug found and fixed along the way. Real extracted paths, irregular per
test set:

- Test 1: `data/raw/4. Bearings/1st_test/1st_test/` (2156 files, 8 cols)
- Test 2: `data/raw/4. Bearings/2nd_test/2nd_test/` (984 files, 4 cols)
- Test 3: `data/raw/4. Bearings/3rd_test/4th_test/txt/` (6324 files, 4
  cols — NASA's own archive labels this "4th_test" internally)

## The dataset

NASA IMS bearing dataset — three separate test runs, each with 4 bearings
on one shaft at a constant 2000 RPM under constant radial load, recorded
until failure. Only 4 of the 12 total bearings across all three tests
actually run to failure within the recording (test 1: bearings 3 & 4;
test 2: bearing 1; test 3: bearing 3) — the rest just run healthy for the
whole recording window. That small number of true run-to-failure
trajectories is *why* evaluation uses leave-one-bearing-out instead of a
random split — see `CHECKPOINTS.md`, module 6 and 7 entries, for the full
reasoning.

Each snapshot file is ~1 second of vibration (20480 samples at 20 kHz),
recorded roughly every 10 minutes across a bearing's life. Test 1 has 8
channels (2 accelerometers per bearing x 4 bearings); tests 2 and 3 have 4
channels (1 per bearing).

## Design decisions made

- **RUL regression vs. health-stage classification (module 7):** decided
  in favor of stage classification (healthy/degrading/critical) as the
  primary output. With only 4 labeled run-to-failure trajectories in the
  entire dataset, a regressor learning "trajectory shape -> time-to-
  failure" would effectively be fit on 4 independent examples — high
  overfitting risk. Classification only needs to learn thresholds on the
  health index, and each bearing still contributes thousands of self-
  labeled windows. RUL regression remains a documented option for later,
  not built. Full reasoning in `CHECKPOINTS.md`, module 8 entry.
- **Cost model constants (module 8):** illustrative only (missed failure
  assumed 10x costlier than a false alarm) — not sourced from a real
  maintenance budget, since none exists for this project. Stated as a
  parameter, not a finding.
- **Drift definition (module 9):** per-feature two-sample KS test,
  reference batch vs. current batch, alarm only when a substantial
  fraction of features drift together (not one feature alone, to avoid
  false alarms from multiple-testing noise).

## Real-data results

First end-to-end run: mean stage accuracy **0.218** — worse than the
~70% trivial "always healthy" baseline, a strong signal of a real bug
rather than a weak model. Diagnosis: the healthy baseline was pooling
*raw* early-life features across 3 different physical bearings, which
have different noise floors/mounting/sensor gain even when nominally
identical — so every window of the held-out 4th bearing looked anomalous
relative to a baseline that didn't represent it. Fix: z-score each
bearing's features against its *own* early-life mean/std before pooling
(`normalize_to_own_baseline` in `run_pipeline.py`) — legitimate per-asset
calibration, not leakage, since it never uses failure-time information.
Result after the fix: **0.544**, more than double, with informative
fold-to-fold variance (0.380-0.726) expected given only 4 real units.
Full diagnostic narrative in `CHECKPOINTS.md`.

## Known limitations (stated, not hidden)

- **Cost-threshold evaluation leakage — found and fixed.** Originally
  tuned directly against the held-out unit's own true labels. Fixed by
  sweeping on pooled *train*-unit scores/labels only, then scoring that
  fixed threshold against the held-out unit. Full before/after numbers in
  `CHECKPOINTS.md`.
- **Alarm threshold doesn't transfer well across bearings, now that the
  leak is fixed and this is visible:** the train-derived threshold misses
  70-80% of real critical windows on 3 of 4 held-out bearings. Likely
  cause: per-unit normalization aligns each bearing's healthy period, but
  not how a critical excursion scales — different failure modes (inner
  race vs. outer race vs. roller) may produce very different magnitude
  jumps. Not fixed further here; stated as a real, disclosed limitation.
- **Small-n evaluation:** only 4 real run-to-failure trajectories exist
  in the whole dataset. Leave-one-bearing-out is the right response, but
  4 folds is still a small, high-variance sample — reported honestly
  rather than oversold.

## Still open

- `scripts/ingest_stream.py` CLI wrapper around `ingestion.stream_snapshots`.
- `README.md` (public) — intentionally last.
- Whether `NOTES_PRIVATE.md` is still needed given `CHECKPOINTS.md` already
  covers ELI5-to-expert reasoning and interview Q&A for every module.

## Where the reasoning lives

- `CHECKPOINTS.md` (gitignored, local only) — every "why this choice"
  question asked during development, with full answers. This is the
  interview-prep record.
- `CLAUDE.md` (gitignored, local only) — the working rules for this repo
  and environment quirks.
- This file — the technical map, safe to extend as new modules land.

# Evidence-bottleneck supervised execution preparation

This package connects the accepted A/B/C experiment to a real bounded parent,
scoring CLI and compact exporter. It does **not** run a model, read real gold,
submit a cluster job, repeat tokenizer job 31996384, or change the experiment.
The original CPU job remains FAILED after its completed tokenizer probes.

## Frozen experiment, new execution connection

The input, A/B/C prompts, decoding, shared selection, 96-call ceiling, 72 planned
route results and scoring/gate remain those in
[the accepted CPU report](SCIFACT_EVIDENCE_BOTTLENECK_CPU_20261002.md).
The strongest archived v2 top1 and original v3 top1 remain in the original scorer.
No inference gain, independent test result or resume metric follows from wiring.

```text
Slurm wrapper: complete source guard + outer wall-clock bound
  -> parent: validate exact authorized release; reserve unique run
  -> prepare: verify pinned runtime + frozen TRAIN24 metadata
              hash generator archive, extract only generator into scratch, read-only
  -> immutable runtime-paths receipt (actual scratch model path + release SHA)
  -> existing bottleneck runner CLI (separate process)
  -> bounded_worker wait/kill/reap + release/path-bound worker-exit.json
  -> physical cost receipt (always 72 planned slots, including failures)
  -> original scorer CLI (separate bounded process, original audit and gold callback)
  -> scorer whitelist compact -> parent verifies scorer exit -> final compact
```

The release freezes the relative model location `input/models/generator/model`.
It contains **no invented absolute future model directory**. The parent passes
the actual extracted path to both CLIs; each checks the same runtime binding.
The copied release is byte-identical to the coordinator's immutable release.
The wrapper's private scratch and source are checked before preparation.
Prepared claims/frames are reused by exact hashes from the previously consumed
TRAIN24; no reselection or new preparation dataset is introduced.

Inference and reports are siblings, never nested. The parent does not create
the inference directory before `run_suite` reserves it exclusively. No resume,
overwrite or automatic retry is allowed. Failure before model loading still
produces the complete denominator and a zero-call receipt, not successful NEI.
Failure after generation preserves completed cost and unknown lower bounds.
Gold is opened only through the existing scorer's complete audit barrier. A
durable `gold-read-started.json` marker prevents a scoring timeout after the
barrier from being mislabeled gold-unread. Compact exports omit raw IDs/text/gold.

## Bounds and review findings

- Future resource ceiling: one A100, 8 CPUs, 32 GiB host RAM, 30 GiB scratch,
  30 minutes Slurm; this is **not authorization**.
- Outer wrapper: TERM at 1,770 seconds, kill-after 15 seconds. Parent subtracts
  wrapper startup from its 1,740-second total; prepare <=300, worker <=1,200,
  score <=180 seconds, each clipped by the same remaining deadline.
- Reuse existing `bounded_worker`, `cpu_stage`, signal handlers, physical ledger,
  generator archive verifier and read-only extraction. No new scheduler/framework.
- Old slot watchdog has a different directory layout. The thin new watcher
  checks `<claim>/<A|selector|B|C>/reserved.json`, ignoring completed stages and
  applying the existing 120-second stage bound (not 120 seconds per entire claim).
- Scoring still needs pinned Torch/Transformers for generation-config audit, but
  constructs no model. Do not send it to the old Torch-free tokenizer environment.

## Reproduction and release procedure

Local/CI synthetic verification uses fake original sentences and output wires;
it runs the actual runner and scoring CLIs in separate processes. Linux uses the
actual production POSIX supervisor; Windows tests only the same CLI/data flow.
Tests cover complete success, early exit, model-load failure, interrupted generation
with unknown usage, scoring timeout after the gold barrier, preparation failure,
exhausted total budget, wrong runtime model path/proof, and draft rejection.

```sh
python -m pytest -q tests/test_scifact_bottleneck_execution.py tests/test_scifact_evidence_bottleneck.py
python -m ruff check src scripts tests
bash -n hpc/scifact_evidence_bottleneck.sbatch

# Run only after a clean exact commit; no upload/submit/weights/gold here.
python scripts/package_scifact_bottleneck_execution.py \
  --repo . --commit <exact-40-hex-commit> --output <new-local-package-directory> \
  --bash <bash-executable> --preflight <original-31996384-preflight.json> \
  --preflight-sha 04d9338ee67fb57dd3e18da2faf777fb208f37a1621eae8a13ac6f474694f3fa
```

The generated `execution-source-receipt.json` binds source archive, actual new
wrapper and `release.draft.json` SHA. It reuses the original tokenizer receipt
only after exact model-facing component and probe-AST equality checks. The
executed tokenizer source remains 8e8c1ed, not the new execution source.

**Future command template, blocked until a new coordinator exact-hash release:**

```sh
# Upload only the frozen source.tar and exact wrapper; coordinator separately
# issues release.json at the path bound in the candidate. Never execute draft.
STAGE=/data/gpfs/projects/punim2936/portfolio_20260903/climate-public-retrieval-v2/envs/evidence-bottleneck-<source12>
export CLIMATE_SOURCE_TAR="$STAGE/source.tar"
export CLIMATE_SOURCE_SHA256=<accepted-source-archive-sha>
export CLIMATE_SOURCE_GIT=<exact-40-hex-commit>
export CLIMATE_BOTTLENECK_RELEASE_FILE="$STAGE/release.json"
export CLIMATE_BOTTLENECK_RELEASE_SHA=<new-authorized-release-sha>
sbatch --test-only "$STAGE/bottleneck.sbatch"
# Only once, after exact approval and duplicate/allocation check:
sbatch --output="$STAGE/slurm-%j.out" "$STAGE/bottleneck.sbatch"
```

The wrapper executes the parent with `--release` and `--release-sha`. The parent
constructs and records actual runner/scorer paths and calls them with the same
`--runtime-paths` and `--model-dir`. A successful worker without successful
scoring/compact is a failed execution, never a completed model experiment.

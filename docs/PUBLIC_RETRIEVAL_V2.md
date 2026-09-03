# Public retrieval v2 protocol

## Decision and evidence boundary

This cycle uses the leakage-audited CLIMATE-FEVER v2 `1,075/230/230`
partition. Training reads `train`; pilot, full candidate selection and every
ranking decision read `validation`. The historical test consumed by the 2026-08-25
BM25 baseline is now permanently sealed, including exact-baseline reruns. The v2
test assignment stays in the provenance manifest but is not exported as a claims
file and is not accepted by any selection command.

The verified manifest digest
`66bf9b2c0157505f504459e7b38285a2aeed0c14770f82c74d1f619a03551f16`
was originally frozen from CRLF-serialized JSON. Preparation therefore records
the native file digest but performs the protocol comparison after explicit CRLF
newline serialization. This is a byte-serialization normalization only: the
source digest, split algorithm, seed, assignments and expected digest are not
changed.

The sole independent transfer event is the official BEIR SciFact test split. The
archive URL and MD5 come from the
[BEIR dataset registry](https://github.com/beir-cellar/beir/wiki/Datasets-available),
and the equivalent MTEB dataset revision is pinned to
`cf10ab6856b15b0e670ef8ae5dae4e266c12d035`. Its qrels are opened only after the
climate configuration is frozen. A project-local atomic ledger permits one
completed evaluation; an infrastructure failure may retry the identical frozen
configuration twice, while a quality failure never triggers tuning or a rerun.

## Frozen adapter matrix

All runs use `Qwen/Qwen3-Embedding-0.6B` revision
`97b0c614be4d77ee51c0cef4e5f07c00f9eb65b3`, 1,024 dimensions, BF16,
claim-grouped hard-negative InfoNCE and LoRA on all linear modules.

| ID | Steps | Rank | Hard negatives | Temperature |
|---|---:|---:|---:|---:|
| `s100-r8-n4-t003` | 100 | 8 | 4 | 0.03 |
| `s100-r8-n8-t005` | 100 | 8 | 8 | 0.05 |
| `s100-r16-n4-t005` | 100 | 16 | 4 | 0.05 |
| `s300-r8-n8-t003` | 300 | 8 | 8 | 0.03 |
| `s300-r16-n4-t003` | 300 | 16 | 4 | 0.03 |
| `s300-r16-n8-t005` | 300 | 16 | 8 | 0.05 |

The cheap pilot uses a deterministic 64-query subset of decisive validation
claims and the full 5,240-document public corpus. Pilot mean deltas can advance
at most two fixed adapters to all decisive validation claims; pilot advancement
is not promotion. If every pilot is negative, the least-bad fixed adapter gets
one full diagnostic run so the negative result and downstream failure mode are
still measured. No hyperparameter is changed after observing a pilot.

Full promotion requires the Recall@5 paired-bootstrap 95% interval lower bound
to be greater than zero and non-negative mean deltas for MRR@10, nDCG@10 and
Evidence F1. Every paired comparison uses exactly 5,000 samples and seed
`20260903`.

## Frozen downstream comparison

The selected full-run adapter (or the labelled diagnostic fallback) is compared
as exact dense retrieval and HNSW, then combined with BM25 through RRF. The LTR
path is LightGBM LambdaMART over the exact RRF Top-100 serving pool. Candidate
width is 100, feature order is the repository `DEFAULT_FEATURES` tuple, and only
serving-reachable positives enter training. A hash of every reachable
`claim/evidence` pair is copied into the served model contract and must match
exactly before the ranker can be promoted.

The expensive path reranks the same RRF Top-100 with
`Qwen/Qwen3-Reranker-4B` revision
`22e683669bc0f0bd69640a1354a6d0aebcfeede5`, then applies one fixed balanced
1:1 rank fusion with RRF. There is no weight sweep.

All stages report Recall@5, MRR@10, nDCG@10, Evidence F1, paired intervals,
index build time/bytes, component P50/P95 and peak Torch GPU memory. The
diagnostic taxonomy includes spelling variants, year/numeric, entity,
geographic and semantic-paraphrase slices. These are deterministic text/gold
heuristics rather than human semantic labels.

## Reproduction and publication

Spartan jobs run from a detached exact commit under the isolated root
`/data/gpfs/projects/punim2936/portfolio_20260903/climate-public-retrieval-v2`.
The packed environment, data, logs, checkpoints, predictions and indexes remain
under that root as content-addressed tar archives rather than directory trees.
Every archive carries an internal payload manifest and its filename carries the
archive SHA-256; downstream allocations verify both before unpacking. To respect
the measured shared-filesystem inode ceiling, each allocation expands its input
archives, environment and model-download cache into the Spartan compute-node
temporary directory (`SLURM_TMPDIR` when exposed, otherwise the allocation's
`TMPDIR`); those caches are ephemeral, never Git artifacts, and never fall back
to `$HOME` or another project. The frozen peak budget is 34 new persistent
inodes with a 68-free-inode admission gate. Submission order is
`sbatch --test-only`, CPU/GPU preflight, six pilots, at most two full
candidates, downstream comparison, one SciFact transfer and compact
publication. See [the Spartan runbook](../hpc/README.md).

GitHub receives only the schema-validated compact record. It never receives
model caches, adapter checkpoints, predictions or large indexes. Until the
Slurm chain completes, this document describes a frozen protocol, not a quality
result.

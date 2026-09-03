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

The sole conditionally authorised independent transfer event is the official
BEIR SciFact test split. The
archive URL and MD5 come from the
[BEIR dataset registry](https://github.com/beir-cellar/beir/wiki/Datasets-available),
and the equivalent MTEB dataset revision is pinned to
`cf10ab6856b15b0e670ef8ae5dae4e266c12d035`. Its qrels are opened only after the
climate configuration is frozen. A project-local atomic ledger permits one
completed evaluation; an infrastructure failure may retry the identical frozen
configuration twice, while a quality failure never triggers tuning or a rerun.
The event is authorised only when a climate adapter passes the full promotion
gate. In the completed cycle no adapter was promoted, so SciFact was not run and
its qrels were never opened.

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

When an adapter passes, the selected full-run adapter is compared as exact dense
retrieval and HNSW, then combined with BM25 through RRF. If the mandatory
diagnostic fallback is invalid or fails promotion, the closeout loads the frozen
base embeddings only and labels the adapter as unused. The LTR
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

## Observed negative closeout

Final pilot array `30005221` completed all six registered configurations. Every
archive hash matched its filename and internal manifest. On the deterministic
64-query validation subset, every candidate exactly tied the base:

- Recall@5 `0.53203125`;
- MRR@10 `0.5338541667`;
- mean Recall@5, MRR@10, nDCG@10 and Evidence-F1 deltas all `0.0`.

CPU selector `30007095` therefore marked every adapter
`advance_eligible=false` and selected `s100-r16-n4-t005` only as the one
pre-registered diagnostic fallback. Full diagnostic `30007124` exited `1:0`:
the runtime reported missing adapter keys, then the reporting layer hit the
legacy `lower` versus `ci_lower` field mismatch. Because adapter integrity was
not established, this is not a valid full effectiveness result. It authorises
neither promotion nor a quality retry. Negative closeout `30007522` froze
`selected_candidate_id=null`, `selected_candidate_promoted=false`, base-only
downstream, forbidden test access and no external transfer. Its archive SHA-256
is `66222cda5a66f0cbc73a79ea39396de6e7f7abcd0c9ab1c73d0557fa4483b348`.

Validation-only downstream job `30007546` completed in `19:10` with exit `0:0`
and batch MaxRSS `15,177,324 K`. It loaded no adapter and reported:

| System | Recall@5 | MRR@10 | nDCG@10 | Evidence F1 |
|---|---:|---:|---:|---:|
| BM25 | 0.4401 | 0.4872 | 0.3966 | 0.2913 |
| Base dense Flat/HNSW | 0.5939 | 0.5909 | 0.5241 | 0.3730 |
| Base RRF | 0.5463 | 0.5995 | 0.5095 | 0.3571 |
| Base Top-100 LambdaMART | 0.6054 | 0.6276 | 0.5488 | 0.3828 |
| Base RRF + Qwen3-4B fusion | 0.6275 | 0.6197 | 0.5525 | 0.3969 |

All five non-BM25 paired comparisons used 5,000 samples and had positive
intervals versus BM25. HNSW matched Flat at Recall@5 `1.0`; the LambdaMART
training/serving contract passed. The downstream archive SHA-256 is
`b4d2a6ecda91d70d429ba98de40315884b80446e8fc604110803820bb7297503`.
These are 126-decisive-claim validation selection results. They do not rescue or
measure the invalid adapter, and they are not independent-test or online
evidence.

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
candidates, downstream comparison, a conditional SciFact transfer only after
promotion, and compact publication. See [the Spartan runbook](../hpc/README.md).

GitHub receives only the schema-validated compact record. It never receives
model caches, adapter checkpoints, predictions or large indexes. The completed
record explicitly stores zero SciFact evaluations, unopened qrels, all six
pilot outcomes, the invalid full diagnostic and the base-only downstream
boundary. Publish job `30009231` completed with exit `0:0`; the source archive
SHA-256 is
`91e0e4f08df015d9caec0f0dd22e2e349a4e136b50b743b32cce6666504e99f3`,
and the redacted JSON copied into Git has SHA-256
`8aa96c1800061aa9c454f3cdd3f78bdd693387ec90a8342344c5db381f481efb`.

## Evidence-grounded candidate bullet

- Executed a six-configuration, validation-only CLIMATE-FEVER LoRA retrieval
  gate on Spartan; preserved an exact-tie negative result, traced an adapter-key
  integrity failure, stopped promotion/SciFact by policy, and published
  content-addressed base-only BM25/dense/HNSW/RRF/LambdaMART/Qwen3 evidence with
  5,000-sample paired intervals and sealed test access.

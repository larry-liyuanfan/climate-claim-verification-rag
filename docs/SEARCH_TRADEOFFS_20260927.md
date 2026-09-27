# Search tradeoffs and representation integrity — 2026-09-27

Status: implementation and integrity preflight; not new model-quality results.
Base commit: `272ab93256202e517a8aebf36992fbdf93dc922e`.

## Findings and repair

The frozen public-v2 environment uses PEFT 0.15.2, whose loader does not
implement the supplied `key_mapping`. Missing checkpoint keys can yield a
warning while initialized LoRA parameters still exist. Counting parameters
did not prove restoration. Sources: [PEFT loader](https://github.com/huggingface/peft/blob/v0.15.2/src/peft/peft_model.py),
[restoration code](https://github.com/huggingface/peft/blob/v0.15.2/src/peft/utils/save_and_load.py).

`adapter_integrity.py` accepts exact checkpoint keys or only the known
`base_model.model.model.* -> base_model.model.*` conversion. Missing keys,
collisions, unknown keys, shapes, non-finite values and unequal restored values
fail before corpus encoding. Fixed label-free query/document probes compare
enabled / disabled / re-enabled adapters after the actual embedding pipeline,
including normalization and dimension truncation. A reproducible difference
over 1e-7 is an integrity check, not a quality improvement.

Public-v2 predictions also retained only five IDs while reporting Top-10
metrics. New ranking lists retain Top-100; ranking metrics use their own
cutoffs and Evidence precision/recall/F1 explicitly use Top-5. `evidence_f1`
remains the pre-registered gate alias alongside `evidence_f1@5`. Other callers
retain legacy behavior unless opting into `evidence_k`. ANN@5-vs-Flat explicitly
compares the first five IDs. Retrieval-only metrics do not report unconfigured
classifier zeroes as accuracy. No verdict model was added.

## Frozen integrity preflight (no quality retry)

- Reuse only the original diagnostic `s100-r16-n4-t005` checkpoint from the
  2026-09-03 run; do not train or reselect among six pilots.
- Reuse the SHA-pinned public-v2 environment with PEFT 0.15.2.
- Run a real tiny-model checkpoint/toggle integration test, then two fixed
  queries and two fixed documents with the original encoder revision.
- CPU-only Slurm preflight: 4 CPU / 16 GiB / 15 min / 20 GiB temporary storage.
  This initial bound reuses the prior GPU preflight cold-load allowance, not
  an assertion that CPU runtime was measured. Only four short probes run;
  no corpus inference. No competing GPU allocation is requested.
- Use a new run destination; old input archives remain read-only. Do not load
  qrels, CLIMATE-FEVER frozen test, SciFact labels or a full-corpus index.
- Keep failure/no-effect results. No adapter promotion or quality retry follows
  automatically; further validation must be separately frozen and justified.

Entry point:

```sh
python scripts/probe_embedding_adapter.py --adapter-dir CHECKPOINT \
  --output NEW_REPORT.json --device cpu
```

Slurm wrapper: `hpc/adapter_integrity_preflight.sbatch`. Test scheduling with
`sbatch --test-only` first. Model caches stay on node-local temporary storage.

## Historical evidence boundaries

The 126-query public validation Recall@5 and Evidence F1@5 remain valid for the
base-only downstream run. Its old MRR@10/nDCG@10 and paired differences/CIs are
censored, not complete Top-10 metrics. Renaming nDCG@10 to nDCG@5 is incorrect
because its ideal denominator used ten. Historical timing retains its original
depth and stage scope; new numbers must not replace the old immutable artifacts.

Restricted 1,208,827-document / 154-query / 20-step LoRA remains offline-dev
evidence, not independent test or online A/B. This public loader defect alone
does not prove or disprove the integrity of that separate run. Pilot ties alone
cannot identify their exact cause. The consumed frozen test remains closed;
SciFact remains unopened.

## Remaining release gates

1. Real preflight artifact/accounting must prove weight equality and nonzero
   output delta. Until then this is tested implementation, not validated repair.
2. Same-query serial-request profiling of low-latency LambdaMART and bounded 4B
   widths must measure encoding, recall, ranking, end-to-end P50/P95, peak memory
   and compute cost. These will be offline benchmarks, not online SLA.
3. Full-ranking metrics and 5,000 paired bootstrap comparisons require new
   validation-only artifacts, without reopening frozen test.
4. Clean-checkout reproduction and strict type checks must pass. Initial local
   run: 96 tests passed; real Torch integration skipped because Torch is not
   installed locally. Full strict mypy exposed historical debt in 13 files;
   repository-wide type cleanliness is not yet achieved.

No current resume or shared career material was modified.

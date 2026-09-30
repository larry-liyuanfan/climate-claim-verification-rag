# Sentence-ID Agent v3 — bounded CPU repair package

Status: CPU implementation/fixtures/tokenizer smoke only. No v3 weights loaded,
model inference, GPU submission, SciFact dev run, held-out result, deployment or
resume update. V1/v2 code and scores remain unchanged. This addresses observed
v2 failure mechanisms; it is not yet an effective-Agent claim.

## Changes and scope

V2 generated too many copied statements, hit512 output tokens9 times, referenced
preview-only documents4 times and often copied an abstain example. V3 removes
all placeholder answer examples and uses an independently versioned contract:

- `answer`: SUPPORTS/REFUTES plus1–3 complete sentence IDs **actually shown in
  the current prompt**. The server renders original text from frozen sources;
  the model does not rewrite quotations or generate a rationale paragraph.
- `read`: choose up to5 already-retrieved source IDs to replace the next context.
  This is a real model-selectable context-loading action, not automatic promotion
  of previews into evidence. Repeated/no-op read is rejected. A sentence removed
  from current context is no longer citable. Budget-truncated sentences are not
  citable either; original indices remain unchanged.
- `rewrite`/`rerank`: bounded retrieval/order changes; immutable original claim,
  stable source IDs, actual-text hashes and candidate-universe checks persist.
- `abstain`: always available, terminal, with a reason category. No tool use is
  forced for presentation; grammar does not turn an abstention into success.

Sources are immutable sentence tuples with hashes calculated from actual text,
not trusted caller hash strings. Single per-run source aliases and combined
source:sentence IDs replace ambiguous duplicate display identifiers. New
retrieval cannot assign changed text to an old ID. Rendering an exact original
quote is provenance by construction, **not entailment, verdict accuracy or
generative answer quality**. A single verdict applies to selected evidence;
across-document mixed labels are not inferred from hidden gold or silently
collapsed. The original SciFact importer/scorer retains that limitation.

## Work controls and budgets

All4 routes share20 candidates, at most5 selected sources,5 model calls,
2 charged repair attempts,5 tools,8,192 input/512 output tokens per generation
and120s per task. Those are equal **caps**, not equal actual work. Record actual
model tokens, attempted/completed tools, failures, rerank pairs and elapsed time.
Context/preview payload-token counts are diagnostic subsets; tokenization across
boundaries means they must not be added to reconstruct the full-prompt count.
Unavailable output usage after a model exception is marked unknown, never
claimed to cost0 tokens. Late answers are rejected with cost retained.

| Route | Deterministic work before the first generation |
|---|---|
| Fixed retrieval | Original claim retrieval→top context |
| Fixed rerank | Original retrieval→4B rerank→top context |
| Deterministic extra | Original retrieval→fixed extra query→RRF→4B rerank→top context |
| Adaptive | Original retrieval; model may choose read/rewrite/rerank or a terminal action |

The extra query is the first deterministic decomposition that passes the **same
query contract** as adaptive rewrite; otherwise try original claim plus fixed
`scientific evidence`. If no query passes the shared length/loop/entity/number/
qualifier/lexical constraints, record a skipped extra query rather than bypass
the contract. The original top-ranked context is not deliberately replaced by
rank6–10 as an artificially weak control. RRF still may change context based on
the additional result. Main comparison remains adaptive versus fixed rerank.

The inherited whole-claim numeric/entity/negation/50%-lexical rewrite contract
can inhibit subclaim or counterevidence queries. It is a conservative current
restriction, **not proof of semantic equivalence or a security necessity**.
Relaxation would be a separate pre-registered development change, not an
unreported asymmetry between routes. V3 changes several mechanisms and caps
from v2; no single-factor causal improvement may be inferred.

## Token packing and grammar

Every packing step counts the real tokenizer/chat template over system text,
dynamic schema, claim, shown full sentences, previews, feedback and budget
metadata together. Complete sentences are retained in original order within
each selected source; overflow does not silently clip a sentence. Previews use
remaining capacity without displacing complete context. If mandatory prompt
overhead alone exceeds the cap, generation does not start. Final provider usage
must equal the counted input length. No empty grammar enum enables arbitrary IDs.

LMFE0.11.3 receives an inline `anyOf` with one enum-tagged object per currently
allowed action. Answer IDs are restricted to currently visible sentences; read
IDs to retrieved sources. Strict Pydantic and controller checks remain after
decoding, including duplicate-ID rejection. No `oneOf`/`$ref`/discriminator schema
is handed directly to this backend. `JsonSchemaParser` and prefix callback are
fresh each generation; only tokenizer vocabulary data is shared. No grammar-free
retry, model change, sampling change or new fine-tuning is introduced.

Only ordinary HF `prefix_allowed_tokens_fn`, greedy sampling and non-thinking
are used. Avoid LMFE's higher-level diagnostics wrapper, which depends on a
removed Transformers method. Its internal error fallback can log the decoded
prefix then emit EOS: this provider isolates root logging to private diagnostic
sinks and restores logging afterwards. A nonempty **attempted** grammar log is
a charged failure, including when no storage capacity remains. The provider-root
quota is cumulative across all attempts and earlier provider instances:128 files
and1MiB, with16KiB per grammar log and32KiB per raw response. No per-UUID quota
reset; older nested files count too. POSIX roots must be owner-only and new files
use0600. Symlinks are rejected. This is a **single-writer serial offline provider**,
not a concurrent web service or multi-process quota service.

Quota truncation retains full-content hash/byte counts and stored-prefix hash/
byte counts in a content-free receipt; it does not change a model decision. A
private I/O or formatting failure never falls back to stderr; it fails that
response closed with already-known usage retained. After a model exception,
unavailable output usage stays unknown. Torch2.1.2/HF4.51.3 integration still needs
allocated preflight.

## CPU evidence and reproduction

The [pinned Qwen tokenizer smoke](verified-runs/agent-v3-tokenizer-cpu-smoke-20260930.json)
checked all4 tokenizer-file SHAs at revision
`350135a4de9a3407be836fa238cccc1d61503a85`. Synthetic prompt cases stayed within
1,024/2,048/8,192 limits at996/2,014/5,782 tokens. The smallest case exposed no
complete sentence and kept answer unavailable. Example terminal/read actions
used7–42 tokens; this is not an exhaustive Unicode/natural-language rewrite
bound, and a rewrite can still hit512. Failures must remain in the ledger.
The fixture provider's abstention is scripted and has **zero model calls**.

```powershell
python -m pip install -e ".[test,dev,langchain,grammar]"
python -m pytest tests/test_agent_v3.py -q
# Tokenizer-only smoke additionally needs transformers4.51.3 and jinja2 3.1.6.
python scripts/smoke_agent_v3_tokenizer.py --tokenizer data/qwen3-4b-tokenizer-v3 --output data/new-v3-smoke.json
```

Local full suite before final payload/quota accounting:373 passed,2 optional
Torch/LightGBM skips. Final targeted CPU validation:31 passed,1 POSIX-only skip
on Windows; Linux CI covers that owner/symlink invariant. These v3 fixtures cover
source mutation, context replacement, preview rejection, repeated reads,
full-prompt overflow, deadline, unknown output usage, multiple calls/instances
sharing a quota, large error logs, private-write/format failures and known usage
retention after diagnostics fail. Ruff passed; mypy43 modules passed in
the validation environment without optional Transformers installed. Installing
Transformers for tokenizer smoke exposes existing dynamic-export typing errors
in old v1 modules; this package does not suppress those or edit old behavior.
CI includes the small pinned grammar extra so its character parser tests do not
silently skip. This does not stand in for real HF/Torch constrained generation.

## Remaining release requirements

Coordinator review of the exact CPU commit→isolated runtime preflight→separately
authorized small development pilot with one GPU allocation. No new job exists
from this package. Before SciFact dev release, additionally audit comparable
identities against the already-consumed Climate train/230validation/authored
tasks; train/dev SciFact grouping alone is insufficient. Gold remains a separate
scorer-only input, with parse/complete-matrix checks before original scoring.
Do not open old sealed tests or infer success from easier schema compliance.

Primary implementation references: [LMFE0.11.3 HF adapter](https://github.com/noamgat/lm-format-enforcer/blob/v0.11.3/lmformatenforcer/integrations/transformers.py),
[parser](https://github.com/noamgat/lm-format-enforcer/blob/v0.11.3/lmformatenforcer/jsonschemaparser.py),
[token fallback](https://github.com/noamgat/lm-format-enforcer/blob/v0.11.3/lmformatenforcer/tokenenforcer.py),
[Qwen3-4B model card](https://huggingface.co/Qwen/Qwen3-4B).

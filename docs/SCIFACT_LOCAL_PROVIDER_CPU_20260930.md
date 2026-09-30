# SciFact Local Qwen/LMFE provider — CPU implementation and smoke

This extends the [document-terminal contract](SCIFACT_DOCUMENT_TERMINAL_CPU_20260930.md)
with an actual provider implementation, but does **not** release SciFact dev/train
inference, a new model experiment, or a Slurm job. Frozen v3/e6d1ee1, job31601753,
its wrapper/guard, and the original-compatible scorer are unchanged.

## Narrow adapter and runtime boundaries

`LocalQwenSciFactProvider` inherits the v3 loader, model-file hash verification,
offline-only loading, tokenizer data initialization, private diagnostics and
cumulative quotas. It explicitly declares `terminal_protocol` and uses
`render_scifact_prompt` for both count and generation. Both tokenize the same
rendered string with `add_special_tokens=False`; generation uses greedy,
single-beam, nonthinking decoding. No API fallback, model switch or new download.

The frozen provider hard-codes its global-label renderer, so this subclass has a
narrow copy of `generate`. A source-parity regression permits only the renderer,
environment guard and effective-config receipt differences. Parser and prefix
callback are new for every call, with only tokenizer data reused. Root logger
redirection is preserved for this **serial offline** provider; not thread-safe
or a concurrent HTTP-service implementation.

LMFE0.11.3's `TokenEnforcer` replaces parser configuration using environment-backed
defaults. Frozen v3's child runner already removes `LMFE_*`. This reusable
adapter instead rejects any `LMFE_*` at initialization and generation; it does
not silently mutate the parent environment. The receipt includes the effective
parser alphabet hash/settings in addition to the inherited initial-config
identity. This distinction is diagnostic, not a change to the frozen v3 job.

Known input/output costs survive decode, grammar-log and diagnostic failures;
model exceptions retain input usage and explicitly mark output usage unknown.
Raw output/grammar text stays in owner-only private storage, under inherited
per-file and cumulative quotas. A full quota retains hashes/counters without
changing decisions; an I/O/grammar error fails closed. No raw-prefix stderr
fallback is introduced.

## Synthetic-only entry

`scripts/smoke_scifact_provider.py` has **no dataset argument**. Default mode uses
the existing four-file SHA-pinned Qwen3-4B tokenizer cache:

```text
python scripts/smoke_scifact_provider.py --tokenizer <cached-tokenizer> --output <new-json>
```

It tests the real tokenizer and LMFE **core** token-prefix filtering without
importing the Torch-dependent HF bridge or loading weights. Regular-token
construction mirrors the installed LMFE0.11.3 HF helper through its public
TokenEnforcerTokenizerData API. Three synthetic nested-document paths exercise
A/B/A visibility, different document labels and original sentence order. Valid
payloads must fit 512 output tokens; encode/call tokenization must agree and
complete prompts must fit8192. Cross-document and stale-document negatives keep
the same valid two-document structure and vary only the relevant identity.
Grammar error logs fail the check; a fallback EOS cannot masquerade as success.

A future **explicitly authorized allocated runtime** may opt in with
`--allow-real-model-generation`, local model/manifest and private-directory paths.
The CLI rejects model arguments without that flag, pins the existing Qwen3-4B
manifest SHA `d1dd9783afdf4e0fbd21eee824834d71b86982f5a5d5f6f371fe07f2f76f3cf6`,
and still permits synthetic fixtures only. Do not run model work on a login node.
This flag is not authorization from this CPU work package.

`synthetic_runtime_smoke` uses A/B/A dynamic anyOf cases followed by a one-token
truncation case. Each case records usage before parsing/checking. Tokenizer/count
failure before generation preserves earlier records and is not counted as a
provider generation attempt. Remaining independent cases continue once, with no
retry. Duplicate-document output is protocol noncompliance caught by the strict
parser, not automatically an HF or grammar backend failure. Forced synthetic
labels/shapes are **not autonomous decisions or scientific-quality evidence**.

## Validation

`python -m pytest tests/test_local_scifact_provider.py -q` runs 16 CPU/mocked-HF
fixtures: template parity, fresh parser/callback, EOS/truncation, multi-document
labels/order, stale/cross-document rejection, known/unknown failure cost, private
quotas/logs, inherited initialization, environment rejection, generation-source
parity, default no-model CLI, core token prefix, and count-failure cost retention.
No real HF model generation, official dev/train data or new Slurm job is used.

Cached-tokenizer smoke results will be recorded separately after this code is
frozen; CPU fixture pass is not a substitute for future allocated HF verification.

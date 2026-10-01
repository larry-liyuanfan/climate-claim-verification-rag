# Read-only FIT24 failure attribution

Implementation and synthetic validation only. This is **not** a new experiment,
an Agent improvement, a supervision target, or permission to read actual gold.
The execution source remains `723c6a8fc4f0956ba91497b278d5f8fe65dc9fee`, job
`31895661`; its tar, release, wrapper, inputs, budget and scorer are unchanged.
The posthoc source has an independent Git SHA and a new private output directory.

## Question and evidence chain

Reuse `coverage` / `document_opportunities` and `strict_whole_answer`, not oracle
packing or controller replay. OR alternatives are alternative complete rationales
within a document; **all** gold documents are required. A per-document hit is not
a complete claim. Preserve semantic `complete` and `first3_reachable` separately:
complete four-sentence-only rationales cannot pass the frozen strict first-three
scorer and are not attributed to model discrimination failure.

Bind original corpus document IDs and sentence indices to actual ordered citable
frames, physical prompt/response receipts, source hashes and alias mappings.
Compare fulltext of the frozen prefix candidates with actual initial visibility:

- Candidate pool incomplete under the existing output contract.
- Candidate pool complete but initial context incomplete.
- Initial context sufficient with an audited correct/wrong direct terminal or
  abstention. Initial tool decisions remain a separate category; a later wrong
  endpoint cannot retroactively make the first step a wrong direct answer.
- Complete but strict-first-three unreachable; unknown integrity; official NEI.

Candidate fulltext is not proof that a legal read could pack that evidence.
Unexecuted read/rewrite reachability remains unknown, and whole-corpus opportunity
is not measured. No `classify_opportunity`, oracle search, model inference or
training is performed.

For executed tools, link the audited proposal, exact executed arguments and source
state, next actual frame/feedback, next decision and final citations. Reuse the
existing transition validator and `tool_changes`; preserve missing frame indices.
Only an observed completed tool with incomplete→complete coverage gets a
subsequent correct/wrong/unresolved transition outcome. Strict-unreachable
coverage has its own outcome. A's model-selected events and B/C's frozen scripted
interventions remain separate; neither is labelled a causal effect.

## Read-only release and failure policy

`scripts/diagnose_scifact_natural_outcomes.py` requires a separately approved,
hash-bound coordinator release naming execution job/source, audit source and a
new `posthoc/scifact-natural-attribution-<audit SHA prefix>` directory. Run only
on an allocated Linux CPU node after Slurm is terminal, the child is reaped and
the existing cost-before-gold file is hash-bound to its exit receipt. No such
release or actual run is part of this implementation package.

The CLI verifies these gates before loading a tokenizer or gold. Only the frozen
24 selected TRAIN claims may be scored. Validation12, dev300, retired test,
old12 and utility8 are not new diagnostic inputs. No original output is changed.
All 24×3 slots remain in the denominator. Missing/illegal/truncated records or
unknown physical cost stay unresolved; an empty failed prediction is not correct
NEI. Reconstruct physical inventory from ledger reservations, not declared IDs
alone. Deduplicate shared calls by original physical ID. Global unknown cost
cannot be assigned away to preserve favourable endpoints.

Synthetic tests cover OR alternatives, partial multi-document coverage, loss of
a previously needed document, source/alias reorder and tampering, strict-first3
limits, actual coverage gain followed by an incorrect abstention, scripted frame
offsets, missing frames, failed NEI, hidden/duplicate calls, nonterminal Slurm,
unreaped workers and absent/changed cost receipts. These tests have no real
claims, remote reads, inference, new GPU submission or quality-result claims.

## Coordinator readback of the unchanged execution (not this posthoc output)

The coordinator reported Slurm `31895661` as `COMPLETED 0:0`, elapsed 666 s,
with 72/72 valid terminal slots and no unknown cost. A's 24 initial decisions
were all answers and strict correctness was 0/24. B's scripted read scored
7/24; C's scripted rerank scored 3/24. **Every strict success was an official NEI
abstention**; ten program-teacher candidates are therefore not ten recovered
positive-evidence examples or evidence of autonomous tool use. B positive evidence
F1 was zero; C sentence-label F1 0.0717489 versus A 0.0682303 does not establish
complete-rationale improvement. No SFT or expanded run is authorized by this.

Reported physical cost is 72 generator calls, 253,288 input / 8,009 output tokens,
and 24 rerank requests / 480 pairs. Original aggregate hashes supplied by the
coordinator are quality `babc1a70c316f91f60f2679864335f458eb1c7d81510993447aef0fe2e41d24b`
and cost `4ab434c72afeda0a105c0e78e5feab6161695125a4aff3d3c254f3a1da40a7aa`.
This package has not independently opened the real output or its gold. Actual
failure attribution awaits a separate release; the implementation's synthetic
tests are not market, held-out-test, or Agent outcome measurements.

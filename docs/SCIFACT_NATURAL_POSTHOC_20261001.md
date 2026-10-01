# Read-only FIT24 failure attribution

Implementation and synthetic validation were followed by a separately released
read-only CPU audit. This is **not** a new model experiment, an Agent improvement,
a supervision target, or permission to read additional gold.
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
the existing cost-before-gold file is hash-bound to its exit receipt. That distinct
release and completed CPU run are recorded below; the original execution remains
unchanged.

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
These were the coordinator's initial aggregate observations. The separately
released audit below subsequently checked the physical frames and fixed TRAIN24
gold on Spartan; no per-claim output or gold was exported. Synthetic tests are
not market, held-out-test, or Agent outcome measurements.

## Completed read-only audit and implementation decision

[Compact aggregate](verified-runs/scifact-natural-posthoc-closeout-31904484.json)
comes from CPU job `31904484`, `COMPLETED 0:0`, elapsed 18 s, MaxRSS 263060 K,
CPU time 6.955 s. Resource allocation was 2 CPU / 4 GiB / 15 min / zero GPU.
There were zero new model calls and zero training updates. The exact tuple was:

- Audit source: `4b27d6aab6bce9f0492058db039cd9dc16b3e436`.
- Tar: `64b65ff7ff4df6d1a6d8df03b80f644d556b5db5118ad7e26f8ab83afe6a04fa`.
- CPU wrapper: `0df533e70e124d8e359dab10a63980a96db9b2e8f0aab00f825c958a8b4a2e8b`.
- Authorized release: `8539fba2ec91f74a2a03bd1ea0684e8621e4baf67a949b57d0df52ca11ca13ee`.
- Compact SHA-256: `7e8aa612431542fcbd330397b3dc7b97aea7dec35e66ba07a25b707fab499192`.

All 72 planned slots remain resolved. Of 24 claims, 15 are evidence-bearing and
9 are official NEI. **14/15 positives already had semantically complete and
strict-first-three-reachable evidence in the actual initial frame. All 14 legal
direct answers were nevertheless wrong.** The remaining positive had incomplete
multi-document coverage in the candidate pool, not merely a context-packing miss.
No case was classified as complete candidates but incomplete initial visibility.

A produced zero model-selected tool events. B/C each executed 24 scripted tools,
with 15 observed positive-coverage cases and 9 separately treated NEI cases.
Neither branch made incomplete evidence complete: zero semantic and zero
first-three coverage transitions. All three arms remained 0/15 strictly correct
on positives; NEI correctness was A 0/9, B 7/9, C 3/9. Thus scripted abstention
recovery is not positive-evidence tool utility and cannot supply that claimed
training signal. Other unexecuted reads/rewrites remain unknown.

**Decision:** prioritize the terminal grounding objective—joint original-document
selection, complete rationale sentences, labels and evidence-conditioned
abstention—while holding retrieved/visible evidence fixed. Do not expand ANN,
candidate width, or autonomous-tool training in response to these results. A
future authorized implementation comparison must improve complete positive
grounding without buying the gain through unsupported answers or blanket
abstention; otherwise reject it. This is a proposed next change, not authorization
to train or a validated improvement. These 24 exposed TRAIN cases do not establish
out-of-sample performance or a causal tool effect.

Raw attribution and per-claim frames remain in the new private Spartan posthoc
directory. The compact includes original attribution/receipt hashes. Original
`quality.json` and `cost-before-gold.json` hashes were rechecked unchanged against
the values above; no original artifact, scorer, model or execution source changed.

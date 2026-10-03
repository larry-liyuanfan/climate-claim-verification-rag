# Stage-aware bounded-decoder confirmation

## Question and fixed comparison

Does enforcing the already-defined document uniqueness and 20-sentence budget
at generation time preserve correct local evidence in the actual final answer?
This is an implementation comparison against job **31918065**, not an
infrastructure retry, another training experiment or an independent test.

The new job **31930176** uses source
`49f168a2517089e43bc91208f6690b5b81f933bc`. Both runs use the same previously
exposed SciFact TRAIN24 selection, 15 positives/9 NEI, immutable observations,
base model, prompts, schemas, scorer and per-episode budgets. Only controller
decoding changes: `plan`/`terminal` use the existing bounded prefix; document
`verify` retains ordinary LMFE. No completed response is rewritten or repaired.

The complete execution chain was reviewed before submission. Source validation
was 1280 passed plus one Windows POSIX-specific skip, 117 clean-archive seam
tests, and Linux CI **36867234612: 1281 passed**. The actual runtime consumer,
asset hashes, normal QoS, quota/duplicate checks and `sbatch --test-only` passed
before the single actual submission. A durable attempt lock prevents retries.

## Why this comparison was warranted

[CPU diagnosis 31926051](verified-runs/scifact-document-gold-decomposition-31926051.json)
replayed existing results without model calls. Of 16 verified annotated gold
documents, ten had correct labels and first-three rationale coverage. Seven
were not retained in the terminal prediction; six involved structural rejection.
Three positive verifier projections matched the strict annotation definition,
but **none is a generated terminal success**. Full selection, first-three
coverage and whole-answer correctness must remain separate.

The comparison retains all 48 slots and physical costs, including failures.
Report fixed/adaptive positives, NEI, structural rejection and paired case
transitions. A direct correct answer, a projected verifier answer or an increased
tool count alone cannot establish autonomous feedback benefit. That claim needs
an actual model proposal → execution → observed feedback → correct subsequent
decision and a same-verifier fixed-route quality/cost comparison.

## Execution identity

- Source archive: `5dc1a5dffa533efbb1e7a8ce5204c100db1da4431d2f35e61f133a0a4147d19b`
- Actual wrapper: `f13f4ffb93b6dc1491d6fc33ce49d54b34eb1170e4b80b00d30acc965ad3e955`
- Authorized release: `313544a52bf363bb2dca399b91e0aa5b0bb646a024e15ceb0257325615e5d602`
- Pre-submit check: `721b4a3168e13db762065cd99070bb9b015206c02fbaf704337ed537dad0615e`
- Resource ceiling: one A100, 8 CPUs, 32 GiB host RAM, 30 GiB scratch, two hours;
  normal QoS/Nice0/no-requeue, maximum 240 physical generations.

## Measured closeout

[Job 31930176](verified-runs/scifact-document-bounded-closeout-31930176.json)
completed 2026-10-01 23:37:49–23:44:20 +10, exit 0:0, 391 seconds, MaxRSS
9,271,396 KiB. The child was reaped before scoring, all 48 slots are present,
and local/remote complete, cost, quality and input-frame hashes agree.

| Diagnostic on the same TRAIN24 | Fixed old → new | Adaptive old → new |
|---|---|---|
| Structurally valid terminals | 15 → 24 | 8 → 24 |
| Strict positive whole answers /15 | 0 → 0 | 0 → 0 |
| Correct NEI /9 | 3 → 3 | 0 → 0 |
| Correct rationalized documents /17 | 4 → 10 | 2 → 6 |
| Predicted documents | 42 → 80 | 31 → 88 |
| Rationalized-document micro-F1 | 0.1356 → 0.2062 | 0.0833 → 0.1143 |
| Sentence-label micro-F1 | 0.0606 → 0.1030 | 0.0526 → 0.0700 |

The legality fix works: all 25 former structural rejections now produce accepted
terminals. It also retains more correctly rationalized documents, **but complete
positive answers remain zero**. More accepted evidence includes substantial
additional predicted evidence; partial recall must not be renamed full grounding.

Actual cost is 120 fixed calls (254,677 input/6,288 output tokens; 202.680 s
ledger elapsed) versus 24 adaptive calls (86,066/3,743 tokens; 115.006 s).
All 144 finished receipts were independently summed; no missing, unknown or
unassigned calls. Backend-only elapsed is 201.698/114.675 s. This is one serial
offline run, not a latency SLA, online saving or an API currency charge.

## Paired wire and feedback audit

All 144 raw responses are complete and parser-valid. Across every paired call,
slot/protocol, observation, schema, prompt hash, token-ID hash, input-token count,
maximum output tokens and base-model state match. Initial frames are byte-identical.
The actual diagnostics identify 24 bounded plan calls, 24 bounded terminal calls
and 96 ordinary verifier calls, with the expected stage/version tags. Effective
parser configuration is unchanged; bounded configuration identity now includes
the actual tokenizer alphabet.

All 96 fixed verifier raw responses are byte-identical: 20 SUPPORTS, 15 REFUTES,
61 INSUFFICIENT. All 96 feedback rows reached the 24 fixed terminal requests.
Exactly the 25 previously invalid terminal raws changed; all 23 previously valid
terminal raws are byte-identical. Fixed accepted 21 answers and three abstentions;
adaptive accepted 24 answers. Adaptive had `verify` available on all 24 requests,
but proposed/executed it **zero times** and consumed no verification feedback.

These representative cases distinguish retained local evidence from full answers:

| TRAIN case / fixed call | Observed terminal behavior |
|---|---|
| 570 / g82 | Unchanged: retains c0 SUPPORTS `[2,3]`, adds c3 despite INSUFFICIENT feedback and unverified c4. |
| 349 / g100 | Now valid at 20 citations: retains c0 REFUTES `[7,8,9,11]`, adds labels for three documents judged INSUFFICIENT. |
| 565 / g142 | Now valid at 20 citations: retains c0 SUPPORTS `[3]`, adds REFUTES labels for three INSUFFICIENT documents and unverified c4. |
| 853 / g16 | Now valid: uses the same c0:8 citation but changes feedback REFUTES to terminal SUPPORTS; also includes INSUFFICIENT and unverified documents. |
| 403 / g04 | Fixed scored-correct abstention is unchanged; adaptive g05 instead gives a valid 20-citation answer and still fails NEI scoring. |

For 570/349/565, the prior CPU audit established a strict-matching positive
verifier projection. Those c0 judgments survive exactly, but the generated
terminal expands beyond that projection. Projection correctness is reused from
the old CPU audit, not inferred from model feedback or a new gold read. All three
old correct NEI cases (403/304/281) remain unchanged. Compact case-result hashes
are in the linked closeout JSON; raw responses remain outside Git.

Disagreeing with a verifier alone does not establish a semantic error. Unverified
documents were legitimately available in the original evidence; INSUFFICIENT
does not prove the opposite label or claim-level NEI. The observed failure is
that legal, expanded final predictions still fail the unchanged whole-answer
scoring, even when a correct local projection was available.

## Decision

Keep the validated generation-time constraints. The remaining primary issue is
**semantic evidence-selection and integration policy**, not JSON/cardinality
legality. Original r1/r2 scores remain unchanged. No additional training,
protected split, resume gain or retry is authorized by this result.

# Component continuation terminal closeout — 31743877

**Completed, not an Agent-effect result.** On 2026-10-01 **02:32:29–02:34:39
Australia/Sydney**, the single authorized A100 job completed `0:0` in 130 seconds.
It executed **36 new invocations**, retaining one paid r2 synthetic response for
**37 cumulative**. All 33 formal outputs were contract-valid; no provider/schema
failures or unknown-cost attempts occurred. Correctness is reported separately.

## Frozen identity and physical audit

- Execution source: `a104115ecc370223617ee3238fb7e5b24af526ea`; archive
  `ad9ebf177ca03cf5bf8e44c38ee4fa3aa2e28c92af0b0178672bdc8d160be0e1`.
- [Original compact](verified-runs/scifact-component-continuation-compact-31743877.json),
  SHA `96bbd5a9f754023894c5f19ef340fb8da4737533fce937639bcd9472b583f7dc`.
- [Read-only physical closeout](verified-runs/scifact-component-continuation-physical-31743877.json),
  SHA `cb08c83559bd020276700dd7bcd15c0cfbc64e0c7d110745d19c6ad43c0123da`.
  It reconciles four synthetic and 33 formal records, original wire/schema/decoder,
  private raw/grammar receipts, tokens, strict parsing and exact cost partitions.
  It verifies 36 new reservations and no new `preflight-00` directory; the old
  r2 hash chain remains unchanged. The separately written private scores match
  every reconciled inference record and frozen target identity.
- Child exit/reaping proof precedes private score/compact creation; worker and
  finished receipts report no scoring-target loading. The pinned operator reads
  scoring targets only after actual child exit. The closeout did **not** rerun
  the scorer or invoke any model, and only read the existing consumed TRAIN data.
- The three new synthetic outputs match their expected answers. The carried
  synthetic abstention remains semantically **false**, though technically valid.
  It is not a new live-provider preflight result or a retroactive r2 success.

Raw responses, gold, per-claim scores and the full physical-file digest manifest
stay on Spartan. Only hashes, aggregate counts and resource metadata are public.
The audit script is `scripts/audit_component_continuation_closeout.py` (SHA
`9fed774315f2d34925ad5148b3a89b5388a990b1261de2108370de820222079c`), run against the
frozen a104115 modules. Ruff/strict mypy and the actual physical audit pass.

## What the component evidence says

| Component and condition | Result | Interpretation boundary |
|---|---|---|
| Screening: 12 claims, frozen full candidate texts | Exact selected-set match **0/12**; **54/60** candidate occurrences selected; all candidates selected in **9/12** cases | Pool gold **6/6** selected with zero omissions, but this alone does not establish useful discrimination |
| Candidate availability, before screening | **9** annotated gold occurrences total, **6** in the fixed pool, **3** missing | Missing pool evidence is not a screening-model omission or automatically a retriever causal fault |
| Relation: one complete gold document for each of nine evidence claims | **5/9** correct | Four annotation-relative relation errors remain even under this controlled full-document condition |
| Relation: three official cited-context NEI controls | **3/3** correct | These are not a newly human-judged set of arbitrary negative document pairs; pooled **8/12** is not natural-retrieval verdict accuracy |
| Rationale: complete gold document **and explicitly correct relation** | First3 **7/9**, any complete alternative **8/9**, exact alternative set **1/9** | Oracle-conditioned localization, not autonomous relation-plus-citation performance |

The 48 selected unannotated document occurrences are **unjudged**, not 48 proven
semantic errors. The three official NEI claims selected 15 candidate occurrences
in screening; this is an annotation-relative diagnostic, not proof each selected
document is factually irrelevant. Six screening inputs have no annotated gold in
their pool (three missing-gold evidence cases plus three official NEI claims).
Counts are summed over claim-document occurrences, not necessarily unique docs.

Gold-document relation confusion (target rows, prediction columns):

| Target | SUPPORT | CONTRADICT | NOT_ENOUGH_INFO |
|---|---:|---:|---:|
| SUPPORT (7) | 4 | 0 | 3 |
| CONTRADICT (2) | 1 | 1 | 0 |

Thus three support cases were judged insufficient, and one contradiction was
judged support. These are diagnostic signals for grounding supervision, not
evidence that LoRA, a different prompt, or an action policy will repair them.

For oracle rationale, one case lacks every complete alternative; another covers
an alternative only beyond its first three output positions. Seven of the eight
complete cases are not exact alternative sets. There are 13 selected sentences
outside the union of annotated alternatives; they are not automatically false
sentences. Minimum excess over an already-complete alternative sums to 15 over
eight cases. This is neither an error rate nor interchangeable with the 13 count.

Across the same nine gold-document cases, paired relation/first3 counts are:
wrong/wrong **1**, wrong/right **3**, right/wrong **1**, right/right **4**. The
rationale side still received an oracle relation, so this is not joint deployed
accuracy. The scorer's prioritized localization tags are hypotheses rather than
a causal or exhaustive partition of underlying failures.

## Cost and resources, including earlier failures

| Scope | Calls | Input / output tokens | Sum of slot elapsed time |
|---|---:|---:|---:|
| Carried r2 invocation | 1 | 277 / 15 | 1.089 s |
| New continuation | 36 | 40,420 / 969 | 30.812 s |
| Cumulative logical matrix | 37 | 40,697 / 984 | 31.901 s |

New job: **130 allocated A100-job seconds**, **115.452 batch CPU seconds**,
batch MaxRSS **10,305,912 KiB**, observed Torch peak allocated GPU memory
**8,802,492,928 bytes**, model load **14.895 s**, operator wall time **123.516 s**.
This is an offline serial diagnostic, not online latency or an SLA.

The versioned three-job chain retains v1's 81 and r2's 96 allocated GPU-job
seconds: **307 cumulative allocated A100-job seconds** including this run.
Batch CPU time totals **269.762 s** including earlier 154.310 s. These are not
active GPU kernel time, money spent, or the total cost of all historical project
experiments. Memory peaks must not be summed as concurrent usage.

## Decision boundary

The execution/format failure is closed; model quality is not. The observed
full-document relation errors and broad selections support investigating **one
grounding-adaptation candidate**, after checking annotation/negative provenance.
They do not justify action-SFT or claim an Agent benefit. No fresh dev/test data,
training, further inference, deployment or resume change occurred in this
closeout. Stage B remains a separate bounded CPU preparation/release decision;
its evaluation must remain non-oracle and use identical base/adapter inputs.

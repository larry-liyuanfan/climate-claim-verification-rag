# Prospective claim-group-mean optimizer contract

Status: **synthetic CPU validation only; no real training is released**. The
historical v2 module/config, /48 optimizer helper and same-logits SGD tests are
unchanged. This separate module does not load data/models, run an epoch or submit
a job. Shared-corpus/data protocol and usable action supervision remain unresolved.

## Mathematical contract and historical boundary

For each decision record, compute causal cross-entropy averaged over its valid
assistant tokens, including EOS; prompt tokens are masked. Sum weighted record
losses within each claim, with the existing trajectory/alternative/state weights
summing to exactly one. With `m` explicitly declared complete claims in a group:

```text
claim_loss[q] = sum(record_weight[r] * assistant_token_mean_CE[r], r in claim q)
group_mean   = sum(claim_loss[q], q in group) / m
epoch_metric_contribution = group_mean * m / 48
```

The new backward coefficient is `record_weight / m`. A claim with more records
or longer assistant text does not receive extra total mass. Missing records do
not shrink the declared group or trigger weight renormalization. The separate
`/48` quantity is a logged epoch contribution, not the optimizer denominator;
aggregating it across updates is not a fixed-parameter end-of-epoch evaluation.

For a four-claim group the historical v2 helper's gradient before clipping is
one twelfth of the new group-mean gradient at identical parameters/data. This
is a **constant loss-scale distinction, not a missing-record-weight bug**.
It does not imply AdamW learns twelve times more slowly: epsilon, clipping and
optimizer state affect the update. No learning-rate or epoch multiplier is
introduced. This does not diagnose the cause of earlier model failures. The
v2 helper itself had only CPU tests/prepared supervision; it must not be described
as the objective used by the earlier v1 checkpoint job.

## API and frozen prospective settings

`scifact_claim_group_mean.py` adds:

- `make_claim_group_mean_adamw(model)` — explicit optimizer construction only.
- `optimizer_step_claim_group_mean(..., group_claim_ids)` — validates the exact
  declared 1–4 unique claims, complete per-claim record weights, every token/hash
  and optimizer settings/parameter ownership **before** zeroing gradients or
  executing a forward. Accumulates all records, clips global L2 norm once, and
  performs exactly one step. Nonfinite gradients reject the step.
- A separately versioned contract and its hash; historical record metadata
  `epoch_normalizer=48` remains unchanged and is not silently repurposed.

Prospective full-run contract: original hash-bound 48 FIT claim IDs, one seeded
shuffle (`20261001`), complete groups of four, one epoch and 12 optimizer steps.
AdamW uses learning rate **1e-4**, weight decay **0**, betas **(0.9, 0.999)**,
epsilon **1e-8**, global L2 clip **1**; amsgrad/maximize/foreach/fused/capturable/
differentiable are false. Short groups of one and three are arithmetic fixtures,
not permission to change the 48-claim run or its group boundaries. A future
authorized trainer must bind and enforce the full-data/shuffle/epoch contract;
no such trainer is built here.

Contract SHA:
`ace07142d9d14e6d7dd58e8dba1a68bfa3eba94fe44a7753491992a82509e6c6`.
Module SHA:
`35868d3c85e4a305fdf9b6e3cbb518ece4a485bcbe43d6ef79228d5232166d29`.
Test source SHA:
`82db81098eab50ef9c3e5316cfdf1fffd96bb0176b29791f6765ab93c7a83005`.

## Real optimizer, synthetic evidence only

The small Torch model has distinct learned input-token and position logits, not
one identical logit vector at all positions. Synthetic claims have 1/2/3/4
records and differing assistant lengths, including two-state trajectories. The
matrix covers group sizes 1/3/4 with clearly unclipped/clipped gradients. A second
synthetic group changes claim IDs, targets and lengths while retaining AdamW
moments. The independent reference uses log-softmax/gather for each state's token
mean and a direct global L2 clipping calculation, rather than the helper CE/clip.

Assertions compare every parameter's **pre-clip and post-clip gradients**, global
norm, two real AdamW parameter updates, first/second moments and step counters.
Checking updates alone would be insufficient because AdamW can be approximately
invariant to a constant gradient scale. An identical-state duplication with
split weights preserves loss, gradients and updates. Failure fixtures cover
missing mass, wrong/duplicate/too many IDs, last-record token tampering and drift
in optimizer learning rate, decay or epsilon. Validation failures preserve
parameters, existing gradients and optimizer state; injected NaN gradients
cannot step.

Validation environment: Python **3.12.14**, Torch **2.7.1+cpu**, Windows. Final
targeted result: **23 passed, 31 deselected, 2.33 seconds**: 17 new tests and six
unchanged historical v2 SGD tests. Ruff and strict mypy passed for the two new
files. Independent static review found no blocking defect. No Linux/GPU parity
claim is made; no large suite, runtime manifest or source archive was rebuilt.

```powershell
.\.venv-validation\Scripts\python.exe -m pytest tests/test_scifact_claim_group_mean.py tests/test_scifact_state_supervision.py -k 'claim_group_mean or actual_backward_fixed_48_denominator' -q
.\.venv\Scripts\python.exe -m ruff check src/climate_rag/scifact_claim_group_mean.py tests/test_scifact_claim_group_mean.py
.\.venv\Scripts\python.exe -m mypy src/climate_rag/scifact_claim_group_mean.py tests/test_scifact_claim_group_mean.py
```

This proves the bounded optimizer interface matches its declared arithmetic on
synthetic data. It is not a trained adapter, new retrieval/Agent score, independent
test, resource benchmark or resume quality gain. No real corpus/claim annotations,
reserved evaluation, training job, current resume or other project were touched.

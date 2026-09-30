# SciFact paired component statistics — CPU preparation only

This package releases a pure statistical function and synthetic tests, not an
inference runner, another data split, or permission to evaluate SciFact dev.
It does not modify the frozen original-compatible scorer or Climate v3 job.

## Fixed estimand and resampling

Primary: **adaptive minus fixed-rerank, abstract-rationalized micro F1**.
Attribution control: adaptive minus deterministic-extra on that same metric.
The other three official metrics are prespecified secondary results, not
posthoc alternatives for choosing a positive claim. All four routes remain
mandatory, including fixed retrieval.

`evaluate_matrix` accepts the original gold/corpus, exact route × claim prediction
matrix, original preparation's `claim_component` mapping, and charged run records.
Data/manifest hashes and the original grouping must be checked by a future
release runner; this CPU function does not load, refreeze or reselect data.
It records a canonical target-membership hash, not private claim IDs in the report.

For each claim/route, call the frozen scorer to obtain each metric's
`correct/predicted/relevant` sufficient statistics. Sum within original claim
components; also check that those sums equal the full-matrix scorer. Construct
the bootstrap universe from **nonempty target/dev groups only**. Global mapping
entries belonging only to train never become zero-size sampling units.

Draw G groups with replacement, **5,000 replicates, NumPy PCG64, seed20260930**.
Sort claims numerically and groups lexicographically before sampling. Record
NumPy version, little-endian integer multiplicity-matrix SHA, target membership
SHA, group sizes, duplicate-draw replicate count and weighted-claim-count range.
Every route uses the exact same integer multiplicities. Duplicated groups are
weights, not duplicate claim objects sent to the scorer.

Each replicate sums the weighted sufficient counts **before** computing
`2 × correct / (predicted + relevant)`. Zero denominators follow the original
scorer (0). Do not average claim F1 or group F1. Subtract the two recomputed route
micro F1 values, then use 2.5%/97.5% quantiles with NumPy `method="linear"` for a
95% paired percentile interval. No bootstrap sign fraction is called a p-value.
With G=1, return the point difference and a null CI with an explicit reason;
empty/duplicate gold or incomplete/duplicate/unknown prediction slots fail closed.

## Failure and cost accounting

Explicit model abstentions and recorded failures remain empty-evidence slots
with distinct termination reasons. Missing files/rows are not synthesized into
abstentions. Accepted nonempty document predictions, NEI gold and mixed-label
multi-document gold retain the original scorer semantics; there is no global
claim verdict accuracy. Prediction bounds and original gold are validated.

Costs cover the **whole question including repairs and failures**, not only the
successful final call. Each record supplies elapsed milliseconds, known input/
output token lower bounds and unknown-usage-attempt count. Validate finite,
nonnegative time and strict integer token counters. A future runtime ledger must
prove completeness; a cost-scope string is a contract, not independent evidence.

For bootstrap cost means, sum weighted group costs and divide by the replicate's
weighted **claim count**, which varies with unequal group sizes. Load and runtime
preflight costs are separate explicit fields (null if not supplied), not silently
amortized into zero-cost requests. If either compared route has unknown usage,
report known per-route lower bounds/unknown counts and **no exact token delta or
token-saving interval**. A difference between two lower bounds is not a bound on
their true difference. All-known token and mean-latency differences have paired
mean intervals with the same weights.

P50/P95 are point quantiles of whole-question elapsed times. **No quantile CI is
computed** in this package, and a P95 of paired per-question differences is never
used. A future quantile CI would need to recompute each route's quantile per
weighted replicate and then subtract. NEI diagnostic CI is also not computed;
the original scorer returns null NEI rate when its denominator is absent.

## CPU validation and reproduction

`python -m pytest tests/test_scifact_statistics.py -q` uses synthetic matrices
only. Coverage includes unequal groups and repeated draws; pooled micro versus
macro differences; identical routes giving zero differences; order-independent
reproduction; explicit failures, NEI and multi-document opposite labels; unknown
cost; illegal times/counters; missing/duplicate/corrupt slots; single-group and
absent-NEI conditions. No real dev model predictions or gold files are read.

All reported intervals remain conditional on the chosen corpus, grouping and
observed evaluation sample. This implementation cannot certify independent
claims or lack of foundation-model pretraining exposure.

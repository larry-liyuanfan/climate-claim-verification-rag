# Decision log

## D1 — Preserve the lexical baseline

The tokenizer lowercases text, normalizes `CO2`/`carbon dioxide`, and retains fact-changing negations. This keeps the BM25 baseline comparable in intent to the Group 045 notebook.

## D2 — Separate smoke and learned encoders

The hash encoder makes tests deterministic and is labeled `hash-baseline-*`. Qwen3/BGE claims require the learned adapter and recorded model name.

## D3 — One mapping for FlatIP, HNSW, and IVF-PQ

FlatIP is the exact normalized-inner-product reference. HNSW trades graph memory for approximate speed; IVF-PQ adds training and compression. A reusable embedding cache avoids repeated encoding, while its ID hash prevents row misalignment.

## D4 — RRF before learned fusion

BM25 and dense scores are not naturally calibrated. RRF is the parameter-light baseline. Learned fusion adds score/rank plus token, number, and year-consistency features.

## D5 — Never rename a fallback as LambdaMART

When LightGBM is present, `auto` trains `LGBMRanker(objective="lambdarank")`. Otherwise it trains and persists a deterministic linear pairwise logistic ranker named `linear_pairwise_ranknet_fallback`.

## D6 — Keep classification decoupled

The historical LoRA classifier/checkpoint is not distributed. Evaluation accepts predictions and calibration accepts logits. Retrieval serving returns no label until a real classifier is configured.

## D7 — Require paired evidence

Improvements are compared per claim on the same split. Paired bootstrap reports difference and interval. Target gains remain aspirations until a restricted-data run creates manifests and outputs.

## D8 — Deploy the measured winner, not the most complex stage

The fixed-dev run selects BM25+dense RRF: it improved Recall@5 and Evidence F1 over BM25 with paired intervals excluding zero. Qwen3-Reranker-0.6B was then evaluated on all 7,700 RRF Top-50 pairs; its mean Recall@5/F1 was lower than RRF and the paired intervals crossed zero, while per-query P50/P95 was 4.88/5.88 s. It therefore adds cost without a defensible gain and is not enabled by default. The trained LambdaMART and deterministic reranker regressed sharply, so they also remain documented negative experiments. IVF-PQ is rejected despite its speed and compression because its Recall@5 versus Flat was only 0.3688; HNSW is the retained ANN default.


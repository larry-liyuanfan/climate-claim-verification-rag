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

The fixed-dev run first selected BM25+dense RRF over BM25. A pure Qwen3-Reranker-0.6B replacement then regressed because it discarded the strong RRF order. Weighted rank fusion corrected that architectural error. The next size gate compared 4B on the identical split and resource-tested it with BF16 on a 20 GB MIG slice. A balanced RRF/4B fusion improved Recall@5, MRR@10, nDCG@10 and Evidence F1 with paired intervals above zero. An 8B eight-claim pilot then tied 4B F1/Recall, slightly reduced MRR and increased P95 by 60.8%, so the full 8B run was stopped at the gate. The 4B model remains the offline quality profile, while HNSW+RRF remains the latency default because even the 4B run adds P50/P95 `4.20/4.82 s` per query. Model size and fusion weights were selected on the same dev split, so the result is not described as independent test generalisation. The trained LambdaMART and deterministic reranker still regress sharply. IVF-PQ remains rejected because Recall@5 versus Flat was only `0.3688`; HNSW remains the ANN default.


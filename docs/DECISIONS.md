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

The fixed-dev run first selected BM25+dense RRF over BM25. A separate dense-encoder size gate then compared 0.6B and 4B at the same 1,024-dimensional output on an evidence-preserving sample: 4B reduced Recall@5, tied F1/MRR, ran about 7.1× slower and used about 5.5× the peak Torch GPU memory, so the full 4B index rebuild was stopped. A pure Qwen3-Reranker-0.6B replacement also regressed because it discarded the strong RRF order. Weighted rank fusion corrected that architectural error. The reranker size gate compared 4B on the identical split and resource-tested it with BF16 on a 20 GB MIG slice. A balanced RRF/4B fusion improved Recall@5, MRR@10, nDCG@10 and Evidence F1 with paired intervals above zero. An 8B eight-claim pilot then tied 4B F1/Recall, slightly reduced MRR and increased P95 by 60.8%, so the full 8B run was stopped at the gate. The 4B reranker remains the offline quality profile, while the 0.6B encoder with HNSW+RRF remains the latency default. Model size and fusion weights were selected on the same dev split, so the result is not described as independent test generalisation. The trained LambdaMART and deterministic reranker still regress sharply. IVF-PQ remains rejected because Recall@5 versus Flat was only `0.3688`; HNSW remains the ANN default.

The next dense-encoder gate was task adaptation, not another parameter-scale
guess. Mined train-claim negatives were converted to Qwen3-Embedding InfoNCE
rows with a claim-grouped validation split and false-negative controls. A
20-step LoRA run and runtime injection preflight completed. The 126-claim
evidence-preserving sampled screen improved Recall@5, MRR and nDCG with positive
paired intervals, but Evidence F1 crossed zero, so no sampled-only promotion
was made. Full-corpus job `29465819` then evaluated all 154 official-dev claims
and 1,208,827 documents. Recall@5 improved `0.2793→0.2970` with a paired 95%
interval `0.0014–0.0350`; MRR, nDCG and Evidence F1 also improved with positive
intervals. The adapter therefore passes the pre-registered offline dev gate and
becomes the preferred dense retrieval candidate. The base 0.6B encoder remains
the fallback because this is dev-set model selection, not an independent test
or online A/B result. The failed first full run also changed artifact policy:
rebuildable multi-gigabyte adapted indexes are opt-in, while metrics, manifests
and hashes are mandatory.


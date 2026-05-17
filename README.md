# Climate Claim Verification RAG

A portfolio-safe climate fact-checking project that combines evidence retrieval, semantic reranking, and sequence classification for climate-related claims.

Given a claim, the system retrieves relevant evidence from a large evidence corpus and predicts one of four labels: `SUPPORTS`, `REFUTES`, `NOT_ENOUGH_INFO`, or `DISPUTED`. The project is presented as a sanitized public summary: restricted benchmark files, raw evidence corpora, notebooks, and private evaluation artifacts are not redistributed.

## Project Highlights

- Built an end-to-end claim verification pipeline covering candidate retrieval, semantic reranking, sequence classification, prediction formatting, and official metric evaluation.
- Implemented a two-stage retrieval system: BM25 candidate generation followed by BGE semantic reranking to select top-5 evidence passages.
- Fine-tuned a Qwen3.5-4B sequence classifier with LoRA on claim-evidence inputs under Colab-style resource constraints.
- Compared retrieval candidate sizes, reranking models, and classifier training strategies to diagnose the retrieval-classification bottleneck.
- Separated development-set diagnostics from public evaluation results to avoid mixing experimental numbers.

## Task Definition

For each claim `c`, the system outputs:

- a ranked evidence set `E = {e1, e2, ..., ek}`;
- a claim label from `SUPPORTS`, `REFUTES`, `NOT_ENOUGH_INFO`, `DISPUTED`.

The official score combines:

| Metric | Meaning |
|---|---|
| Evidence Retrieval F-score | Whether predicted evidence IDs match the gold evidence IDs |
| Claim Classification Accuracy | Whether the final label is correct |
| Harmonic Mean | Balanced score between retrieval and classification |

## System Architecture

```text
Claim
  -> BM25 candidate retrieval, top-N evidence pool
  -> BGE semantic reranking, top-5 final evidence
  -> Qwen3.5-4B LoRA sequence classifier
  -> Four-way claim label + evidence IDs
```

| Stage | Design | Reason |
|---|---|---|
| Preprocessing | Lowercase and whitespace cleanup while preserving numbers, units, entities, and chemical symbols | Climate claims often depend on exact values, locations, and scientific terms |
| BM25 Retrieval | Retrieve top-1000 evidence candidates from the full corpus | Efficient first-stage recall with manageable reranking cost |
| BGE Reranking | Rank BM25 candidates by claim-evidence semantic similarity | Handles paraphrases such as `human emissions` vs `anthropogenic greenhouse gases` |
| Sequence Classification | Fine-tune Qwen3.5-4B with LoRA on claim + retrieved evidence | Matches inference-time noisy evidence instead of only clean gold evidence |

## Retrieval Experiments

BM25 candidate size and reranker choice were evaluated on the development set using Recall@5.

| Candidate Pool | Reranker | Recall@5 |
|---|---|---:|
| BM25 top-1000 | MiniLM | 0.204 |
| BM25 top-5000 | MiniLM | 0.191 |
| BM25 top-1000 | BGE | 0.223 |
| BM25 top-5000 | BGE | 0.206 |

**Finding:** BM25 top-1000 with BGE reranking gave the best tested retrieval trade-off. Expanding the candidate pool to 5000 introduced more weakly related evidence and made reranking noisier.

## Classifier Training Strategy

The main classifier issue was the distribution gap between training-time gold evidence and inference-time retrieved evidence.

| Training Strategy | Classification Accuracy |
|---|---:|
| Gold evidence training, tested on noisy retrieval | 0.45 |
| Retrieved evidence with relabelled retrieval misses | 0.38 |
| Retrieved evidence with original labels | 0.6169 |

**Finding:** Training on retrieved evidence while keeping original labels produced the best classification performance. Relabelling retrieval misses as `NOT_ENOUGH_INFO` distorted the label distribution and made the model over-conservative.

## End-to-End Results

### Development / Reported System Diagnostics

| System | Evidence F | Accuracy | Harmonic Mean |
|---|---:|---:|---:|
| BM25(1000) + BGE + Qwen3.5-4B LoRA | 0.19 | 0.61 | 0.29 |

These numbers reflect the final report configuration and show a clear pattern: classification accuracy was stronger than retrieval F-score, so retrieval remained the main bottleneck.

### Public Evaluation Snapshot

| Metric | Value |
|---|---:|
| Public evaluation rank | 5 |
| Harmonic Mean | 0.35 |
| Evidence Retrieval F-score | 0.26 |
| Claim Classification Accuracy | 0.57 |

The public snapshot is kept separate from the report diagnostics because it comes from a different evaluation setting.

## Error Analysis

The most common errors came from retrieval, not final label classification.

| Error Type | Example Failure Mode | Impact |
|---|---|---|
| Lexical overlap without factual relevance | Evidence shares words such as `households`, `millions`, or `emissions` but does not verify the same claim | Reranker may promote topically related but invalid evidence |
| Entity and scope mismatch | Claim refers to Australia, while retrieved evidence discusses the United Kingdom or a different population | Classifier receives misleading context |
| Numerical mismatch | Evidence contains similar percentages or years but refers to a different quantity | Supports/refutes distinction becomes unstable |
| BM25 recall ceiling | Gold evidence is absent from the top-1000 candidate pool | BGE reranker cannot recover missing evidence |

## Resource Constraints

A full dense retrieval attempt with BGE embeddings over roughly 1.2M evidence passages was explored but was not used in the final public configuration. Encoding took about 56 minutes on a Colab T4 GPU, and the FAISS IndexFlatIP setup ran into memory pressure. A lighter MiniLM dense retrieval attempt also created memory pressure when combined with the Qwen classifier.

The final pipeline therefore uses BM25 for candidate recall, BGE for reranking, and explicit memory cleanup between retrieval and classification stages.

## What This Project Demonstrates

- Retrieval-augmented fact-checking system design.
- Trade-off analysis between lexical retrieval, semantic reranking, and sequence classification.
- LoRA fine-tuning of an open-source LLM-style classifier under constrained GPU resources.
- Evidence-level error analysis for entity, scope, and numerical mismatch.
- Metric-aware development using Evidence F-score, Accuracy, and Harmonic Mean.

## Tech Stack

Python, BM25, BGE reranking, MiniLM comparison, Qwen3.5-4B, LoRA, RAG, sequence classification, retrieval evaluation, Colab GPU workflow.

## Resume-Ready Summary

Built a two-stage climate claim verification system with BM25 top-1000 candidate retrieval, BGE semantic reranking, and Qwen3.5-4B LoRA sequence classification; compared MiniLM/BGE rerankers and classifier training strategies, achieving public evaluation rank 5 with H-mean 0.35, Evidence F-score 0.26, and Claim Accuracy 0.57.

## Public Data Policy

This repository is a portfolio-safe summary. Restricted benchmark files, raw evidence corpora, prediction files, notebooks, and private evaluation artifacts are intentionally excluded.
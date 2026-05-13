# Climate Claim Verification RAG

This project documents a multi-stage retrieval-augmented fact-checking system for climate science claims. The task is to retrieve relevant evidence from an evidence corpus and classify each claim into one of four labels: `SUPPORTS`, `REFUTES`, `NOT_ENOUGH_INFO`, or `DISPUTED`.

The repository is prepared as a sanitized portfolio version. It focuses on task design, modeling strategy, and evaluation results without redistributing restricted course data.

## Task

Given a climate science claim, the system needs to:

1. Retrieve relevant evidence candidates from the evidence knowledge base.
2. Rerank candidates by claim-evidence relevance.
3. Predict the final claim label.
4. Evaluate both retrieval quality and classification accuracy.

## Method

The system follows a three-stage RAG architecture:

| Stage | Purpose | Implementation idea |
|---|---|---|
| Bi-Encoder retrieval | Fast candidate recall | Encode claims and evidence into dense vectors and retrieve top-k candidates |
| Cross-Encoder reranking | Fine-grained relevance scoring | Score claim-evidence pairs with joint encoding |
| Transformer / open-source LLM verification | Final label prediction | Use retrieved evidence to predict claim labels |

Additional retrieval optimization:

- Used BGE-M3 as the semantic retrieval model.
- Constructed hard negative samples to improve discrimination between relevant evidence and semantically similar but incorrect evidence.
- Tuned the pipeline on the development split with official evaluation metrics.

## Evaluation

Official evaluation metrics include evidence retrieval F-score, claim classification accuracy, and their harmonic mean.

| Metric | Value |
|---|---:|
| Public evaluation rank | 5 |
| Harmonic Mean | 0.35 |
| Evidence Retrieval F-score | 0.26 |
| Claim Classification Accuracy | 0.57 |

## Key Takeaways

- Retrieval and classification need to be optimized together; improving one metric alone may not improve the harmonic mean.
- Hard negative sampling is useful for claims whose surface wording is close to misleading evidence.
- Cross-Encoder reranking improves evidence quality before final classification, especially for ambiguous or disputed claims.

## Tech Stack

Python, BGE-M3, Bi-Encoder retrieval, Cross-Encoder reranking, Transformer models, RAG, hard negative sampling.

## Notes

This is a portfolio-safe project summary. Dataset files and restricted course materials are not included.
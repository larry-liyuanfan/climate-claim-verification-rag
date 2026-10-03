"""Isolated Torch process for a public learned-demo query; never imports FAISS."""
from __future__ import annotations

import json
from pathlib import Path
import sys
from time import perf_counter


def main() -> None:
    value = json.loads(sys.stdin.read())
    import torch
    from climate_rag.dense import SentenceTransformerEncoder
    torch.set_num_threads(4)
    start = perf_counter()
    encoder = SentenceTransformerEncoder(str(Path(value["model_dir"])), query_prompt_name="query", device="cpu")
    load_ms = (perf_counter() - start) * 1000
    text = value["claim"]
    prompt = encoder._model.prompts["query"] + text
    if len(encoder._model.tokenizer.encode(prompt)) > encoder._model.max_seq_length:
        raise ValueError("query_would_truncate")
    start = perf_counter()
    vector = encoder.encode_queries([text])
    print(json.dumps({"vector": vector.tolist(), "query_encode_ms": (perf_counter() - start) * 1000,
                      "query_model_load_ms": load_ms}))


if __name__ == "__main__":
    main()

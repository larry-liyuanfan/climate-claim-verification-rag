"""Rebuild public base-embedding HNSW assets; NOT historical LoRA/LTR recovery."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys
from time import perf_counter

from climate_rag.io import iter_evidence
from score_saved_citation_nli import digest

MODEL = "Qwen/Qwen3-Embedding-0.6B"
REVISION = "97b0c614be4d77ee51c0cef4e5f07c00f9eb65b3"
CORPUS_SHA = "c14315aee9feecbbbbc3b0c7101d978b52b47cee9a613336304bcaa736460c71"


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--evidence", type=Path, required=True)
    p.add_argument("--model-dir", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--phase", choices=["encode", "index"], help=argparse.SUPPRESS)
    args = p.parse_args()
    repo = Path(__file__).resolve().parents[1]
    for path in (args.model_dir, args.output):
        if (not path.is_absolute() or path.resolve().is_relative_to(repo)
                or any(part.casefold() in {"onedrive", "求职"} for part in path.resolve().parts)):
            raise ValueError("learned_assets_must_stay_outside_git_onedrive")
    if args.output.exists() and args.phase != "index":
        raise ValueError("unique_reconstruction_output_required")
    if digest(args.evidence) != CORPUS_SHA:
        raise ValueError("registered_public_corpus_required")
    if args.phase == "index":
        build_index(args.evidence, args.model_dir, args.output)
        return
    # Asset acquisition precedes expensive encoding. Download no executable code.
    from huggingface_hub import snapshot_download
    snapshot_download(MODEL, revision=REVISION, local_dir=args.model_dir,
                      allow_patterns=["*.json", "*.safetensors", "*.txt", "1_Pooling/*"])
    if args.phase is None:
        common = [sys.executable, "-X", "utf8", str(Path(__file__).resolve()), "--evidence", str(args.evidence),
                  "--model-dir", str(args.model_dir), "--output", str(args.output)]
        # Never load Torch and FAISS OpenMP runtimes in one Windows process.
        # No KMP_DUPLICATE_LIB_OK or duplicate-runtime suppression.
        subprocess.run([*common, "--phase", "encode"], check=True)
        subprocess.run([*common, "--phase", "index"], check=True)
        return
    import torch
    torch.set_num_threads(4)
    import numpy as np
    from climate_rag.dense import SentenceTransformerEncoder
    started = perf_counter()
    encoder = SentenceTransformerEncoder(str(args.model_dir), query_prompt_name="query", device="cpu")
    load_seconds = perf_counter() - started
    docs = list(iter_evidence(args.evidence))
    if len(docs) != 5240:
        raise ValueError("public_document_count_drift")
    # Audit context lengths; do not silently truncate evidence for this demo.
    token_lengths = [len(encoder._model.tokenizer.encode(d.text)) for d in docs]
    if max(token_lengths) > encoder._model.max_seq_length:
        raise ValueError("document_would_be_silently_truncated")
    started = perf_counter()
    args.output.mkdir(parents=True)
    (args.output / "encoding-source.py").write_bytes(Path(__file__).read_bytes())
    vectors = np.lib.format.open_memmap(args.output / "encoded.npy", mode="w+", dtype="float32", shape=(len(docs), encoder.dimension))  # type: ignore[no-untyped-call]
    for offset in range(0, len(docs), 8):
        rows = docs[offset:offset + 8]
        vectors[offset:offset + len(rows)] = encoder.encode_documents([d.text for d in rows], batch_size=8)
        if offset == 0 or offset % 128 == 0:
            print(json.dumps({"encoded": offset + len(rows), "of": len(docs), "seconds": perf_counter() - started}), flush=True)
    vectors.flush()
    embedding_seconds = perf_counter() - started
    model_files = {str(v.relative_to(args.model_dir)).replace("\\", "/"): digest(v)
                   for v in sorted(args.model_dir.rglob("*")) if v.is_file() and ".cache" not in v.parts}
    result = {"kind": "new_public_base_embedding_hnsw_reconstruction_not_historical_recovery",
        "model": MODEL, "revision": REVISION, "model_files": model_files,
        "corpus_sha256": CORPUS_SHA, "documents": len(docs), "vectors": len(vectors),
        "dimension": encoder.dimension, "batch_size": 8, "cpu_threads": 4,
        "query_prompt": "query", "max_document_tokens": max(token_lengths),
        "embedding_seconds": embedding_seconds, "encoded_sha256": digest(args.output / "encoded.npy"),
        "model_load_seconds": load_seconds,
        "ef_search": 64, "ltr_available": False, "reranker4b_available": False,
        "gold_read": False, "test_read": False, "quality_gain_claimed": False}
    with (args.output / "encoded-receipt.json").open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(result, stream, indent=2)
        stream.write("\n")
    print(json.dumps({"encoding_complete": len(docs), "seconds": embedding_seconds}))


def build_index(evidence: Path, model_dir: Path, output: Path) -> None:
    if any((output / name).exists() for name in ("index.faiss", "dense_index.json", "reconstruction.json")):
        raise ValueError("index_final_or_partial_artifact_already_exists")
    import numpy as np
    import faiss
    from climate_rag.dense import FaissANNIndex
    from climate_rag.io import write_json
    faiss.omp_set_num_threads(4)
    result = json.loads((output / "encoded-receipt.json").read_bytes())
    if (result["corpus_sha256"] != CORPUS_SHA or result["revision"] != REVISION
            or result["model"] != MODEL or digest(output / "encoded.npy") != result["encoded_sha256"]
            or any(digest(model_dir / name) != sha for name, sha in result["model_files"].items())):
        raise ValueError("encoding_receipt_drift")
    docs = list(iter_evidence(evidence))
    vectors = np.load(output / "encoded.npy", mmap_mode="r", allow_pickle=False)
    if vectors.shape != (len(docs), result["dimension"]) or not np.isfinite(vectors).all():
        raise ValueError("encoding_shape_or_numeric_error")
    started = perf_counter()
    index = FaissANNIndex(result["dimension"], kind="hnsw", hnsw_m=32, hnsw_ef_construction=200)
    index.index.hnsw.efSearch = 64
    index.build(vectors)
    result["index_build_seconds"] = perf_counter() - started
    index.save(output / "index.faiss")
    write_json(output / "dense_index.json", {"schema_version": 1,
        "encoder": {"type": "sentence_transformer", "name": str(model_dir), "dimension": result["dimension"],
                    "query_prefix": "", "query_prompt_name": "query", "adapter_path": None, "revision": None},
        "backend": {"type": "faiss", "kind": "hnsw", "dimension": result["dimension"], "parameters": index.parameters},
        "index_file": "index.faiss", "doc_ids": [d.evidence_id for d in docs], "texts": [d.text for d in docs]})
    result.update(index_bytes=(output / "index.faiss").stat().st_size,
                  index_sha256=digest(output / "index.faiss"), metadata_sha256=digest(output / "dense_index.json"),
                  encoding_source_sha256=digest(output / "encoding-source.py"),
                  index_build_source_sha256=digest(Path(__file__)))
    with (output / "reconstruction.json").open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(result, stream, indent=2)
        stream.write("\n")
    print(json.dumps({k: v for k, v in result.items() if k != "model_files"}))


if __name__ == "__main__":
    main()

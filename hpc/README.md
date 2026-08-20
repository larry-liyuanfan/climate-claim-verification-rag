# Spartan runbook

Verified account/project: `punim2936`. The restricted data directory is:

```text
/data/gpfs/projects/punim2936/nlp/COMP90042_2026-main/data
```

It contains the 167 MB evidence JSON and train/dev/test claim files. The scripts bind it read-only as `/data`; they never copy it into the repository or `$HOME`.

## Build once

```bash
export REPO_DIR="$PWD"
export DATA_DIR=/data/gpfs/projects/punim2936/nlp/COMP90042_2026-main/data
export ARTIFACT_DIR=/data/gpfs/projects/punim2936/climate-rag-artifacts
export CONTAINER_IMAGE=/data/gpfs/projects/punim2936/climate-rag-artifacts/climate-rag.sif
mkdir -p "$ARTIFACT_DIR"
sbatch --account=punim2936 hpc/build_image.sbatch
```

`build_image.sbatch` pins the Spartan module prerequisites and moves both the
Apptainer cache and temporary build directory into project storage. This avoids
the small home-directory quota being consumed by image layers.

If an unprivileged image build still exceeds the user's build-layer quota,
Spartan's reviewed PyTorch/CUDA module can provide the heavy runtime while a
project-storage virtual environment holds only the search dependencies:

```bash
bash hpc/submit_native.sh
```

The wrapper submits one environment job and makes BM25/Qwen dense jobs depend
on it. This is a recorded fallback, not a silent change of runtime.

All Hugging Face model downloads are redirected to
`${ARTIFACT_DIR}/.cache/huggingface` in project storage. The dense jobs must not
fall back to `$HOME/.cache`; the home quota is too small for Qwen3 model files.

Confirm the current GPU partition with `sinfo`/project access, then submit:

```bash
export SPARTAN_GPU_PARTITION=your_gpu_partition
sbatch --account=punim2936 hpc/build_bm25.sbatch
sbatch --account=punim2936 --partition="$SPARTAN_GPU_PARTITION" hpc/build_dense_flat.sbatch
sbatch --account=punim2936 --partition="$SPARTAN_GPU_PARTITION" hpc/build_ann_array.sbatch
sbatch --account=punim2936 --partition="$SPARTAN_GPU_PARTITION" hpc/prepare_ltr.sbatch
sbatch --account=punim2936 hpc/train_ltr.sbatch
sbatch --account=punim2936 --partition="$SPARTAN_GPU_PARTITION" hpc/evaluate_five_stage.sbatch
```

`build_ann_array.sbatch` requires the embedding cache and sidecar produced by `build_dense_flat.sbatch`. HNSW and IVF-PQ then reuse exactly the same rows.

For the already verified native Spartan environment, `build_ann_native.sbatch`
builds HNSW and IVF-PQ on CPU nodes without consuming a GPU. IVF-PQ uses a
deterministic 200,000-vector training sample. After both array tasks succeed,
`benchmark_ann_native.sbatch` encodes the fixed dev claims once and records
FlatIP-referenced Recall@5/10/50, batched QPS, single-query P50/P95, load time,
index bytes and bytes/document. These are ANN engineering metrics; they do not
replace gold-evidence retrieval metrics.

`prepare_ltr.sbatch` runs BM25+dense retrieval on the training claims, removes gold evidence from the negative pool, and emits `ltr_features.jsonl`. Training consumes that file; no manually assembled feature table is required.

The `*_native.sbatch` LTR chain uses the already verified native environment and
is CPU-only: HNSW hard-negative mining on train claims, LightGBM LambdaMART, then
a fixed dev comparison with paired bootstrap. Its final stage is explicitly the
deterministic feature reranker; it must not be described as Qwen3 reranking.

Local Qwen3 reranking uses a measured pilot before the full run. The pilot config
limits evaluation to eight claims and `evaluate_qwen_reranker_pilot.sbatch`
records model-load and per-query timing. Only after that gate should
`evaluate_qwen_reranker_native.sbatch` score the complete RRF Top-50 candidate
set. `compare_qwen_reranker_native.sbatch` performs the paired RRF-vs-reranker
comparison without repeating model inference. Model cache, predictions and
restricted evidence stay in project storage.

The 8B configs are an optional size/Pareto gate, not a promised improvement.
They force BF16 and batch size 1. Run the eight-claim pilot first on a measured
GPU shape; only submit the full config if the pilot fits memory, produces finite
scores and offers a defensible quality/latency trade-off. The full Slurm script
accepts `EXPERIMENT_CONFIG` so 0.6B, 4B and 8B runs remain explicit and hashed.
If the shared project filesystem lacks room for the 8B weight shards, set
`MODEL_CACHE_MODE=node-local`. Both reranker scripts then place the Hugging Face
cache under the Slurm node's ephemeral `${SLURM_TMPDIR}`/`${TMPDIR}` (falling
back to `/tmp`) and the full script permits the one-off download. This avoids
writing weights to `$HOME` or uploading them as artifacts; it intentionally
trades a repeated download for a bounded pilot/full gate.


Before reporting a result, retain Slurm logs, `run_manifest.json`, `metrics.json`, per-claim predictions, `sacct` elapsed/MaxRSS, allocated hardware, and storage/API cost. Do not claim an improvement if it was not run on the same split/final K with a paired comparison.

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

`prepare_ltr.sbatch` runs BM25+dense retrieval on the training claims, removes gold evidence from the negative pool, and emits `ltr_features.jsonl`. Training consumes that file; no manually assembled feature table is required.


Before reporting a result, retain Slurm logs, `run_manifest.json`, `metrics.json`, per-claim predictions, `sacct` elapsed/MaxRSS, allocated hardware, and storage/API cost. Do not claim an improvement if it was not run on the same split/final K with a paired comparison.

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_native_dense_uses_project_hugging_face_cache() -> None:
    script = (ROOT / "hpc" / "build_dense_native.sbatch").read_text()
    assert 'HF_HOME="${ARTIFACT_DIR}/.cache/huggingface"' in script
    assert 'HF_HUB_CACHE="${HF_HOME}/hub"' in script
    assert 'TRANSFORMERS_CACHE="${HF_HOME}/transformers"' in script
    assert "/home/" not in script


def test_container_dense_jobs_bind_cache_under_artifacts() -> None:
    for name in ("build_dense_flat.sbatch", "build_ann_array.sbatch"):
        script = (ROOT / "hpc" / name).read_text()
        assert "APPTAINERENV_HF_HOME=/artifacts/.cache/huggingface" in script
        assert "APPTAINERENV_HF_HUB_CACHE=/artifacts/.cache/huggingface/hub" in script
        assert "/home/" not in script


def test_embedding_lora_pilot_is_project_cached_and_bounded() -> None:
    script = (ROOT / "hpc" / "train_embedding_lora_pilot.sbatch").read_text()
    assert 'HF_HOME="${ARTIFACT_ROOT}/.cache/huggingface"' in script
    assert '${REPO_DIR:?set detached, exact-SHA REPO_DIR}' in script
    assert 'RUNTIME_ENV="${SWIFT_ENV:-${NODE_LOCAL_ROOT}/climate-swift-venv-3.9.3}"' in script
    assert 'PIP_CACHE_DIR="${NODE_LOCAL_ROOT}/climate-swift-pip-cache"' in script
    assert '"ms-swift==3.9.3"' in script
    assert "--model Qwen/Qwen3-Embedding-0.6B" in script
    assert "--task_type embedding" in script
    assert "--loss_type infonce" in script
    assert '--max_steps "${MAX_STEPS:-20}"' in script
    assert "/home/" not in script


def test_embedding_data_prep_requires_exact_sha_inputs() -> None:
    script = (ROOT / "hpc" / "prepare_embedding_training.sbatch").read_text()
    assert '${REPO_DIR:?set detached, exact-SHA REPO_DIR}' in script
    assert '${HARD_NEGATIVES:?set HARD_NEGATIVES}' in script
    assert "module load GCC/11.3.0 OpenMPI/4.1.4 PyTorch/2.1.2-CUDA-12.2.0" in script
    assert '--split-seed "${SPLIT_SEED:-45}"' in script
    assert "/home/" not in script

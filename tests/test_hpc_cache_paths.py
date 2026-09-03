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
    assert "#SBATCH --partition=gpu-a100-mig" in script
    assert "#SBATCH --gres=gpu:1g.20gb:1" in script
    assert "#SBATCH --gpus=" not in script
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
    assert 'PYTHONPATH="${REPO_DIR}/src:${PYTHONPATH:-}"' in script
    assert '--split-seed "${SPLIT_SEED:-45}"' in script
    assert "/home/" not in script


def test_candidate_supported_ltr_copies_git_objects_to_node_local_storage() -> None:
    script = (ROOT / "hpc" / "rebuild_candidate_supported_ltr.sbatch").read_text()
    assert "git clone --no-local --no-checkout" in script
    assert "git clone --shared" not in script
    assert "trap 'rm -rf" in script
    assert '${CLIMATE_GIT_COMMIT:?Set an exact pushed Git commit}' in script


def test_public_v2_jobs_preserve_module_pythonpath() -> None:
    scripts = sorted((ROOT / "hpc").glob("public_v2_*.sbatch"))
    assert len(scripts) == 9
    expected = 'PYTHONPATH="${REPO_DIR}/src${PYTHONPATH:+:${PYTHONPATH}}"'
    for script_path in scripts:
        assert expected in script_path.read_text(), script_path.name
    runtime = (ROOT / "hpc" / "public_v2_runtime.sh").read_text()
    assert 'runtime_site_packages="${runtime_env}/lib/python3.10/site-packages"' in runtime
    assert 'PYTHONPATH="${runtime_site_packages}${PYTHONPATH:+:${PYTHONPATH}}"' in runtime


def test_public_v2_pilots_use_frozen_hugging_face_revision() -> None:
    script = (ROOT / "hpc" / "public_v2_train_pilots.sbatch").read_text()
    assert "--model Qwen/Qwen3-Embedding-0.6B" in script
    assert "--model_revision 97b0c614be4d77ee51c0cef4e5f07c00f9eb65b3" in script
    assert "--use_hf true" in script

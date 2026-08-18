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

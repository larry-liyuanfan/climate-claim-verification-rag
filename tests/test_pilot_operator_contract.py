"""Static operator invariants; not GPU/Slurm runtime validation."""
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[1] / "hpc/budget_agent_pilot_observed.sbatch"


def test_exact_frozen_source_and_base_wrapper() -> None:
    text = SCRIPT.read_text()
    assert "ca9fa53f2d4b70091fd08a8a1512e263f329fc61" in text
    assert 'bash <(tar -xOf "${CLIMATE_SOURCE_TAR}" hpc/budget_agent_pilot.sbatch)' in text
    assert "climate-authored-pilot-ca9fa53-20260929-user-resume" in text
    assert 'test "${CLIMATE_AGENT_PHASE:-}" = pilot' in text
    assert "python" not in text.lower().replace("python source", "")


def test_one_released_gpu_envelope() -> None:
    text = SCRIPT.read_text()
    for directive in ("--gres=gpu:A100:1", "--cpus-per-task=8", "--mem=32G",
                      "--time=00:15:00", "--tmp=30G", "--signal=B:USR1@60"):
        assert f"#SBATCH {directive}" in text
    assert 'export CUDA_VISIBLE_DEVICES=' not in text
    assert 'test -n "${CUDA_VISIBLE_DEVICES:-}"' in text


def test_diagnostics_do_not_claim_exact_peaks_or_retry() -> None:
    text = SCRIPT.read_text()
    assert '"precise_cuda_peak_measured":false' in text
    assert '"separate_model_load_time_measured":false' in text
    assert 'trap finish EXIT' in text
    assert "snapshot pre_timeout" in text
    assert 'exit "${OBS_STATUS}"' in text
    assert "sbatch " not in text
    assert "2097152" in text  # bounded copying of own small result/consumption receipts
    assert '"${directory}/run.json" "${directory}/run.consumed.json"' in text

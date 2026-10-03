"""Synthetic subprocess harness, never an executable real-model route."""
import copy
import hashlib
import json
import os
from pathlib import Path
import sys
import time

REPO = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO / "scripts"), str(REPO / "src"), str(REPO / "tests")]

import pytest  # noqa: E402
from climate_rag.scifact_grounding import GoldClaim, Rationale  # noqa: E402
from climate_rag.scifact_semantic_contract import MODEL_SHA  # noqa: E402
from climate_rag.scifact_utility_contract import identity  # noqa: E402
from test_scifact_evidence_bottleneck import Backend, SELECT, LABEL  # noqa: E402
from test_scifact_relation_verifier import assessment  # noqa: E402
from test_scifact_document_verifier import inputs  # noqa: E402
import scifact_bottleneck_execution as execution  # noqa: E402


def fixture_release(tmp):
    frame, corpus = inputs()
    claims = [{"id": i, "claim": frame["observation"]["immutable_claim"]} for i in range(1, 25)]
    frames = [copy.deepcopy(frame) for _ in claims]
    prepared = tmp / "prepared"
    (prepared / "inference").mkdir(parents=True)
    raw = b"\n".join(json.dumps({"doc_id": d.doc_id, "title": d.title, "abstract": list(d.sentences),
                              "structured": False}).encode() for d in corpus.values())
    (prepared / "inference/corpus.jsonl").write_bytes(raw)
    run = tmp / "run"
    return {"scope": "synthetic_fixture", "protocol": execution.PROTOCOL, "generation_contract": None,
        "routes": list(execution.ROUTES), "max_generations": 96, "planned_route_results": 72,
        "source_git": "a"*40, "model_sha256": MODEL_SHA, "frames_identity": identity(frames),
        "ordered_ids_sha256": identity([c["id"] for c in claims]), "prepared": str(prepared),
        "run_directory": str(run), "output": str(run / "inference"), "reports": str(run / "reports"),
        "execution_release_file": str(tmp / "release.json"),
        "status": "authorized_synthetic_only", "source_archive": str(tmp / "synthetic.tar"),
        "source_archive_sha256": "b"*64, "max_operator_seconds": 90, "max_prepare_seconds": 20,
        "max_worker_seconds": 30, "scoring_reserve_seconds": 30}


def validate_fixture(release):
    assert release["scope"] == "synthetic_fixture" and release["status"] == "authorized_synthetic_only"


def patch_boundaries(mp, mode):
    import scifact_evidence_input as adapter
    import climate_rag.scifact_semantic_contract as contract
    import run_scifact_evidence_bottleneck as runner
    import score_scifact_evidence_bottleneck as scorer
    frame, corpus = inputs()
    claims = [{"id": i, "claim": frame["observation"]["immutable_claim"]} for i in range(1, 25)]
    frames = [copy.deepcopy(frame) for _ in claims]
    release_file = Path(sys.argv[sys.argv.index("--release")+1])
    mp.setattr(execution, "trusted_work", lambda: release_file.parent.parent / "scratch")
    mp.setattr(execution, "validate_release", validate_fixture)
    mp.setattr(adapter, "check_prepared", lambda p, r: ({}, claims))
    mp.setattr(adapter, "load_frames", lambda *args: frames)
    raw = b"\n".join(json.dumps({"doc_id": d.doc_id, "title": d.title, "abstract": list(d.sentences),
                              "structured": False}).encode() for d in corpus.values())
    mp.setattr(contract, "CORPUS_SHA", hashlib.sha256(raw).hexdigest())
    mp.setattr(runner, "CORPUS_SHA", contract.CORPUS_SHA)
    mp.setattr(contract, "TOKENIZER_SHA", {})
    mp.setattr(runner, "TOKENIZER_SHA", {})
    mp.setattr(runner, "verify_source_tree", lambda *args: None)
    mp.setattr(runner, "verify_manifest", lambda *args: {})
    original_require = runner.require
    mp.setattr(runner, "require", lambda condition, message:
               None if message == "allocated_gpu_required" else original_require(condition, message))
    backend = Backend([assessment(), SELECT, LABEL, LABEL]*24)
    generate = backend.generate
    def delayed(*args):
        if backend.calls == 1 and mode == "worker_timeout":
            time.sleep(60)
        return generate(*args)
    backend.generate = delayed
    def provider(*args, **kwargs):
        if mode == "load_failure":
            raise RuntimeError("synthetic_model_load_failure")
        return backend
    mp.setattr(runner, "BottleneckProvider", provider)
    mp.setattr(runner, "GenerationBinding", lambda *args: None)
    from transformers import AutoTokenizer
    mp.setattr(AutoTokenizer, "from_pretrained", lambda *args, **kwargs: backend.base.tokenizer)
    def gold(ids, corpus, reports):
        # No synthetic gold is available to either preparation or runner.
        proof = json.loads((reports.parent / "inference/worker-exit.json").read_bytes())
        assert proof["child_reaped"] and proof["returncode"] == 0
        assert (reports / "audited-cost-before-gold.json").exists()
        (reports / "gold-read-started.json").write_text('{"synthetic":true}')
        if mode == "score_timeout":
            time.sleep(60)
        return [GoldClaim(i, "Synthetic claim", {77: (Rationale("SUPPORT", (2, 7)),)}, (77,)) for i in ids]
    mp.setattr(scorer, "load_original_gold", gold)


def main():
    phase, mode = sys.argv[1:3]
    sys.argv = [sys.argv[0], *sys.argv[3:]]
    os.environ["SLURM_JOB_ID"] = "synthetic-only"
    os.environ["CUDA_VISIBLE_DEVICES"] = "synthetic-no-GPU"
    with pytest.MonkeyPatch.context() as mp:
        patch_boundaries(mp, mode)
        if phase == "worker":
            import run_scifact_evidence_bottleneck as runner
            runner.main()
            if mode == "worker_exit":
                raise SystemExit(7)
        else:
            import score_scifact_evidence_bottleneck as scorer
            scorer.main()


if __name__ == "__main__":
    main()

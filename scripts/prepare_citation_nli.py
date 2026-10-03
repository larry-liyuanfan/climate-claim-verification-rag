"""Acquire one pinned public automatic-evaluation model; never upload inputs."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from huggingface_hub import snapshot_download

from climate_rag.semantic_proxy import MODEL, REVISION
from score_saved_citation_nli import digest

FILES = ("config.json", "model.safetensors", "tokenizer.json", "tokenizer_config.json",
         "special_tokens_map.json", "added_tokens.json", "spm.model")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--destination", type=Path, required=True)
    args = p.parse_args()
    path = args.destination.resolve()
    if (not args.destination.is_absolute() or path.is_relative_to(Path(__file__).resolve().parents[1])
            or any(v.casefold() in {"onedrive", "求职"} for v in path.parts)):
        raise ValueError("model_cache_must_stay_outside_git_onedrive")
    if (path / "verified-model.json").exists():
        raise ValueError("already_prepared_do_not_reacquire")
    snapshot_download(MODEL, revision=REVISION, local_dir=path, allow_patterns=list(FILES))
    result = {"model": MODEL, "revision": REVISION, "files": {name: digest(path / name) for name in FILES},
              "bytes": sum((path / name).stat().st_size for name in FILES), "gold_read": False}
    with (path / "verified-model.json").open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(result, stream, indent=2)
        stream.write("\n")
    print(json.dumps({"bytes": result["bytes"], "model_manifest_sha256": digest(path / "verified-model.json")}))


if __name__ == "__main__":
    main()

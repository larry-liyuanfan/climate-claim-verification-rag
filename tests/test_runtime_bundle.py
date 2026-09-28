import pytest

from climate_rag.runtime_bundle import resolve_bundle_args


def arguments(tmp_path):
    (tmp_path / "input").write_text("fixture")
    args = ["--provider", "local-qwen", "--phase", "pilot"]
    for key in ("evidence", "protocol", "model-dir", "model-manifest", "execution-manifest"):
        args.extend(["--" + key, "{INPUT}/input"])
    for key in ("expected-protocol-sha256", "expected-execution-sha256"):
        args.extend(["--" + key, "a" * 64])
    return args


def test_valid_bundle_resolves_only_inside_node(tmp_path):
    result = resolve_bundle_args(arguments(tmp_path), tmp_path)
    assert "{INPUT}/input" not in result
    assert str(tmp_path / "input") in result


@pytest.mark.parametrize("value", ["/persistent/private-data", "{INPUT}/../secret", "{INPUT}/absent"])
def test_bundle_rejects_external_or_missing_path(tmp_path, value):
    args = arguments(tmp_path)
    args[args.index("--evidence") + 1] = value
    with pytest.raises(ValueError):
        resolve_bundle_args(args, tmp_path)


@pytest.mark.parametrize("extra", [["--output", "/outside"], ["--phase", "validation"],
                                  ["--reranker-dir", "{INPUT}/input"]])
def test_no_output_override_duplicates_or_unpaired_reranker(tmp_path, extra):
    with pytest.raises(ValueError):
        resolve_bundle_args(arguments(tmp_path) + extra, tmp_path)

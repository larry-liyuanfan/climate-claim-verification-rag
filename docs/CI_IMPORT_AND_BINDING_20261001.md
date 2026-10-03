# CI collection and adapter-binding repair

The `quality` run [36841857351](https://github.com/larry-liyuanfan/climate-claim-verification-rag/actions/runs/36841857351)
at `d6a509f494da2e0c07b30fcab8e6a4880d3a2469` installed its pinned dependencies,
including CPU Torch, and passed Ruff/source type checks. Test collection then
failed in nine modules importing top-level `scripts/` entry points. An external
developer `PYTHONPATH` had masked this checkout-test configuration defect.

## Bounded corrections

- Declare `scripts` in pytest's `pythonpath` configuration. A subprocess regression
  clears external `PYTHONPATH` and `PYTEST_ADDOPTS`, collects exactly the nine
  affected modules, and checks that their test nodes were actually collected.
- Derive the read-continuation policy's top-level `adapter_active` and
  `adapter_model_sha256` from the same inherited checkpoint policy. The real
  runner binds this complete policy. Do not add permissive provider defaults.
- Replace a legacy partial-scoring fixture's placeholder training SHA with its
  actual frozen legacy `TRAINING_SHA`. The mixed-checkpoint release requirement,
  pre-gold cost persistence and twelve-question failure denominator are unchanged.
  Existing mixed-checkpoint negative tests still reject a missing release before
  loading gold or publishing a score.

No tests are disabled and no workflow step is removed. This is a runtime/testing
contract repair, not a new model result. It does not consume data or submit a job.
New policy/source hashes must be packaged anew; old frozen releases, artifacts
and successful historical executions are not rewritten or reinterpreted.

## Reproduction and scope

With the workflow's validation dependencies installed, clear external
`PYTHONPATH`, then run `python -m pytest -q`, `ruff check src scripts tests`,
`mypy src/climate_rag`, `mypy scripts/package_scifact_source.py` and the tracked
secret scan. The authoritative Linux full-suite result is the `quality` run
on the repair commit, not a focused test result or Windows-only platform skip.

Local Windows/Python 3.12 validation with CPU Torch 2.7.1 and no external
`PYTHONPATH`: **1113 passed, 1 existing POSIX-only skip**, seven warnings in
120.68 seconds. The targeted contract/import checks passed all 71 cases; Ruff
and the existing CI source/package-script type checks passed. `pip check` found
no broken requirements. Existing WSL Ubuntu supplies POSIX/`os.killpg`; POSIX is
an operating-system interface, not a missing Python package.

The source and package-script mypy checks match the existing CI contract. An
additional exploratory strict check of the read-continuation entry point follows
imports into five older operator scripts and reports 92 pre-existing annotation
errors there. This bounded repair does not claim whole-`scripts/` type cleanliness
or silently suppress those errors.

# Local validation environments

## Why Torch and POSIX appeared missing

These are different issues, not two missing pip packages:

- `.venv` is the lightweight lint/type-check environment and does not contain
  Torch. The isolated `.venv-validation` already contains Torch `2.7.1+cpu`,
  Transformers `4.51.3`, PEFT `0.15.2`, Accelerate `1.6.0` and Safetensors `0.5.3`.
  Select its interpreter explicitly; activating a different environment does
  not make these packages available there.
- POSIX denotes operating-system interfaces. Windows Python reports
  `os.name == "nt"`; installing a package or launching it from Git Bash does not
  provide Linux ownership, permissions or process-group semantics. Use Linux
  for these checks, never mock the platform to make a production preflight pass.
- The base project and `.[test]` do not install the complete optional model stack.
  A minimal CLI installation is not evidence of full-suite readiness.

References: [Python POSIX availability](https://docs.python.org/3/library/posix.html)
and [PyTorch platform-specific installation](https://pytorch.org/get-started/locally/).

## Windows: use the existing Torch environment

Run from the Climate checkout in PowerShell:

```powershell
$env:HF_HUB_OFFLINE = '1'
$env:TRANSFORMERS_OFFLINE = '1'
& .\.venv-validation\Scripts\python.exe -m pip check
& .\.venv-validation\Scripts\python.exe -m pytest -q tests/test_torch_compat.py tests/test_adapter_integrity.py tests/test_scifact_claim_group_mean.py
```

This is CPU validation, not GPU availability or a model-quality benchmark.
Do not install CUDA wheels or download model weights merely to run these fixtures.

## WSL Ubuntu: isolated POSIX checks

The existing Ubuntu installation now has a dedicated minimal environment:
`/home/larry/.local/share/climate-rag/posix-validation`.
It contains Python `3.12.3`, pip `26.2.1` and pytest `8.3.5`. It deliberately does
not duplicate the Windows Torch stack. System Python, authentication and the
frozen Spartan runtime were not modified.

Run from PowerShell using direct `--exec` arguments, not an interpolated shell
string. The test's temporary files reside on the Linux filesystem under `/tmp`;
the source checkout is read from the Windows mount.

```powershell
wsl -d Ubuntu --exec env PYTHONPATH=/mnt/e/Project/_codex_worktrees/climate-representation-eval/src PYTHONDONTWRITEBYTECODE=1 /home/larry/.local/share/climate-rag/posix-validation/bin/python -m pytest -q -p no:cacheprovider /mnt/e/Project/_codex_worktrees/climate-representation-eval/tests/test_private_diagnostics_v3.py
```

The environment was created with `python3 -m venv --without-pip` at that explicit
Linux path because Ubuntu lacked `ensurepip`. pip was bootstrapped from
`https://bootstrap.pypa.io/get-pip.py` **inside this venv only**, followed by
`python -m pip install pytest==8.3.5`. No sudo/global pip installation was used.
For a different host, first locate its Linux user and checkout paths; do not
assume the paths above are portable.

## Verified on 2026-10-01

Source checked: `c364e85be0a70d70f5aaf1ab6ca06f1a5c46194a`.
Only documentation changed after these checks.

| Check | Observed result |
|---|---|
| Windows Torch import and tensor arithmetic | `2.7.1+cpu`; `[2, 3] * 2 == [4, 6]` |
| Windows validation environment dependency consistency | `pip check`: no broken requirements |
| Torch compatibility, real tiny PEFT checkpoint, claim-group optimizer fixtures | **26 passed, 0 skipped**, 14.91 s |
| WSL private diagnostic quota/permissions/symlink fixtures | **4 passed, 0 skipped**, 0.22 s |
| WSL platform | `os.name=posix`, `posix` import and `os.killpg` available |
| WSL environment dependency consistency | `pip check`: no broken requirements |

The four WSL checks include `test_symlink_in_diagnostic_root_fails_closed`, which
is intentionally skipped by Windows Python. This is a complementary targeted
validation, **not** a new full Linux-suite run, Slurm integration test, GPU test,
training run or held-out evaluation. No new Spartan job was submitted.
Historical full-suite receipts retain their original source revisions and scope
in `docs/verified-runs/local-torch-validation-20260930.json` and
`docs/verified-runs/scifact-bounded-cpu-closeout-20260930.json`.

"""Reuse exact-tree packaging and real wrapper positive/negative checks."""
import argparse
from pathlib import Path

from package_scifact_utility8 import freeze


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--commit", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bash", required=True)
    args = parser.parse_args()
    print(freeze(args.repo, args.commit, args.output, args.bash,
                 wrapper_path="hpc/scifact_natural.sbatch", tag="natural-fit24"))

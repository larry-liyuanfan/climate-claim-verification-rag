"""Base-only natural FIT24 run, sharing the frozen utility transport and loader."""
from run_scifact_natural_operator import validate_release
from run_scifact_utility8 import arguments, run_inference


if __name__ == "__main__":
    run_inference(arguments(), validate_release, natural_fit=True)

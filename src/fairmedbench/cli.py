import argparse
from pathlib import Path
import yaml

from fairmedbench.demo import run_demo
from fairmedbench.ddi_experiment import run_ddi


def main():
    parser = argparse.ArgumentParser(prog="fairmedbench")
    parser.add_argument("mode", choices=["demo", "ddi"])
    parser.add_argument("--config", default="configs/demo.yaml")
    args = parser.parse_args()
    config = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    if args.mode == "demo":
        run_demo(config)
    else:
        run_ddi(config)


if __name__ == "__main__":
    main()

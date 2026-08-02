"""Fetch the datasets this project can train on.

    python download_data.py                  # the defaults: MNIST + Fashion-MNIST
    python download_data.py --dataset celeba64
"""

import argparse

import bootstrap  # noqa: F401  (puts the repository root on sys.path)
from common import data

DEFAULTS = ["mnist", "fashion-mnist"]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", nargs="*", default=DEFAULTS, choices=data.DATASET_NAMES)
    parser.add_argument("--data-root", default="data")
    args = parser.parse_args()

    for name in args.dataset:
        print(data.download_dataset(name, args.data_root))


if __name__ == "__main__":
    main()

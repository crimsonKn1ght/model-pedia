"""Fetch the dataset used to evaluate DDIM sampling.

    python data.py

DDIM adds no training of its own, so this only needs the dataset that the
project-08 DDPM was trained on, in order to compute reference metrics.
"""

import argparse

import bootstrap  # noqa: F401
from common import data, utils

DEFAULTS = ["mnist"]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", nargs="*", default=DEFAULTS, choices=data.DATASET_NAMES)
    parser.add_argument("--data-root", default=str(utils.DEFAULT_DATA_ROOT))
    args = parser.parse_args()

    for name in args.dataset:
        print(data.download_dataset(name, args.data_root))


if __name__ == "__main__":
    main()

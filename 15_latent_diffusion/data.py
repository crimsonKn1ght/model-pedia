"""Fetch the datasets this project can train on.

    python data.py
    python data.py --dataset celeba64
"""

import argparse

import bootstrap  # noqa: F401
from common import data, utils

DEFAULTS = ["fashion-mnist"]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", nargs="*", default=DEFAULTS, choices=data.DATASET_NAMES)
    parser.add_argument("--data-root", default=str(utils.DEFAULT_DATA_ROOT))
    args = parser.parse_args()

    for name in args.dataset:
        print(data.download_dataset(name, args.data_root))


if __name__ == "__main__":
    main()

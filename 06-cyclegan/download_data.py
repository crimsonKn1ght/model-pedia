"""Prepare the two unpaired domains.

    python download_data.py                  # CIFAR-10 (horse and deer classes)
    python download_data.py --horse2zebra    # the original CycleGAN dataset

``--horse2zebra`` needs outbound access to ``people.eecs.berkeley.edu``.  If
that host is unreachable, the class-subset and two-dataset tasks still work and
demonstrate exactly the same mechanics.
"""

import argparse

import bootstrap  # noqa: F401
from common import data, utils

DEFAULTS = ["cifar10"]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", nargs="*", default=DEFAULTS, choices=data.DATASET_NAMES)
    parser.add_argument("--data-root", default=str(utils.DEFAULT_DATA_ROOT))
    parser.add_argument("--horse2zebra", action="store_true")
    args = parser.parse_args()

    for name in args.dataset:
        print(data.download_dataset(name, args.data_root))

    if args.horse2zebra:
        try:
            path = data.download_archive(data.HORSE2ZEBRA_URL, args.data_root, "horse2zebra")
            print(f"horse2zebra: ready at {path}")
        except Exception as exc:
            print(f"horse2zebra: NOT AVAILABLE ({exc})")
            print("the class-subset task does not need this download")


if __name__ == "__main__":
    main()

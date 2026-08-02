"""Prepare a paired dataset for Pix2Pix.

Two options:

    python download_data.py                    # edges -> photo, built from a
                                               # standard dataset, no extra
                                               # download and always available
    python download_data.py --facades          # the real pix2pix facades set

The edges->photo task is derived on the fly: the target is a real photo, the
input is its Sobel edge map.  The pairing is exact, which is what Pix2Pix needs,
and it is the same setup as the paper's ``edges2shoes``.

``--facades`` fetches the original dataset from the Berkeley mirror.  It needs
outbound access to ``efrosgans.eecs.berkeley.edu``; if that host is unreachable
the edges->photo task still works.
"""

import argparse

import bootstrap  # noqa: F401
from common import data, utils

DEFAULTS = ["cifar10"]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", nargs="*", default=DEFAULTS, choices=data.DATASET_NAMES)
    parser.add_argument("--data-root", default=str(utils.DEFAULT_DATA_ROOT))
    parser.add_argument("--facades", action="store_true", help="also fetch the facades dataset")
    args = parser.parse_args()

    for name in args.dataset:
        print(data.download_dataset(name, args.data_root))

    if args.facades:
        try:
            path = data.download_archive(data.FACADES_URL, args.data_root, "facades")
            print(f"facades: ready at {path}")
        except Exception as exc:  # network or mirror problems
            print(f"facades: NOT AVAILABLE ({exc})")
            print("the edges->photo task does not need this download")


if __name__ == "__main__":
    main()

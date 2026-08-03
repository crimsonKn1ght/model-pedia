"""There is nothing new to train here - that is the whole point of this project.

    python train.py --help

DDIM changes how an already-trained DDPM is *sampled*, so this script is a thin delegate to
project 12's training loop, kept so the folder has the same entry points as every other
project and can be smoke-tested on its own. If you already have a project 12 checkpoint,
skip this and go straight to ``evaluate.py``.

The delegate loads project 12's ``train`` module by path rather than by name, because this
file is also called ``train.py`` and a plain import would find itself.
"""

from __future__ import annotations

from _paths import load_diffusion_module

_train = load_diffusion_module("train")

build_parser = _train.build_parser
run_training = _train.run_training


def main() -> None:
    _train.main()


if __name__ == "__main__":
    main()

"""Make project 12's U-Net, diffusion process, datasets and metrics importable here.

This project has no model of its own - that is the point. DDIM is a different way to
*sample* from an already-trained DDPM, so the honest structure is to reuse project 12's
U-Net, diffusion process, datasets and metrics rather than copy them and let the two drift
apart. Every other project in this repository is self-contained; this one and project 14
are explicitly not, and the reference table says as much when it lists "same trained DDPM"
as this row's dataset.

Project 12 is **appended** to the path rather than prepended, so a module that exists here
wins and only the missing ones fall through - ``from model import ...`` and
``from data import ...`` reach project 12, while ``from train import ...`` stays local.
Where a local file has to shadow a project 12 module of the same name, use
:func:`load_diffusion_module` to reach the original unambiguously.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

DIFFUSION_PROJECT = Path(__file__).resolve().parent.parent / "12_diffusion"

if not DIFFUSION_PROJECT.is_dir():
    raise ImportError(
        f"expected project 12 at {DIFFUSION_PROJECT}. This project reuses its U-Net and "
        "diffusion process; run it from a full checkout of the repository."
    )

if str(DIFFUSION_PROJECT) not in sys.path:
    sys.path.append(str(DIFFUSION_PROJECT))


def load_diffusion_module(name: str) -> ModuleType:
    """Import ``<12_diffusion>/<name>.py`` under a distinct name, shadowing nothing."""
    alias = f"diffusion_project_{name}"
    if alias in sys.modules:
        return sys.modules[alias]
    spec = importlib.util.spec_from_file_location(alias, DIFFUSION_PROJECT / f"{name}.py")
    if spec is None or spec.loader is None:
        raise ImportError(f"could not load {name}.py from {DIFFUSION_PROJECT}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[alias] = module
    spec.loader.exec_module(module)
    return module

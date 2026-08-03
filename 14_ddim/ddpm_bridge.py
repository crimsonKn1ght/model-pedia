"""Loads the U-Net and diffusion process defined in project 13.

DDIM is a *sampler*, not a new model: it reuses a DDPM's trained noise
predictor unchanged.  Rather than duplicate that architecture here, this module
imports it directly from ``13_ddpm/model.py`` so the two projects can never
drift apart.  The folder name starts with a digit, so it cannot be imported with
normal ``import`` syntax and is loaded by file path instead.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
from types import ModuleType

DDPM_MODEL_PATH = Path(__file__).resolve().parent.parent / "13_ddpm" / "model.py"


def load_ddpm_module() -> ModuleType:
    if not DDPM_MODEL_PATH.exists():
        raise FileNotFoundError(
            f"expected the DDPM model definition at {DDPM_MODEL_PATH}; "
            "this project builds on project 13"
        )
    spec = importlib.util.spec_from_file_location("ddpm_model", DDPM_MODEL_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


_ddpm = load_ddpm_module()

UNet = _ddpm.UNet
GaussianDiffusion = _ddpm.GaussianDiffusion
make_beta_schedule = _ddpm.make_beta_schedule

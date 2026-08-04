"""Run every project's train -> evaluate pipeline on random tensors.

    python scripts/smoke_test.py
    python scripts/smoke_test.py --projects 01_classic_classifier 04_autoencoder
    python scripts/smoke_test.py --skip-metrics

``scripts/metric_selftest.py`` runs first, because this script deliberately says
nothing about whether a number is correct - only that one was produced - and a
broken metric would sail through every check below.

No dataset is downloaded and no pretrained weight is fetched: each project's
``--smoke-test`` flag swaps in a tiny synthetic split. This checks that the
scripts import, that shapes line up end to end, that a checkpoint round-trips
through ``evaluate.py`` and that every figure is written. It says nothing about
whether the models learn anything - that is what the real runs are for.

Projects with more than one training stage (the VQ-VAE's code prior, latent
diffusion's autoencoder) list the follow-up scripts under ``extra``, so the whole
pipeline is exercised rather than just the first stage.
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

# project -> flags for the train/evaluate pass, and the study scripts to exercise
PROJECTS = {
    "01_classic_classifier": {
        "train": [],
        "studies": [["compare.py", "--models", "mlp", "lenet"]],
    },
    "02_cnn_classifier": {
        "train": [],
        "studies": [["compare.py", "--study", "depth"]],
    },
    "03_transfer_learning": {
        # The study lives in train.py --mode both rather than a separate script.
        "train": ["--backbone", "mobilenet_v3_small", "--mode", "finetune", "--image-size", "64"],
        "studies": [
            ["train.py", "--backbone", "mobilenet_v3_small", "--mode", "both", "--image-size", "64"]
        ],
    },
    "04_autoencoder": {
        "train": [],
        "studies": [["sweep.py", "--latent-dims", "4", "16"]],
    },
    "05_denoising_autoencoder": {
        "train": [],
        "studies": [["compare.py", "--study", "skips"]],
    },
    "06_variational_autoencoder": {
        "train": [],
        "studies": [["compare.py", "--study", "likelihood"]],
    },
    "07_vector_quantized_ae": {
        "train": [],
        # The prior is a second training stage, so it gets its own entry in the recipe.
        "extra": [["train_prior.py", "--checkpoint", "{checkpoint}"]],
        "studies": [["compare.py", "--study", "codebook_rule"]],
    },
    "08_gan": {
        "train": [],
        "studies": [["compare.py", "--study", "loss"]],
    },
    "09_conditional_gan": {
        "train": [],
        "studies": [["compare.py", "--study", "mode"]],
    },
    "10_image_to_image_gan": {
        "train": [],
        "studies": [["compare.py", "--study", "skips"]],
    },
    "11_normalizing_flow": {
        "train": [],
        "studies": [["compare.py", "--study", "dequantisation"]],
    },
    "12_diffusion": {
        "train": [],
        "studies": [["compare.py", "--study", "prediction"]],
    },
    "13_faster_diffusion": {
        # train.py delegates to project 12; evaluate.py is the new code here.
        "train": [],
        "evaluate": ["--out-dir", "outputs/smoke/eval"],
        "studies": [["compare.py"]],
    },
    "14_latent_diffusion": {
        # Stage one first, then the latent diffusion model trains on top of it.
        "train_script": "train_autoencoder.py",
        "train": [],
        "extra": [["train.py", "--autoencoder", "{checkpoint}", "--out-dir", "{out_dir}"]],
        "checkpoint_name": "best.pt",
        "studies": [["compare.py", "--study", "channels"]],
    },
    "15_vision_transformer": {
        "train": [],
        "studies": [["compare.py", "--study", "arch"]],
    },
    "16_self_supervised": {
        "train": [],
        "studies": [["compare.py", "--study", "method"]],
    },
    "17_masked_image_modeling": {
        "train": [],
        # Fine-tuning two classifiers is the slow half of evaluate.py and adds
        # nothing a shape check needs.
        "evaluate": ["--tasks", "recon", "probe"],
        "studies": [["compare.py", "--study", "target"]],
    },
    "18_semantic_segmentation": {
        "train": [],
        "studies": [["compare.py", "--study", "loss"]],
    },
    "19_object_detection": {
        "train": [],
        "studies": [["compare.py", "--study", "box_loss"]],
    },
    "20_image_captioning": {
        "train": [],
        "studies": [["compare.py", "--study", "smoothing"]],
    },
    "21_unpaired_translation": {
        "train": [],
        "studies": [["compare.py", "--study", "cycle"]],
    },
}

SMOKE_OUTPUT_DIR = "outputs/smoke"


def run(command: list[str], cwd: Path) -> tuple[bool, str]:
    result = subprocess.run(
        command, cwd=cwd, capture_output=True, text=True, check=False
    )
    return result.returncode == 0, (result.stdout + result.stderr)


def smoke_one(project: str, config: dict, keep_output: bool) -> tuple[bool, float, str]:
    project_dir = REPO_ROOT / project
    out_dir = Path(tempfile.mkdtemp(prefix=f"smoke_{project}_"))
    started = time.time()
    try:
        ok, log = run(
            [
                sys.executable,
                config.get("train_script", "train.py"),
                "--smoke-test",
                "--out-dir",
                str(out_dir),
                *config["train"],
            ],
            project_dir,
        )
        if not ok:
            return False, time.time() - started, log

        # A project with more than one training stage (VQ-VAE's prior, latent
        # diffusion's second stage) lists the follow-ups under "extra".
        for extra in config.get("extra", []):
            produced = next(out_dir.rglob("*.pt"), None)
            if produced is None:
                return False, time.time() - started, f"nothing written under {out_dir}"
            command = [
                sys.executable,
                *[
                    part.format(checkpoint=str(produced), out_dir=str(out_dir))
                    for part in extra
                ],
                "--smoke-test",
            ]
            ok, log = run(command, project_dir)
            if not ok:
                return False, time.time() - started, log

        checkpoint = next(
            (path for path in out_dir.rglob(config.get("checkpoint_name", "best.pt"))), None
        )
        if checkpoint is None:
            return False, time.time() - started, f"no checkpoint written under {out_dir}"

        ok, log = run(
            [
                sys.executable,
                "evaluate.py",
                "--smoke-test",
                "--checkpoint",
                str(checkpoint),
                *config.get("evaluate", []),
            ],
            project_dir,
        )
        if not ok:
            return False, time.time() - started, log

        figures = sorted(path.name for path in checkpoint.parent.glob("*.png"))
        if not figures:
            return False, time.time() - started, "evaluate.py wrote no figures"

        # The comparison scripts have their own argument handling worth checking.
        for study in config["studies"]:
            command = [sys.executable, *study, "--smoke-test"]
            if study[0] in {"sweep.py", "train.py"}:
                command += ["--out-dir", str(out_dir / "study")]
            ok, log = run(command, project_dir)
            if not ok:
                return False, time.time() - started, log
            figures.append(f"{study[0]} ok")

        return True, time.time() - started, ", ".join(figures)
    finally:
        if not keep_output:
            shutil.rmtree(out_dir, ignore_errors=True)
            shutil.rmtree(project_dir / SMOKE_OUTPUT_DIR, ignore_errors=True)
            # Leave no empty outputs/ behind if the project had none before.
            outputs = project_dir / "outputs"
            if outputs.is_dir() and not any(outputs.iterdir()):
                outputs.rmdir()


def main() -> int:
    parser = argparse.ArgumentParser(description="Smoke-test every project.")
    parser.add_argument("--projects", nargs="+", default=sorted(PROJECTS), choices=sorted(PROJECTS))
    parser.add_argument("--keep-output", action="store_true", help="do not delete the temp dirs")
    parser.add_argument("--skip-metrics", action="store_true", help="skip the metric self-test")
    args = parser.parse_args()

    failures = []

    # The metrics come first. This script only checks that the pipelines run; if
    # FID or mean IoU or BLEU is wrong, every pipeline still runs and every number
    # they report is meaningless, so the metrics are checked against known answers
    # before anything else is trusted.
    if not args.skip_metrics:
        print(f"{'metric self-test':26s} ... ", end="", flush=True)
        started = time.time()
        ok, log = run([sys.executable, "scripts/metric_selftest.py"], REPO_ROOT)
        elapsed = time.time() - started
        if ok:
            passed = [line for line in log.splitlines() if line.startswith("all ")]
            print(f"ok ({elapsed:.1f}s)  [{passed[-1] if passed else 'passed'}]")
        else:
            print(f"FAILED ({elapsed:.1f}s)")
            failures.append(("metric self-test", log))

    for project in args.projects:
        print(f"{project:26s} ... ", end="", flush=True)
        ok, elapsed, detail = smoke_one(project, PROJECTS[project], args.keep_output)
        if ok:
            print(f"ok ({elapsed:.1f}s)  [{detail}]")
        else:
            print(f"FAILED ({elapsed:.1f}s)")
            failures.append((project, detail))

    total = len(args.projects) + (0 if args.skip_metrics else 1)
    if failures:
        for name, detail in failures:
            print(f"\n===== {name} =====\n{detail.strip()[-3000:]}")
        print(f"\n{len(failures)} of {total} checks failed")
        return 1

    print(f"\nall {total} checks passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Run every project's train -> evaluate pipeline on random tensors.

    python scripts/smoke_test.py
    python scripts/smoke_test.py --projects 01_classic_classifier 04_autoencoder

No dataset is downloaded and no pretrained weight is fetched: each project's
``--smoke-test`` flag swaps in a tiny synthetic split. This checks that the
scripts import, that shapes line up end to end, that a checkpoint round-trips
through ``evaluate.py`` and that every figure is written. It says nothing about
whether the models learn anything - that is what the real runs are for.

Finishes in well under a minute on a laptop CPU.
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
                "train.py",
                "--smoke-test",
                "--out-dir",
                str(out_dir),
                *config["train"],
            ],
            project_dir,
        )
        if not ok:
            return False, time.time() - started, log

        checkpoint = next(out_dir.rglob("best.pt"), None)
        if checkpoint is None:
            return False, time.time() - started, f"no checkpoint written under {out_dir}"

        ok, log = run(
            [sys.executable, "evaluate.py", "--smoke-test", "--checkpoint", str(checkpoint)],
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
    args = parser.parse_args()

    failures = []
    for project in args.projects:
        print(f"{project:26s} ... ", end="", flush=True)
        ok, elapsed, detail = smoke_one(project, PROJECTS[project], args.keep_output)
        if ok:
            print(f"ok ({elapsed:.1f}s)  [{detail}]")
        else:
            print(f"FAILED ({elapsed:.1f}s)")
            failures.append((project, detail))

    if failures:
        for project, detail in failures:
            print(f"\n===== {project} =====\n{detail.strip()[-3000:]}")
        print(f"\n{len(failures)} of {len(args.projects)} projects failed")
        return 1

    print(f"\nall {len(args.projects)} projects passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

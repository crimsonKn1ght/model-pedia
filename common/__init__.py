"""Shared utilities for the model-pedia generative model projects.

Importing this package from a project folder works because every project
script calls :func:`common.bootstrap.add_repo_root` (or simply lives inside a
repository checkout that is on ``sys.path``).
"""

from common import data, metrics, utils, viz

__all__ = ["data", "metrics", "utils", "viz"]

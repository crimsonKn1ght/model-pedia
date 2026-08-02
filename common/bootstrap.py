"""Makes ``import common`` work when a script is run from inside a project folder.

Every project script starts with::

    import bootstrap  # noqa: F401  (adds the repository root to sys.path)

Each project folder contains a one-line ``bootstrap.py`` that imports this
module, so the repository root ends up on ``sys.path`` no matter which
directory the script is launched from.
"""

import sys
from pathlib import Path


def add_repo_root() -> Path:
    """Insert the repository root into ``sys.path`` and return it."""
    root = Path(__file__).resolve().parent.parent
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    return root


REPO_ROOT = add_repo_root()

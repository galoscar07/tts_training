"""External path resolution.

`tts_training` keeps **data outside the module**: corpora and output manifests
live wherever the caller points, not inside the package. This makes the
module portable (it can move to its own repo) and keeps large datasets out of
the code tree.

Resolution order for the datasets directory:
  1. an explicit argument (CLI `--datasets-dir`, or the `datasets_dir` param);
  2. the `TTS_DATASETS_DIR` environment variable;
  3. this repo's `datasets/` directory, if it exists (dev convenience).

If none resolve, callers must pass an explicit `corpus_root`.
"""

from __future__ import annotations

import os
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
_ENV_VAR = "TTS_DATASETS_DIR"


def datasets_dir(explicit: str | os.PathLike | None = None) -> Path | None:
    """The parent directory that holds corpora (each in a subdir named by the
    dataset key). Returns None if nothing resolves and no explicit value was
    given."""
    if explicit is not None:
        return Path(explicit)
    env = os.environ.get(_ENV_VAR)
    if env:
        return Path(env)
    repo_datasets = _REPO_ROOT / "datasets"
    if repo_datasets.exists():
        return repo_datasets
    return None


def corpus_root(dataset: str, *, datasets_dir_override: str | os.PathLike | None = None,
                corpus_root_override: str | os.PathLike | None = None) -> Path:
    """Resolve where a specific corpus lives. An explicit `corpus_root` wins;
    otherwise it's `<datasets_dir>/<dataset>`."""
    if corpus_root_override is not None:
        return Path(corpus_root_override)
    base = datasets_dir(datasets_dir_override)
    if base is None:
        raise FileNotFoundError(
            f"cannot locate corpus {dataset!r}: no --corpus-root given, "
            f"${_ENV_VAR} unset, and no ./datasets/ in the repo."
        )
    return base / dataset

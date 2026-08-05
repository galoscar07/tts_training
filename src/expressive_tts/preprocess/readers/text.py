"""Plain-text input reading: `--text`, `--stdin`, `--input-file`.

See readme.md section 1.2. CSV/TSV/JSON/directory/manifest input modes are
not implemented yet.
"""

from __future__ import annotations

import sys
from pathlib import Path


def read_input_text(
    *,
    text: str | None = None,
    stdin: bool = False,
    input_file: str | Path | None = None,
) -> str:
    """Return the input text from exactly one of the supported sources."""
    sources = [source for source in (text is not None, stdin, input_file is not None) if source]
    if len(sources) == 0:
        raise ValueError("no input source given: pass one of text, stdin, input_file")
    if len(sources) > 1:
        raise ValueError("multiple input sources given: pass exactly one of text, stdin, input_file")

    if text is not None:
        return text
    if stdin:
        return sys.stdin.read()
    return Path(input_file).read_text(encoding="utf-8")

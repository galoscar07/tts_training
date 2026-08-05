"""One-time setup: download the Stanza Romanian model (tokenize, pos, lemma,
depparse). Requires network access; ~217MB.

Stanza is the linguistic-layer *fallback* backend (see
`expressive_tts.preprocess.linguistic`): used automatically when the
preferred transformer backend, Trankit, can't load — notably under Python
3.13, where trankit 1.1.1 crashes on import. Run this on such interpreters;
run `download_trankit_model.py` instead on Python <=3.12 for the transformer
backend.
"""

from __future__ import annotations

import stanza


def main() -> None:
    stanza.download("ro", processors="tokenize,pos,lemma,depparse")
    print("Romanian Stanza model downloaded.")


if __name__ == "__main__":
    main()

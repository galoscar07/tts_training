"""One-time setup: download the Trankit Romanian pipeline (tokenize, POS,
morphology, lemma, dependency parsing), built on XLM-RoBERTa via HuggingFace
`transformers`. Requires network access; downloads the shared XLM-R encoder
(~1.1GB) plus the Romanian adapter. Run once before using the
`linguistic`/`tokens` layers (and, transitively, `phonemes`/`syllables`/
`lexical_stress`, which depend on tokens).

Models are cached under `.cache/models/trankit` so later runs are offline.
"""

from __future__ import annotations

from expressive_tts.preprocess.linguistic import LANGUAGE, MODEL_CACHE_DIR


def main() -> None:
    from trankit import Pipeline

    MODEL_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    # Constructing the Pipeline triggers the download of the XLM-R encoder
    # and the Romanian adapter into the cache directory.
    Pipeline(LANGUAGE, cache_dir=str(MODEL_CACHE_DIR))
    print(f"Romanian Trankit model downloaded to {MODEL_CACHE_DIR}.")


if __name__ == "__main__":
    main()

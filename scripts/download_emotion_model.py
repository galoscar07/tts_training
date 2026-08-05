"""One-time setup: download the transformer emotion model
(`tabularisai/multilingual-emotion-classification`, a multilingual
XLM-RoBERTa classifier) into the local cache so the emotion layer runs
offline afterwards. Requires network access on the first run (~1.1GB).

Cached under `.cache/models/emotion_transformer`.
"""

from __future__ import annotations

from expressive_tts.preprocess.emotion import MODEL_CACHE_DIR, MODEL_ID


def main() -> None:
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    MODEL_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    AutoTokenizer.from_pretrained(MODEL_ID, cache_dir=str(MODEL_CACHE_DIR))
    AutoModelForSequenceClassification.from_pretrained(MODEL_ID, cache_dir=str(MODEL_CACHE_DIR))
    print(f"Emotion model '{MODEL_ID}' downloaded to {MODEL_CACHE_DIR}.")


if __name__ == "__main__":
    main()

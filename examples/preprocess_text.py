"""Minimal example of the Python API. See readme.md section 1.8.

The "default" profile now includes linguistic analysis, phonemes, stress,
and emotion — run these one-time setup steps first:
    ./.venv/bin/pip install -e ".[dev,linguistic,ai]"
    ./.venv/bin/python scripts/download_trankit_model.py   # POS/lemma/deps (XLM-R)
    ./.venv/bin/python scripts/download_emotion_model.py    # transformer emotion
    ./.venv/bin/python scripts/fetch_emotion_lexicon.py     # NRC lexicon (focus layer)
Also requires the `espeak` binary on PATH.

Run with: ./.venv/bin/python examples/preprocess_text.py
"""

from expressive_tts.preprocess import PreprocessPipeline


def main() -> None:
    pipeline = PreprocessPipeline.from_profile("default")

    result = pipeline.process(
        "Dr. Popescu a trimis 25 kg pe 12.07.2026, la ora 14:30.",
        id="example-001",
    )

    print("original:  ", result.original_text)
    print("clean:     ", result.clean_text)
    print("normalized:", result.normalized_text)
    print()
    print(f"{len(result.trace)} normalization(s) applied:")
    for entry in result.trace:
        print(f"  [{entry.operation}] {entry.original!r} -> {entry.replacement!r}")

    print()
    emotion_result = pipeline.process("Uau! Chiar am reușit?", id="example-002")
    sentence = emotion_result.sentences[0]
    print("sentence_type:", sentence.sentence_type)
    emotion = sentence.emotion
    print(f"emotion: {emotion.label} (confidence={emotion.confidence}, intensity={emotion.intensity})")
    for evidence in emotion.evidence:
        print(f"  evidence: {evidence.span!r} -> {evidence.rule}")


if __name__ == "__main__":
    main()

"""Preprocess a single Romanian sentence and print the result.

Usage:
    ./.venv/bin/python examples/try_sentence.py "Sunt foarte fericit că am reușit!"

If no sentence is given on the command line, a default one is used. Requires
the linguistic backend (Stanza or Trankit) and the transformer emotion model
to be installed/downloaded — see src/expressive_tts/preprocess/README.md.
"""

import sys

from expressive_tts.preprocess import linguistic
from expressive_tts.preprocess.pipeline import PreprocessPipeline

DEFAULT = "Sunt foarte fericit că am reușit în sfârșit!"


def main() -> None:
    text = " ".join(sys.argv[1:]).strip() or DEFAULT
    pipeline = PreprocessPipeline.from_profile("expressive")
    result = pipeline.process(text)

    print(f"POS/linguistic backend: {linguistic.active_backend()}")
    print(f"input:      {text}")
    print(f"normalized: {result.normalized_text}")
    print()

    for i, sentence in enumerate(result.sentences):
        print(f"sentence {i}: {sentence.text!r}")
        print(f"  type: {sentence.sentence_type} | negated: {sentence.is_negated}")

        emotion = sentence.emotion
        if emotion is not None:
            print(
                f"  emotion: {emotion.label} "
                f"(confidence={emotion.confidence}, intensity={emotion.intensity}, "
                f"valence={emotion.valence}, arousal={emotion.arousal})"
            )
            top = sorted(emotion.distribution.items(), key=lambda kv: -kv[1])[:3]
            print("  top emotions: " + ", ".join(f"{k}={v}" for k, v in top))

        print("  tokens (text/UPOS/lemma):")
        for token in sentence.tokens:
            print(f"    {token.text:<16} {str(token.upos):<6} {token.lemma}")
        print()


if __name__ == "__main__":
    main()

"""Behavior sanity check for the Phase 11 context processor against
data/evaluation/context.jsonl.

NOT a macro-F1 evaluation, deliberately — none of `context.jsonl`'s
paragraphs (3 hand-curated + 40 programmatically added for eval-set
scaling) carry per-sentence gold emotion labels, and hand-labeling them
after the fact just to manufacture an F1 number would be a fragile,
non-authoritative metric dressed up as real evaluation (objectives.md's
own acceptance criterion "context smoothing must improve or preserve
macro F1" is left honestly unchecked for this reason — see
preprocess/objectives.md Phase 11 and the Phase 9 precedent for skipping a
metric when no real reference exists, `data/evaluation/README.md`).

What this actually checks, over every real paragraph in context.jsonl:
- the schema guarantee that the local prediction is never discarded;
- how often context actually changes the label, and why, printed for
  manual inspection;
- no exceptions.

Usage:
    ./.venv/bin/python -m objective_evaluation.evaluate_context
"""

from __future__ import annotations

from pathlib import Path

from expressive_tts.preprocess.pipeline import PreprocessPipeline
from objective_evaluation.schemas import ContextParagraph

DATA_EVALUATION = Path(__file__).resolve().parents[2] / "data" / "evaluation"


def load_paragraphs() -> list[ContextParagraph]:
    return [
        ContextParagraph.model_validate_json(line)
        for line in (DATA_EVALUATION / "context.jsonl").open(encoding="utf-8")
    ]


def main() -> None:
    paragraphs = load_paragraphs()
    pipeline = PreprocessPipeline()

    total_sentences = 0
    changed_count = 0

    print(f"Phase 11 context behavior sanity check — {len(paragraphs)} real paragraphs\n")

    for paragraph in paragraphs:
        text = " ".join(paragraph.sentences)
        result = pipeline.process(text, include={"emotion", "context"})
        print(f"{paragraph.id} ({paragraph.text_register}, {len(result.sentences)} sentences)")

        for sentence in result.sentences:
            total_sentences += 1
            assert sentence.emotion is not None, "local prediction was not computed — schema violation"
            assert sentence.context_emotion is not None, "context layer did not run — schema violation"

            local = sentence.emotion
            context = sentence.context_emotion
            if context.changed:
                changed_count += 1
                print(
                    f"  CHANGED: local={local.label}(conf={local.confidence:.2f}) -> "
                    f"context={context.label}(conf={context.confidence:.2f}) | reason={context.reason}"
                )
            else:
                print(f"  unchanged: {local.label}(conf={local.confidence:.2f})")
        print()

    print(f"Local prediction preserved for all {total_sentences} sentences (schema-guaranteed, verified).")
    print(f"Context changed {changed_count}/{total_sentences} predictions ({changed_count / total_sentences:.1%}).")
    print()
    print(
        f"This is a behavior sanity check on {len(paragraphs)} real paragraphs, not a "
        "macro-F1 measurement — see this script's module docstring for why."
    )


if __name__ == "__main__":
    main()

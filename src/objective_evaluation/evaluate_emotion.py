"""Evaluate the Phase 6 rule-based emotion baseline against the real,
gold-labeled subset of data/evaluation/{dev,test}.jsonl.

Per preprocess/objectives.md's acceptance criterion for this phase
("Confidence threshold is selected only on the development set"): inspect
`--split dev` output while tuning thresholds in preprocess/emotion.py, then
report `--split test` once, unchanged.

Usage:
    ./.venv/bin/python -m objective_evaluation.evaluate_emotion --split dev
    ./.venv/bin/python -m objective_evaluation.evaluate_emotion --split test
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from expressive_tts.preprocess.pipeline import PreprocessPipeline
from objective_evaluation.schemas import EvaluationExample

DATA_EVALUATION = Path(__file__).resolve().parents[2] / "data" / "evaluation"


def load_gold(split: str) -> list[EvaluationExample]:
    rows = [
        EvaluationExample.model_validate_json(line)
        for line in (DATA_EVALUATION / f"{split}.jsonl").open(encoding="utf-8")
    ]
    return [
        row
        for row in rows
        if row.emotion.provenance.value == "source" and "multi_sentence" not in row.phenomena
    ]


def predict(pipeline: PreprocessPipeline, text: str) -> str:
    result = pipeline.process(text, include={"emotion"})
    return result.sentences[0].emotion.label


def macro_f1(confusion: dict[str, Counter], labels: list[str]) -> tuple[float, dict[str, float]]:
    per_class = {}
    for label in labels:
        tp = confusion[label][label]
        fp = sum(confusion[other][label] for other in labels if other != label)
        fn = sum(confusion[label][other] for other in labels if other != label)
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
        per_class[label] = f1
    macro = sum(per_class.values()) / len(per_class) if per_class else 0.0
    return macro, per_class


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", choices=["dev", "test"], default="dev")
    args = parser.parse_args()

    gold_examples = load_gold(args.split)
    pipeline = PreprocessPipeline()

    predictions = []
    for example in gold_examples:
        predicted = predict(pipeline, example.text)
        predictions.append((example, predicted))

    labels = sorted({e.emotion.value for e, _ in predictions} | {p for _, p in predictions})
    confusion: dict[str, Counter] = {label: Counter() for label in labels}
    correct = 0
    covered = 0
    for example, predicted in predictions:
        gold = example.emotion.value
        confusion[gold][predicted] += 1
        if predicted == gold:
            correct += 1
        if predicted != "unspecified":
            covered += 1

    n = len(predictions)
    accuracy = correct / n if n else 0.0
    coverage = covered / n if n else 0.0
    macro, per_class = macro_f1(confusion, labels)

    from expressive_tts.preprocess.emotion import MODEL_ID, PRODUCER

    print("Component: Transformer emotion classifier (Phase 6/7)")
    print("Version:", PRODUCER, f"({MODEL_ID})")
    print("Evaluation dataset:", f"data/evaluation/{args.split}.jsonl (gold, single-sentence subset)")
    print("Number of examples:", n)
    print("Primary metric: macro F1")
    print(f"Primary result: {macro:.3f}")
    print(f"Accuracy: {accuracy:.3f}")
    print(f"Coverage (not 'unspecified'): {coverage:.3f}")
    print()
    print("Per-category results (F1):")
    for label in labels:
        print(f"  {label:12s} {per_class[label]:.3f}  (n_gold={sum(confusion[label].values())})")
    print()
    print("Confusion matrix (rows=gold, cols=predicted):")
    header = "gold\\pred".ljust(12) + "".join(label[:8].ljust(10) for label in labels)
    print(header)
    for gold_label in labels:
        row = gold_label.ljust(12) + "".join(str(confusion[gold_label][pred]).ljust(10) for pred in labels)
        print(row)
    print()
    print("Known limitations:")
    print("  - Only", n, "gold-labeled single-sentence examples available this pass")
    print("  - No gold valence/arousal/intensity in the evaluation set, so those aren't scored")
    print("  - NRC EmoLex Romanian translations have some noise (e.g. 'trist' also tagged 'anger')")
    print("Decision: baseline established; revisit after scaling data/evaluation/ to full size")

    for example, predicted in predictions:
        if predicted != example.emotion.value:
            print(f"  MISS {example.id}: gold={example.emotion.value} pred={predicted} | {example.text}")


if __name__ == "__main__":
    main()

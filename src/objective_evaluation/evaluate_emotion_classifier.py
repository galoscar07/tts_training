"""Evaluate the Phase 7 trained emotion classifier against the rule
baseline (preprocess/objectives.md Phase 7 acceptance criteria: must
outperform the rule baseline in macro F1; must run locally; must not be
adopted if improvement is negligible or confidence is poorly calibrated).

Evaluated on the classifier's own held-out dev split
(`.cache/models/emotion_classifier/dev_split.json`, produced by
`objective_evaluation.train_emotion_classifier` — real REDv2 gold-labeled
tweets, never seen during training, and already excluded from
`data/evaluation/{dev,test}.jsonl`). Not evaluated against
`data/evaluation/test.jsonl` directly: that set mixes in RONEC/Tier-2 rows
the classifier was never trained to handle (no emotion label at all), so
scoring against it would conflate "doesn't know this domain" with
"worse at emotion classification" — the held-out REDv2 dev split is the
fair like-for-like comparison against the rule baseline, which was itself
built and tuned on this kind of text.

Usage:
    ./.venv/bin/python -m objective_evaluation.evaluate_emotion_classifier
"""

from __future__ import annotations

import json
from pathlib import Path
import time

import joblib

from expressive_tts.preprocess.pipeline import PreprocessPipeline
from objective_evaluation.train_emotion_classifier import build_features

ROOT = Path(__file__).resolve().parents[2]
MODEL_DIR = ROOT / ".cache" / "models" / "emotion_classifier"


def macro_f1(y_true: list[str], y_pred: list[str], labels: list[str]) -> tuple[float, dict[str, float]]:
    per_class = {}
    for label in labels:
        tp = sum(1 for t, p in zip(y_true, y_pred) if t == label and p == label)
        fp = sum(1 for t, p in zip(y_true, y_pred) if t != label and p == label)
        fn = sum(1 for t, p in zip(y_true, y_pred) if t == label and p != label)
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        per_class[label] = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return sum(per_class.values()) / len(per_class), per_class


def brier_score(y_true: list[str], probabilities: list[dict[str, float]], labels: list[str]) -> float:
    total = 0.0
    for true_label, probs in zip(y_true, probabilities):
        total += sum((probs.get(label, 0.0) - (1.0 if label == true_label else 0.0)) ** 2 for label in labels)
    return total / len(y_true)


def main() -> None:
    bundle = joblib.load(MODEL_DIR / "model.joblib")
    classifier, vectorizer, labels = bundle["classifier"], bundle["vectorizer"], bundle["labels"]
    dev_rows = json.loads((MODEL_DIR / "dev_split.json").read_text(encoding="utf-8"))

    texts = [r["text"] for r in dev_rows]
    gold = [r["label"] for r in dev_rows]

    pipeline = PreprocessPipeline()

    print(f"Evaluating on {len(texts)} held-out examples, labels={labels}")
    if bundle.get("dropped_labels"):
        print(f"Note: {bundle['dropped_labels']} excluded from training (too few examples after eval-set scaling)")
    print()

    # --- rule baseline ------------------------------------------------------
    rule_pred = []
    for text in texts:
        result = pipeline.process(text, include={"emotion"})
        rule_pred.append(result.sentences[0].emotion.label if result.sentences else "unspecified")
    rule_macro_f1, rule_per_class = macro_f1(gold, rule_pred, labels)

    # --- trained classifier --------------------------------------------------
    X = build_features(pipeline, texts, vectorizer, fit=False)
    start = time.perf_counter()
    clf_pred = list(classifier.predict(X))
    clf_elapsed_ms = (time.perf_counter() - start) * 1000
    clf_proba = classifier.predict_proba(X)
    clf_probabilities = [dict(zip(classifier.classes_, row)) for row in clf_proba]
    clf_macro_f1, clf_per_class = macro_f1(gold, clf_pred, labels)
    brier = brier_score(gold, clf_probabilities, labels)

    print("Rule baseline:")
    print(f"  macro F1: {rule_macro_f1:.3f}")
    for label in labels:
        print(f"    {label}: {rule_per_class[label]:.3f}")
    print()
    print("Trained classifier:")
    print(f"  macro F1: {clf_macro_f1:.3f}")
    for label in labels:
        print(f"    {label}: {clf_per_class[label]:.3f}")
    print(f"  Brier score (multiclass, lower is better, 0=perfect): {brier:.3f}")
    print(f"  Inference time for {len(texts)} examples: {clf_elapsed_ms:.1f} ms total, "
          f"{clf_elapsed_ms / len(texts):.2f} ms/example")
    print()

    improvement = clf_macro_f1 - rule_macro_f1
    print(f"Improvement over rule baseline: {improvement:+.3f} macro F1")

    # objectives.md: "must not be adopted if improvement is negligible or
    # confidence is poorly calibrated" — a real decision, not assumed.
    NEGLIGIBLE_IMPROVEMENT = 0.03
    POOR_CALIBRATION_BRIER = 0.5  # a multiclass Brier this high means predicted probabilities are unreliable
    if improvement < NEGLIGIBLE_IMPROVEMENT:
        decision = "NOT ADOPTED — improvement over the rule baseline is negligible or negative."
    elif brier > POOR_CALIBRATION_BRIER:
        decision = "NOT ADOPTED — macro F1 improved but confidence is poorly calibrated (high Brier score)."
    else:
        decision = "CLEARS THE BAR on this held-out split — see report before wiring into the pipeline."
    print(f"\nDecision: {decision}")


if __name__ == "__main__":
    main()

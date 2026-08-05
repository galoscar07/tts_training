"""Train the optional Phase 7 emotion classifier (preprocess/objectives.md
Phase 7: "Determine whether a locally runnable trained model improves on
the rule-based baseline").

Per the phase's own recommendation ("do not train a language model from
scratch"; classical classifiers preferred unless the dataset is large
enough for a transformer), and given the realistic scale here even after
eval-set scaling (a few hundred gold examples across 6 classes — not
transformer-scale), this trains a scikit-learn logistic regression over:
- the existing rule baseline's own per-category evidence weights
  (`EmotionAnnotation.distribution`), valence, arousal (features the rule
  engine already computes — reusing them, not re-deriving from scratch);
- a negation flag and one-hot sentence type;
- word 1-2gram TF-IDF over the raw text.

**Training-pool construction, to keep the July test set completely held
out** (objectives.md's explicit requirement): starts from
`data/external/redv2_sample.json` (the full filtered REDv2 candidate
pool, 354 rows) and excludes every text already used anywhere in
`data/evaluation/dev.jsonl`/`test.jsonl` — not just `test.jsonl` — so
scoring `objective_evaluation.evaluate_emotion_classifier` against either split later is a
genuine held-out comparison, not partial leakage. This shrinks the
trainable pool considerably (see the printed pool size below) — an
honest cost of not touching the evaluation set.

Requires the `ml` extra: `pip install -e ".[ml]"`.

Usage:
    ./.venv/bin/python -m objective_evaluation.train_emotion_classifier
"""

from __future__ import annotations

import json
import random
from pathlib import Path

from scipy.sparse import csr_matrix, hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

from expressive_tts.preprocess.pipeline import PreprocessPipeline
from objective_evaluation.schemas import EvaluationExample
from expressive_tts.preprocess.schemas import SENTENCE_TYPES

ROOT = Path(__file__).resolve().parents[2]
DATA_EXTERNAL = ROOT / "data" / "external"
DATA_EVALUATION = ROOT / "data" / "evaluation"
MODEL_DIR = ROOT / ".cache" / "models" / "emotion_classifier"

LABELS = ["angry", "fear", "happy", "neutral", "sad", "surprise"]  # excludes "unspecified" — not a training target
SENTENCE_TYPE_LIST = sorted(SENTENCE_TYPES)
SEED = 20260725
TRAIN_RATIO = 0.8


def _normalize(text: str) -> str:
    return " ".join(text.split()).strip().casefold()


def _held_out_texts() -> set[str]:
    texts = set()
    for split in ("dev", "test"):
        for line in (DATA_EVALUATION / f"{split}.jsonl").open(encoding="utf-8"):
            texts.add(_normalize(EvaluationExample.model_validate_json(line).text))
    return texts


def _load_pool() -> list[dict]:
    held_out = _held_out_texts()
    rows = json.loads((DATA_EXTERNAL / "redv2_sample.json").read_text(encoding="utf-8"))
    seen = set()
    pool = []
    for row in rows:
        key = _normalize(row["text"])
        if key in held_out or key in seen:
            continue
        seen.add(key)
        pool.append(row)
    return pool


def _rule_features(pipeline: PreprocessPipeline, text: str) -> list[float]:
    result = pipeline.process(text, include={"emotion"})
    sentence = result.sentences[0] if result.sentences else None
    emotion = sentence.emotion if sentence else None

    distribution = emotion.distribution if emotion else {}
    dist_features = [distribution.get(label, 0.0) for label in LABELS if label != "neutral"]  # dist has no "neutral" key
    valence = emotion.valence if emotion and emotion.valence is not None else 0.5
    arousal = emotion.arousal if emotion and emotion.arousal is not None else 0.5
    negation = 1.0 if sentence and any(t.feats.get("Polarity") == "Neg" for t in sentence.tokens) else 0.0
    sentence_type = sentence.sentence_type if sentence else None
    type_onehot = [1.0 if sentence_type == t else 0.0 for t in SENTENCE_TYPE_LIST]

    return dist_features + [valence, arousal, negation] + type_onehot


def build_features(pipeline: PreprocessPipeline, texts: list[str], vectorizer: TfidfVectorizer, *, fit: bool):
    rule_matrix = csr_matrix([_rule_features(pipeline, t) for t in texts])
    tfidf_matrix = vectorizer.fit_transform(texts) if fit else vectorizer.transform(texts)
    return hstack([rule_matrix, tfidf_matrix]).tocsr()


def main() -> None:
    pool = _load_pool()
    print(f"Trainable pool (REDv2, held-out-from-eval-set): {len(pool)} examples")

    by_label: dict[str, list[dict]] = {}
    for row in pool:
        by_label.setdefault(row["emotion"], []).append(row)

    rng = random.Random(SEED)
    train_rows, dev_rows = [], []
    for label in LABELS:
        rows = by_label.get(label, [])
        rng.shuffle(rows)
        split_point = round(len(rows) * TRAIN_RATIO)
        train_rows.extend(rows[:split_point])
        dev_rows.extend(rows[split_point:])
    rng.shuffle(train_rows)
    rng.shuffle(dev_rows)

    train_counts = {label: sum(1 for r in train_rows if r["emotion"] == label) for label in LABELS}
    print(f"Train: {len(train_rows)}  Dev (held out from training, used only to select the classifier): {len(dev_rows)}")
    print("Per-label train counts:", train_counts)

    # A label can end up with zero (or too few) training rows once
    # eval-set scaling has already drawn most/all of its real candidates
    # into dev/test — "surprise" hits exactly this (only 28 REDv2
    # candidates existed for it total). Report honestly and drop it from
    # the trainable label set rather than silently training on nothing.
    MIN_TRAIN_PER_LABEL = 5
    trainable_labels = [label for label in LABELS if train_counts[label] >= MIN_TRAIN_PER_LABEL]
    dropped = [label for label in LABELS if label not in trainable_labels]
    if dropped:
        print(
            f"WARNING: dropping label(s) with <{MIN_TRAIN_PER_LABEL} training examples "
            f"(exhausted by eval-set scaling's own sampling): {dropped}"
        )
        train_rows = [r for r in train_rows if r["emotion"] in trainable_labels]
        dev_rows = [r for r in dev_rows if r["emotion"] in trainable_labels]

    pipeline = PreprocessPipeline()
    vectorizer = TfidfVectorizer(ngram_range=(1, 2), max_features=1500, lowercase=True)

    train_texts = [r["text"] for r in train_rows]
    train_labels = [r["emotion"] for r in train_rows]
    X_train = build_features(pipeline, train_texts, vectorizer, fit=True)

    classifier = LogisticRegression(max_iter=2000, class_weight="balanced")
    classifier.fit(X_train, train_labels)

    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    import joblib

    # joblib/pickle format — acceptable here because this cache is
    # process-local and gitignored, never distributed or loaded from an
    # untrusted source (the same environment that writes it is the only
    # one that ever reads it); not the "never unpickle untrusted data"
    # scenario that format would otherwise be a real risk for.
    joblib.dump(
        {
            "classifier": classifier,
            "vectorizer": vectorizer,
            "labels": trainable_labels,
            "dropped_labels": dropped,
            "sentence_type_list": SENTENCE_TYPE_LIST,
            "seed": SEED,
            "train_size": len(train_rows),
        },
        MODEL_DIR / "model.joblib",
    )

    dev_meta = [{"text": r["text"], "label": r["emotion"], "text_id": r["text_id"]} for r in dev_rows]
    (MODEL_DIR / "dev_split.json").write_text(json.dumps(dev_meta, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"Saved model + vectorizer to {MODEL_DIR / 'model.joblib'}")
    print(f"Saved held-out dev split ({len(dev_rows)} examples) to {MODEL_DIR / 'dev_split.json'}")
    print("Run ./.venv/bin/python -m objective_evaluation.evaluate_emotion_classifier next.")


if __name__ == "__main__":
    main()

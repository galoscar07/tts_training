"""Emotion fine-tuning — SCAFFOLD / DEFERRED.

This is intentionally not runnable yet. The blocker is data, not code: MARA,
SWARA, and Common Voice are all *neutral* read speech, so there is no
emotional Romanian audio to fine-tune the acoustic model on. Our preprocess
pipeline labels *text* emotion, which conditions synthesis, but the model
can only learn emotional *acoustics* from emotional *audio*.

What is already in place for when emotional audio exists:
  * `data.manifest --with-emotion` emits a per-utterance emotion label
    (4th manifest column), from `expressive_tts.preprocess`.
  * `data.formatter.coqui_formatter` surfaces it as `emotion_name` per sample.

Intended approach once emotional audio is available (pick one):
  1. **Emotion embedding** — add an emotion-id embedding alongside VITS's
     speaker embedding (a small change to `VitsArgs`/the model forward),
     initialise the base (neutral) checkpoint, and fine-tune on emotional
     audio with the emotion id from the manifest.
  2. **(speaker, emotion) speaker-slot hack** — treat each
     (speaker, emotion) pair as a distinct "speaker" id (no model change),
     fine-tune from the base checkpoint. Cheapest; conflates speaker and
     emotion capacity.

Neither is wired until there is emotional Romanian speech to train on. See
README.md.
"""

from __future__ import annotations


def main(argv: list[str] | None = None) -> None:  # pragma: no cover - deferred
    raise NotImplementedError(
        "Emotion fine-tuning is deferred: no emotional Romanian speech dataset "
        "is available yet (MARA/SWARA/Common Voice are neutral). See "
        "tts_training/finetune.py and README.md for the intended approach."
    )


if __name__ == "__main__":
    main()

"""Coqui `formatter` for the manifest produced by `manifest.py`.

A Coqui formatter takes `(root_path, meta_file)` and returns a list of sample
dicts. Our manifest rows are `audio_file|phonemes|speaker|emotion`, with
`audio_file` relative to `root_path` (the corpus dir the manifest was built
against). We hand Coqui the phoneme string as `text`; because
`use_phonemes=False`, Coqui tokenizes it directly against our symbol set.

Pure-Python, no Coqui import — but it's only *called* from the training code
on the GPU box.
"""

from __future__ import annotations

import os


def coqui_formatter(root_path: str, meta_file: str, **kwargs) -> list[dict]:
    items: list[dict] = []
    meta_path = os.path.join(root_path, meta_file) if not os.path.isabs(meta_file) else meta_file
    with open(meta_path, encoding="utf-8") as handle:
        for raw in handle:
            line = raw.rstrip("\n")
            if not line.strip():
                continue
            parts = line.split("|")
            if len(parts) < 3:
                continue
            audio_rel, phonemes, speaker = parts[0], parts[1], parts[2]
            emotion = parts[3] if len(parts) > 3 else ""
            items.append(
                {
                    "text": phonemes,
                    "audio_file": os.path.join(root_path, audio_rel),
                    "speaker_name": speaker,
                    "emotion_name": emotion or None,
                    "root_path": root_path,
                    "language": "ro",
                }
            )
    return items

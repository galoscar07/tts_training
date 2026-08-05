"""Romanian expressive VITS training.

A self-contained module (sibling to `expressive_tts` and
`objective_evaluation`) that trains a VITS acoustic model on Romanian speech,
using the `expressive_tts.preprocess` frontend for text→phoneme conversion.
It is deliberately separable so it can move to its own git repository later:
its only in-project dependency is the preprocess pipeline.

Plan (see README.md):
  1. base, multi-speaker, *neutral* VITS trained from scratch on MARA
     (+ SWARA / Common Voice when available);
  2. later, an emotion fine-tune conditioned on the frontend's emotion
     labels — scaffolded here, but deferred until emotional Romanian *speech*
     audio exists (MARA/SWARA/CV are all neutral read speech).

Layering / import safety:
  - `frontend` and `data.manifest` depend only on the preprocess pipeline and
    the standard library, so **data preparation runs locally (e.g. Apple M1)
    without Coqui TTS installed**.
  - `data.formatter`, `vits.config`, `train`, and `finetune` import Coqui TTS
    lazily (inside functions). They are meant to run on a GPU box; importing
    this package never requires `coqui-tts`.
"""

LANGUAGE = "ro"
SAMPLE_RATE = 22050  # datasets/mara wavs are 22.05 kHz mono 16-bit

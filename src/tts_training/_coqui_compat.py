"""Compatibility shim for importing coqui-tts against newer `transformers`.

`import TTS` eagerly imports XTTS/Tortoise, which reference helpers that recent
`transformers` releases removed/renamed (e.g. `isin_mps_friendly`, dropped in
transformers 5.x). We don't use XTTS, but the package-level import fails anyway.
Call `ensure_coqui_importable()` BEFORE importing anything from `TTS` to add the
missing symbols back.

Kept deliberately defensive: each shim is added only if absent, and any failure
here is swallowed so it never makes things worse than the original ImportError.
"""

from __future__ import annotations


def ensure_coqui_importable() -> None:
    try:
        import torch
        import transformers.pytorch_utils as pu
    except Exception:
        return

    # transformers >=5 removed this MPS fallback helper; on CPU/CUDA it's just
    # torch.isin.
    if not hasattr(pu, "isin_mps_friendly"):
        def isin_mps_friendly(elements, test_elements):  # noqa: ANN001
            return torch.isin(elements, test_elements)

        pu.isin_mps_friendly = isin_mps_friendly

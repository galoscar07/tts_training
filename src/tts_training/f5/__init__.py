"""F5-TTS integration (fine-tune on MARA+SWARA reusing our IPA phoneme
manifests).

F5-TTS is a *separate* framework (`f5-tts`), not Coqui — a flow-matching DiT
with its own trainer, data format, and vocoder. This subpackage only owns the
bridge: `prepare` converts the manifests we already built for VITS
(`audio|phonemes|speaker|emotion`) into F5's expected on-disk layout
(`wavs/` + `metadata.csv`), preserving our accented IPA phoneme text. From
there you run F5's own `prepare_csv_wavs` + finetune CLI (see README).

Note: F5 has no speaker embeddings — it's zero-shot / reference-conditioned.
The `speaker` column is ignored here; at inference you pick a voice by
providing a reference clip.
"""

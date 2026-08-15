# tts_training — Romanian expressive VITS

Trains a VITS acoustic model on Romanian speech, using the
`expressive_tts.preprocess` frontend for text → IPA phonemes. Self-contained:
its only in-project dependency is the preprocess pipeline (a package import,
not a repo-path dependency), and **all data lives outside the module** — so it
lifts cleanly into its own git repo.

## Plan

1. **Base model** — multi-speaker, **neutral** VITS trained from scratch on
   MARA (+ SWARA / Common Voice).
2. **Emotion fine-tune** — *deferred*. Conditions on the frontend's emotion
   labels, but needs emotional Romanian **speech** audio, which we don't have
   yet (MARA/SWARA/CV are neutral). Scaffolded in `finetune.py`.

## Data lives outside the module

Corpora and manifests are never stored in the package. Point the tools at them
via, in priority order:

1. `--corpus-root /path/to/corpus` (per-command), or
2. `$TTS_DATASETS_DIR` (parent dir holding `<dataset>/`), or
3. this repo's `datasets/` (dev convenience — MARA/HRIA are here).

Manifests are written wherever you pass `--out`.

## Supported corpora

| key | speakers | layout | notes |
|---|---|---|---|
| `mara` | 1 (`mara`) | LJSpeech `id\|text`, `wavs/` | on disk here |
| `hria` | 1 (`catalina`) | LJSpeech `id\|text`, `catalina/data/*.WAV` | on disk here |
| `swara` | many (dir name) | `<speaker>/*.wav` + sidecar `.txt` **or** central `id\|text` | bring your own `--corpus-root` |
| `common_voice` | many (`client_id`) | `validated.tsv` + `clips/*.mp3` | bring your own `--corpus-root`; mp3 48 kHz |

Adding another corpus: write a reader in `data/readers.py` and register it in
`data/manifest.py::DATASETS`.

## Two-machine workflow

Data prep runs anywhere (no Coqui). **Training needs a CUDA GPU** — VITS from
scratch is a multi-day job and is not expected to train on Apple M1.

### 1. Prepare manifests (local, e.g. M1)

```bash
# neutral base manifest (fast — phonemes only, no emotion model)
python -m tts_training.data.manifest --dataset mara --out out/mara.manifest

# fastest neutral path: normalization + stressed eSpeak IPA, no Stanza syntax
python -m tts_training.data.manifest --dataset mara --out out/mara.manifest --phonetics-only

# external corpora
python -m tts_training.data.manifest --dataset swara --corpus-root /data/SWARA --out out/swara.manifest
python -m tts_training.data.manifest --dataset swara_train --corpus-root /data/SWARA --out out/swara_train.manifest
python -m tts_training.data.manifest --dataset swara_test --corpus-root /data/SWARA --out out/swara_test.manifest
python -m tts_training.data.manifest --dataset common_voice --corpus-root /data/cv-ro --out out/cv.manifest

# quick smoke test
python -m tts_training.data.manifest --dataset mara --out out/mara_smoke.manifest --limit 20

# with per-utterance emotion labels (loads the transformer; deferred fine-tune)
python -m tts_training.data.manifest --dataset mara --out out/mara_emo.manifest --with-emotion
```

Manifest format (pipe-separated), `audio_file` relative to the corpus root:

```
wavs/mara_chp01_0002.wav|ˈa rəmˈas mˈaɾa , səɾˈaka , ...|mara|
```

`data.manifest` reports rows skipped (missing wav / empty text / empty
phonemes) and warns if any phoneme symbol falls outside the canonical
inventory (`configs/preprocess/phoneme_inventory.yaml`).

### 2. Train (GPU box)

```bash
pip install -e ".[training]"          # coqui-tts + torch (CUDA)
python -m tts_training.train \
    --manifest out/mara.manifest --corpus-root /data/mara \
    --output out/vits_ro_base
```

Multiple corpora → repeat `--manifest`/`--corpus-root` (paired); all speakers
share one multi-speaker model.

## How the frontend plugs into Coqui

- `frontend/symbols.py` builds Coqui's `characters` vocab from the phoneme
  inventory (via `expressive_tts.preprocess.phonemizer.phoneme_inventory()`).
  Every symbol is one codepoint (affricates are sequences like `tʃ`/`dʒ`), so
  the default per-character tokenizer gives one id per phoneme. Order is
  deterministic (`sorted`) → stable ids.
- `use_phonemes=False`: the manifest text is already phonemized.
- `data/formatter.py` is the Coqui formatter that reads our manifest.
- `vits/config.py` assembles the `VitsConfig` (22.05 kHz, multi-speaker).

## Files

| Path | Needs Coqui? | Purpose |
|---|---|---|
| `paths.py` | no | resolve external corpus/manifest locations |
| `frontend/symbols.py` | only `characters_config()` | phoneme symbol set / Coqui characters |
| `data/readers.py` | no | per-corpus transcript readers (MARA/HRIA/SWARA/CV) |
| `data/manifest.py` | **no** | build manifest via the preprocess pipeline |
| `data/formatter.py` | called from training | Coqui formatter for our manifest |
| `vits/config.py` | yes (lazy) | base VITS config |
| `train.py` | yes (lazy) | base training entry point |
| `finetune.py` | — | emotion fine-tune scaffold (deferred) |
| `synthesize.py` | yes (lazy) | inference: text → phonemes → VITS → wav, with `--postprocess` |
| `postprocess.py` | **no** | audio realism filter chain (numpy/scipy) |

## Synthesis & post-processing

Inference mirrors training: text is phonemized by the frontend first, then fed
to VITS (the model expects phonemes, `use_phonemes=False`). `--postprocess`
adds an optional audio pass **after** synthesis to make the output sound less
synthetic.

```bash
python -m tts_training.synthesize \
    --checkpoint out/vits_ro_base/<run>/best_model.pth \
    --config     out/vits_ro_base/<run>/config.json \
    --text "Bună ziua, acesta este un test." \
    --speaker mara \
    --out sample.wav \
    --postprocess
```

The post-process chain (`postprocess.PostProcessConfig`), in order:
high-pass (DC/rumble removal) → trim silence → light room reverb → faint
room-tone noise floor → loudness normalization (EBU R128) → peak limiting.
Reverb/room-tone are seeded, so a given input is reproducible. Tune or disable
any stage via `PostProcessConfig`. It's pure numpy/scipy, so you can also
filter existing wavs without a model:

```python
from tts_training.synthesize import postprocess_file
postprocess_file("raw.wav", "clean.wav")           # default realism preset
```

For true LUFS loudness (instead of the RMS fallback), `pip install pyloudnorm`.

## F5-TTS fine-tune (alternative acoustic model)

F5-TTS is a **separate** flow-matching framework (`f5-tts`), not Coqui —
different trainer, data format, and vocoder, and **no speaker embeddings**
(it's zero-shot / reference-conditioned; you pick a voice at inference with a
reference clip). We reuse the *same* MARA+SWARA IPA-phoneme manifests; the
only bridge we own is `f5/prepare.py`, which turns them into F5's on-disk
layout (symlinked `wavs/` + `metadata.csv`, phoneme text preserved).

Plan: **fine-tune** a pretrained F5 checkpoint (the DiT/flow backbone starts
pretrained — feasible on one 2080 Ti) with a **custom IPA-phoneme vocab**.
Caveat: the pretrained text understanding (EN/ZH) does not transfer to IPA —
F5 re-initializes the text-embedding table for our vocab, so that layer learns
from your data while the acoustic backbone benefits from pretraining. Use the
**char** tokenizer so the IPA text is never transliterated (F5's prepare
defaults to pinyin).

```bash
pip install -e ".[f5]"          # installs f5-tts (GPU box)

# 1. manifests -> F5 dataset dir (symlinks; no 2.6 GB copy)
python -m tts_training.f5.prepare \
    --manifest out/mara.manifest        --corpus-root datasets/MARA \
    --manifest out/swara_train.manifest --corpus-root datasets/SWARA \
    --out data/ro_mara_swara_char

# 2. pip-installed f5-tts resolves data/ and ckpts/ relative to its package,
#    not the repo — symlink them to the repo so it finds our dataset.
ln -sfn "$PWD/data"  "$(python -c 'import f5_tts,os;print(os.path.dirname(os.path.dirname(os.path.dirname(f5_tts.__file__))))')/data"

# 3. finetune mode EXTENDS the base vocab, so fetch it where F5 looks
mkdir -p data/Emilia_ZH_EN_pinyin
cp "$(python -c "from huggingface_hub import hf_hub_download; print(hf_hub_download('SWivid/F5-TTS','F5TTS_v1_Base/vocab.txt'))")" \
   data/Emilia_ZH_EN_pinyin/vocab.txt

# 4. build raw.arrow / duration.json / vocab.txt. NO --pretrain = finetune
#    mode = extend the base vocab with our IPA symbols (no embedding mismatch).
python -m f5_tts.train.datasets.prepare_csv_wavs \
    data/ro_mara_swara_char data/ro_mara_swara_char

# 5. fine-tune from the pretrained base checkpoint
f5-tts_finetune-cli --exp_name F5TTS_v1_Base --dataset_name ro_mara_swara \
    --tokenizer char --finetune --pretrain <path/to/F5TTS_v1_Base/model_1250000.safetensors> \
    --batch_size_per_gpu 1600 --learning_rate 1e-5
```

Notes: `prepare_csv_wavs` has **no** `--tokenizer` flag; **omitting** `--pretrain`
selects finetune/extend-vocab mode (which needs step 3's base vocab). Exact
flags vary by `f5-tts` version — the adapter in step 1 is the stable part. On
11 GB VRAM start with a small `--batch_size_per_gpu` (frames) and raise it.

## Extracting to its own repo

Copy `src/tts_training/`, add `expressive-tts` (the frontend) as a dependency
or vendor `src/expressive_tts/`, keep the `training` extra (`coqui-tts`). No
other repo files are required — data and manifests are already external.

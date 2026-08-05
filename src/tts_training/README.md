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

## Extracting to its own repo

Copy `src/tts_training/`, add `expressive-tts` (the frontend) as a dependency
or vendor `src/expressive_tts/`, keep the `training` extra (`coqui-tts`). No
other repo files are required — data and manifests are already external.

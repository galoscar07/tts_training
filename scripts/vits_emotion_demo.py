#!/usr/bin/env python
"""Single-file listening demo for a trained VITS.

For every combination of
  * emotion   — each `catalina_<emotion>` speaker in the model (falls back to
                whatever speakers exist, e.g. a single `catalina`),
  * source    — sentences sampled FROM the training dataset vs NOVEL sentences
                the model never saw,
  * postprocess — raw synthesizer output vs the realism post-chain,
it synthesizes a clip and writes an `index.html` that lays them all out as audio
players grouped by emotion, so you can A/B raw-vs-pp and dataset-vs-novel by ear.

Copy the whole --out-dir to your laptop and open index.html:
    scp -r <box>:.../out/vits_emo_demo . && open vits_emo_demo/index.html

Import order matters: Coqui-TTS loads before numpy/scipy/soundfile. The frontend
(Stanza/espeak) runs in a SUBPROCESS so it never shares a process with Coqui.

Usage:
    python scripts/vits_emotion_demo.py \
        --run-dir out/training_runs/vits_catalina_emo/<run> \
        --corpus-root datasets/CATALINA \
        --num-dataset 3 --num-novel 4 \
        --out-dir out/vits_emo_demo
"""

from __future__ import annotations

# --- Coqui first (import-order matters) ------------------------------------
try:
    import torch
    import transformers.pytorch_utils as _pu

    if not hasattr(_pu, "isin_mps_friendly"):
        _pu.isin_mps_friendly = lambda e, t: torch.isin(e, t)
except Exception:
    pass
from TTS.utils.synthesizer import Synthesizer  # noqa: E402

import argparse  # noqa: E402
import html  # noqa: E402
import random  # noqa: E402
import subprocess  # noqa: E402
import sys  # noqa: E402
import tempfile  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import soundfile as sf  # noqa: E402

from tts_training.postprocess import PostProcessConfig, postprocess  # noqa: E402

# Sentences the model has NOT seen — a spread of statement / question /
# exclamation / imperative, to probe generalization beyond the dataset.
NOVEL_SENTENCES = [
    "Mâine dimineață plecăm la munte cu toată familia.",
    "Îmi poți spune cât costă biletul până la Cluj?",
    "Nu pot să cred că am câștigat premiul cel mare!",
    "Te rog să închizi fereastra, s-a făcut foarte frig.",
    "Astăzi vremea este senină și soarele strălucește puternic.",
    "De ce nu mi-ai spus nimic despre această schimbare?",
    "Copiii se joacă fericiți în parcul din apropiere.",
    "Liniștea nopții era întreruptă doar de foșnetul frunzelor.",
]


def phonemize_batch(sentences: list[str]) -> list[str]:
    """Phonemize many sentences in one subprocess call (line-for-line)."""
    if not sentences:
        return []
    with tempfile.TemporaryDirectory() as d:
        i = Path(d) / "in.txt"
        o = Path(d) / "out.txt"
        i.write_text("\n".join(sentences) + "\n", encoding="utf-8")
        subprocess.run(
            [sys.executable, "-m", "tts_training.phonemize", "--in", str(i), "--out", str(o)],
            check=True,
        )
        out_lines = o.read_text(encoding="utf-8").splitlines()
    if len(out_lines) != len(sentences):
        raise SystemExit(
            f"phonemizer returned {len(out_lines)} lines for {len(sentences)} sentences — "
            "cannot align text to phonemes"
        )
    return [ln.strip() for ln in out_lines]


def dataset_sentences(corpus_root: Path, n: int, seed: int, max_chars: int = 120) -> list[str]:
    """Sample `n` distinct real transcripts from the CATALINA corpus."""
    from tts_training.data.readers import catalina_reader

    seen: list[str] = []
    pool: set[str] = set()
    for utt in catalina_reader(corpus_root):
        t = (utt.text or "").strip()
        if t and t not in pool and len(t) <= max_chars:
            pool.add(t)
            seen.append(t)
    random.Random(seed).shuffle(seen)
    return seen[:n]


def target_speakers(synth, override: str | None) -> list[str]:
    m = getattr(synth.tts_model, "speaker_manager", None)
    names = list(getattr(m, "speaker_names", None) or getattr(m, "name_to_id", {}) or []) if m else []
    if override:
        want = [s.strip() for s in override.split(",") if s.strip()]
        return [s for s in want if s in names] or want
    emo = [s for s in names if s.startswith("catalina_")]
    return emo or names or ["catalina"]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run-dir", required=True, type=Path, help="dir with config.json/speakers.pth/checkpoints")
    ap.add_argument("--checkpoint", type=Path, default=None, help="default: newest checkpoint_*.pth in run-dir")
    ap.add_argument("--corpus-root", type=Path, default=Path("datasets/CATALINA"))
    ap.add_argument("--num-dataset", type=int, default=3, help="sentences sampled from the dataset")
    ap.add_argument("--num-novel", type=int, default=4, help="novel sentences (from --novel-file or built-in)")
    ap.add_argument("--novel-file", type=Path, default=None, help="text file of novel sentences (one per line)")
    ap.add_argument("--emotions", default=None, help="comma list to force speakers, e.g. catalina_angry,catalina_calm")
    ap.add_argument("--out-dir", type=Path, default=Path("out/vits_emo_demo"))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no-cuda", action="store_true")
    args = ap.parse_args()

    ckpt = args.checkpoint
    if ckpt is None:
        chs = sorted(args.run_dir.glob("checkpoint_*.pth"), key=lambda p: p.stat().st_mtime)
        if not chs:
            raise SystemExit(f"no checkpoint_*.pth in {args.run_dir}")
        ckpt = chs[-1]
    print(f"checkpoint: {ckpt}")

    # Gather sentences (text) for both sources, then phonemize uniformly.
    ds = dataset_sentences(args.corpus_root, args.num_dataset, args.seed)
    if args.novel_file:
        novel_pool = [ln.strip() for ln in args.novel_file.read_text(encoding="utf-8").splitlines() if ln.strip()]
    else:
        novel_pool = NOVEL_SENTENCES
    nv = novel_pool[: args.num_novel]

    items: list[tuple[str, str, str]] = []  # (source, text, phonemes)
    for source, texts in (("dataset", ds), ("novel", nv)):
        for text, ph in zip(texts, phonemize_batch(texts)):
            if ph:
                items.append((source, text, ph))
    if not items:
        raise SystemExit("no sentences to synthesize")

    syn = Synthesizer(
        tts_checkpoint=str(ckpt),
        tts_config_path=str(args.run_dir / "config.json"),
        tts_speakers_file=str(args.run_dir / "speakers.pth"),
        use_cuda=not args.no_cuda,
    )
    speakers = target_speakers(syn, args.emotions)
    print(f"speakers: {speakers}")

    audio_dir = args.out_dir / "audio"
    audio_dir.mkdir(parents=True, exist_ok=True)
    pp_cfg = PostProcessConfig()
    records = []  # dicts for the HTML
    for idx, (source, text, ph) in enumerate(items):
        for spk in speakers:
            wav = np.asarray(syn.tts(ph, speaker_name=spk), dtype=np.float32)
            sr = syn.output_sample_rate
            base = f"{source}_{spk}_{idx:02d}"
            raw_rel = f"audio/{base}_raw.wav"
            pp_rel = f"audio/{base}_pp.wav"
            sf.write(str(args.out_dir / raw_rel), wav, sr)
            sf.write(str(args.out_dir / pp_rel), postprocess(wav, sr, pp_cfg), sr)
            records.append(dict(source=source, emotion=spk, text=text, raw=raw_rel, pp=pp_rel))
        print(f"  [{idx + 1}/{len(items)}] {source}: {text[:60]}")

    _write_html(records, speakers, ckpt, args.out_dir)
    print(f"done -> {args.out_dir}/index.html  ({len(records)} clips)")


def _write_html(records, speakers, ckpt, out_dir) -> None:
    def esc(s: str) -> str:
        return html.escape(str(s), quote=True)

    def player(rel: str) -> str:
        return f'<audio controls preload="none" src="{esc(rel)}"></audio>'

    parts = [
        "<!doctype html><meta charset='utf-8'>",
        "<title>VITS emotion demo</title>",
        "<style>",
        "body{font-family:system-ui,Arial,sans-serif;margin:24px;max-width:1100px}",
        "h1{font-size:20px}h2{margin-top:32px;border-bottom:2px solid #ddd;padding-bottom:4px}",
        "table{border-collapse:collapse;width:100%;margin:12px 0}",
        "td,th{border:1px solid #ddd;padding:8px;vertical-align:top;text-align:left}",
        "th{background:#f5f5f5}.txt{font-size:14px;color:#333;max-width:360px}",
        ".src{font-size:12px;color:#888;text-transform:uppercase}",
        "audio{height:32px}", "code{font-size:11px;color:#666}",
        "</style>",
        "<h1>VITS emotion demo</h1>",
        f"<p><code>checkpoint: {esc(ckpt)}</code></p>",
        "<p>Each row is one sentence. <b>raw</b> = synthesizer output, "
        "<b>pp</b> = post-processed. <span class='src'>dataset</span> = sentence the model "
        "trained on; <span class='src'>novel</span> = unseen sentence.</p>",
    ]
    for spk in speakers:
        parts.append(f"<h2>{esc(spk)}</h2>")
        parts.append("<table><tr><th>sentence</th><th>raw</th><th>post-processed</th></tr>")
        rows = [r for r in records if r["emotion"] == spk]
        rows.sort(key=lambda r: (r["source"] != "dataset", r["text"]))
        for r in rows:
            parts.append(
                "<tr>"
                f"<td class='txt'><span class='src'>{esc(r['source'])}</span><br>{esc(r['text'])}</td>"
                f"<td>{player(r['raw'])}</td>"
                f"<td>{player(r['pp'])}</td>"
                "</tr>"
            )
        parts.append("</table>")
    (out_dir / "index.html").write_text("\n".join(parts), encoding="utf-8")


if __name__ == "__main__":
    main()

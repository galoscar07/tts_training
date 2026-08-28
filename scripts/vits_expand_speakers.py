#!/usr/bin/env python
"""Warm-start a VITS checkpoint for a NEW speaker set.

Why this exists
---------------
`train.py` rebuilds the speaker table from the manifest
(`SpeakerManager.set_ids_from_data`, ids assigned in **sorted name order**) and
sets `num_speakers` from it. When you restore a checkpoint whose speaker table
has a different number of rows, coqui's trainer falls back to *partial*
initialization and silently DROPS every shape-mismatched tensor — including
`emb_g.weight`, the speaker embedding. The emotion fine-tune therefore starts
with a **randomly initialized** speaker embedding and has to relearn Catalina's
voice from ~1.4k utterances, which is exactly what a muffled/unstable
fine-tune sounds like.

This script rewrites `emb_g.weight` to the new shape *before* training, copying
each target speaker's row from a source speaker, so the restore is a clean
strict load and every emotion slot starts from the voice the model already
knows.

Mapping rule for each target speaker (first match wins):
  1. explicit `--map target=source`
  2. exact same name in the source checkpoint
  3. name with the trailing `_<token>` stripped (catalina_angry -> catalina)
  4. the only source speaker, if the source has exactly one
  5. the mean of all source rows

Because expanding the embedding invalidates the optimizer moments for that
tensor, the optimizer/scaler state is dropped (this is a fine-tune restore, so
fresh moments are what you want anyway — and it shrinks the file a lot).

Usage (on the box, from the repo root):
    python scripts/vits_expand_speakers.py \
        --checkpoint out/training_runs/vits_catalina_ft/<run>/checkpoint_XXXX.pth \
        --speakers-file out/training_runs/vits_catalina_ft/<run>/speakers.pth \
        --manifest out/catalina_emotions.manifest \
        --out out/vits_catalina_emo_init.pth
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

EMB_KEYS = ("emb_g.weight", "module.emb_g.weight")


def _load_source_ids(speakers_file: Path | None) -> dict[str, int]:
    """`speakers.pth` is a torch-saved {name: id} dict (older runs wrap it in a
    SpeakerManager-like object with `.ids`)."""
    if speakers_file is None or not speakers_file.exists():
        return {}
    obj = torch.load(str(speakers_file), map_location="cpu", weights_only=False)
    if isinstance(obj, dict) and obj and all(isinstance(v, int) for v in obj.values()):
        return obj
    ids = getattr(obj, "ids", None)
    if isinstance(ids, dict):
        return ids
    print(f"  ! could not parse speaker ids from {speakers_file} ({type(obj)})")
    return {}


def _target_speakers(manifest: Path) -> list[str]:
    """Same ordering train.py will use: sorted unique speaker column."""
    names = set()
    with manifest.open(encoding="utf-8") as handle:
        for raw in handle:
            parts = raw.rstrip("\n").split("|")
            if len(parts) >= 3 and parts[2].strip():
                names.add(parts[2].strip())
    return sorted(names)


def _find_emb(state: dict) -> str:
    for key in EMB_KEYS:
        if key in state:
            return key
    raise SystemExit(
        "no speaker embedding (emb_g.weight) in the checkpoint — was it trained "
        "with use_speaker_embedding=True?"
    )


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--checkpoint", required=True, type=Path, help="source .pth to warm-start from")
    p.add_argument("--speakers-file", type=Path, default=None, help="source speakers.pth (name -> id)")
    p.add_argument("--manifest", type=Path, default=None, help="new manifest; target speakers are its sorted speaker column")
    p.add_argument("--target-speakers", default=None, help="comma list, overrides --manifest")
    p.add_argument("--map", action="append", default=[], metavar="TARGET=SOURCE",
                   help="force one mapping (repeatable)")
    p.add_argument("--jitter", type=float, default=0.0,
                   help="std of gaussian noise added to each copied row, to break symmetry between slots")
    p.add_argument("--keep-optimizer", action="store_true",
                   help="keep optimizer/scaler state (only valid if the shape is unchanged)")
    p.add_argument("--out", required=True, type=Path)
    args = p.parse_args(argv)

    if not args.target_speakers and not args.manifest:
        p.error("pass --manifest or --target-speakers")

    forced = {}
    for item in args.map:
        target, _, source = item.partition("=")
        if not source:
            p.error(f"--map expects TARGET=SOURCE, got {item!r}")
        forced[target.strip()] = source.strip()

    targets = ([s.strip() for s in args.target_speakers.split(",") if s.strip()]
               if args.target_speakers else _target_speakers(args.manifest))
    if not targets:
        raise SystemExit("no target speakers found")

    print(f"loading {args.checkpoint}")
    ckpt = torch.load(str(args.checkpoint), map_location="cpu", weights_only=False)
    state = ckpt["model"]
    key = _find_emb(state)
    src_emb = state[key]
    n_src, dim = src_emb.shape

    src_ids = _load_source_ids(args.speakers_file)
    if src_ids:
        print(f"source speakers ({len(src_ids)}): {', '.join(sorted(src_ids)[:8])}"
              + (" ..." if len(src_ids) > 8 else ""))
    print(f"source emb_g: [{n_src}, {dim}]")
    print(f"target speakers ({len(targets)}): {', '.join(targets)}")

    mean_row = src_emb.mean(dim=0)
    new_emb = torch.empty(len(targets), dim, dtype=src_emb.dtype)

    for i, name in enumerate(targets):
        source_name, row = None, None
        candidates = []
        if name in forced:
            candidates.append(forced[name])
        candidates.append(name)
        if "_" in name:
            candidates.append(name.rsplit("_", 1)[0])
        for cand in candidates:
            if cand in src_ids and src_ids[cand] < n_src:
                source_name, row = cand, src_emb[src_ids[cand]]
                break
        if row is None and n_src == 1:
            source_name, row = "(only source speaker)", src_emb[0]
        if row is None:
            source_name, row = "(mean of all source rows)", mean_row
        row = row.clone()
        if args.jitter > 0:
            row += torch.randn_like(row) * args.jitter
        new_emb[i] = row
        print(f"  [{i}] {name:<24} <- {source_name}")

    state[key] = new_emb
    shape_changed = n_src != len(targets)

    if shape_changed and not args.keep_optimizer:
        for field in ("optimizer", "scaler", "amp_scaler"):
            if ckpt.pop(field, None) is not None:
                print(f"dropped {field} state (speaker table resized)")
    elif args.keep_optimizer and shape_changed:
        print("! --keep-optimizer with a resized speaker table will fail on restore")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(ckpt, str(args.out))
    size_mb = args.out.stat().st_size / 1e6
    print(f"\nwrote {args.out}  ({size_mb:.0f} MB)  emb_g: [{len(targets)}, {dim}]")
    print("restore it with:  --restore-path " + str(args.out))


if __name__ == "__main__":
    main()

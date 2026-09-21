#!/usr/bin/env python3
"""Digest a Coqui VITS training log into a short report you can paste anywhere.

A multi-day VITS log is hundreds of MB, and `tail` only shows the last step —
which is exactly what you cannot judge "is it still learning?" from. This
walks the whole log, keeps the per-epoch eval losses, and prints a compact
table plus a health check (NaN, AMP-scaler collapse, LR decay, best-model
staleness, throughput).

    python3 scripts/vits_train_report.py                   # newest run
    python3 scripts/vits_train_report.py path/to/train.log
    python3 scripts/vits_train_report.py --rows 80         # longer table

Stdlib only, no Coqui import — it parses the console log, so it runs on the
GPU box or on a log you copied to your laptop.
"""

from __future__ import annotations

import argparse
import glob
import os
import re
import sys
import time
from typing import Optional

ANSI = re.compile(r"\x1b\[[0-9;]*[A-Za-z]")
# `   --> TIME: ... -- STEP: 25/1780 -- GLOBAL_STEP: 12345`
RE_STEP = re.compile(r"GLOBAL_STEP:\s*(\d+)")
RE_EPOCH = re.compile(r">\s*EPOCH:\s*(\d+)\s*/\s*(\d+)")
RE_EPOCH_TIME = re.compile(r"EPOCH TIME:\s*([\d.]+)\s*sec")
# `     | > loss_mel: 18.4  (19.1)`  and  `     | > avg_loss_mel:18.4`
RE_METRIC = re.compile(
    r"^\s*\|\s*>\s*([A-Za-z0-9_]+)\s*:\s*"
    r"(-?(?:\d+\.?\d*(?:[eE][-+]?\d+)?|nan|inf))"
    r"(?:\s*\(\s*(-?(?:\d+\.?\d*(?:[eE][-+]?\d+)?|nan|inf))\s*\))?",
    re.IGNORECASE,
)
RE_BEST = re.compile(r">\s*BEST MODEL\s*:\s*(\S+)")
RE_CKPT = re.compile(r">\s*CHECKPOINT\s*:\s*(\S+)")

# Eval keys worth a column, in display order. VITS reports loss_1 as the total
# generator loss and loss_0 as the total discriminator loss.
COLUMNS = [
    ("mel", "avg_loss_mel"),
    ("dur", "avg_loss_duration"),
    ("kl", "avg_loss_kl"),
    ("feat", "avg_loss_feat"),
    ("gen", "avg_loss_gen"),
    ("disc", "avg_loss_disc"),
]
# These say "the optimizer is alive", not "the model is good".
DIAG_KEYS = ["current_lr_0", "current_lr_1", "grad_norm_0", "grad_norm_1", "amp_scaler", "scaler"]


def _f(text: str) -> float:
    return float(text)


def is_bad(value: float) -> bool:
    return value != value or value in (float("inf"), float("-inf"))


class Epoch:
    __slots__ = ("index", "step", "eval", "train", "seconds", "best")

    def __init__(self, index: int) -> None:
        self.index = index
        self.step: Optional[int] = None
        self.eval: dict[str, float] = {}
        self.train: dict[str, float] = {}
        self.seconds: Optional[float] = None
        self.best = False


def default_log(repo_root: str) -> Optional[str]:
    """Newest training log: the controller's pointer first, then a glob."""
    for pointer in (
        os.path.join(repo_root, "out/training_runs/control/current-log"),
        os.path.join(repo_root, "out/training_runs/control/current-emo-log"),
    ):
        if os.path.isfile(pointer):
            with open(pointer, encoding="utf-8", errors="replace") as handle:
                path = handle.read().strip()
            if path and os.path.isfile(path):
                return path
    candidates = glob.glob(
        os.path.join(repo_root, "out/training_runs/*/logs/training-*.log")
    ) + glob.glob(os.path.join(repo_root, "out/training_runs/*/*/trainer_*_log.txt"))
    candidates = [c for c in candidates if not os.path.islink(c)]
    return max(candidates, key=os.path.getmtime) if candidates else None


def parse(path: str) -> dict:
    epochs: list[Epoch] = []
    current: Optional[Epoch] = None
    section = None  # "train_step" | "train_end" | "eval_end"
    max_epoch = None
    run_dir = None
    last_step = 0
    lr_first: dict[str, float] = {}
    lr_last: dict[str, float] = {}
    diag_last: dict[str, float] = {}
    bad: list[tuple[int, str]] = []          # (epoch, key) with NaN/Inf
    ooms: list[int] = []
    tracebacks: list[int] = []
    best_saves: list[tuple[int, int]] = []   # (epoch, step)
    tail: list[str] = []
    first_line_time = last_line_time = None

    with open(path, encoding="utf-8", errors="replace") as handle:
        for raw in handle:
            line = ANSI.sub("", raw.rstrip("\n"))
            tail.append(line)
            if len(tail) > 400:
                del tail[:200]

            if run_dir is None and line.startswith(" --> ") and "/" in line:
                run_dir = line[5:].strip()

            stamp = re.search(r"(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})", line)
            if stamp:
                first_line_time = first_line_time or stamp.group(1)
                last_line_time = stamp.group(1)

            match = RE_EPOCH.search(line)
            if match:
                current = Epoch(int(match.group(1)))
                max_epoch = int(match.group(2))
                epochs.append(current)
                section = None
                continue

            if "CUDA out of memory" in line:
                ooms.append(current.index if current else -1)
            if "Traceback (most recent call last)" in line:
                tracebacks.append(current.index if current else -1)

            match = RE_BEST.search(line)
            if match:
                if current:
                    current.best = True
                    best_saves.append((current.index, last_step))
                continue
            if RE_CKPT.search(line):
                continue

            if "EVAL PERFORMANCE" in line:
                section = "eval_end"
                continue
            if "TRAIN PERFORMA" in line:  # upstream spells it "PERFORMACE"
                section = "train_end"
                match = RE_EPOCH_TIME.search(line)
                if match and current:
                    current.seconds = _f(match.group(1))
                match = RE_STEP.search(line)
                if match:
                    last_step = int(match.group(1))
                    if current:
                        current.step = last_step
                continue

            match = RE_STEP.search(line)
            if match:
                section = "train_step"
                last_step = int(match.group(1))
                if current and current.step is None:
                    current.step = last_step
                continue

            match = RE_METRIC.match(line)
            if not match:
                continue
            key = match.group(1)
            value = _f(match.group(2))
            running = _f(match.group(3)) if match.group(3) else None

            if is_bad(value) or (running is not None and is_bad(running)):
                bad.append((current.index if current else -1, key))

            if key in DIAG_KEYS:
                diag_last[key] = value
                if key.startswith("current_lr"):
                    lr_first.setdefault(key, value)
                    lr_last[key] = value
                if current is not None:
                    # Keep these per-epoch too: a decaying LR or a collapsing
                    # AMP scaler is exactly what a flat loss curve looks like.
                    current.train[key] = value
            if current is None:
                continue
            if section == "eval_end":
                current.eval[key] = value
            elif section == "train_end":
                current.train[key] = value
            elif section == "train_step" and running is not None:
                # Running mean over the epoch so far — the best in-epoch signal.
                current.train.setdefault(f"avg_{key}", running)
                current.train[f"avg_{key}"] = running

    return {
        "path": path,
        "run_dir": run_dir,
        "epochs": epochs,
        "max_epoch": max_epoch,
        "last_step": last_step,
        "lr_first": lr_first,
        "lr_last": lr_last,
        "diag_last": diag_last,
        "bad": bad,
        "ooms": ooms,
        "tracebacks": tracebacks,
        "best_saves": best_saves,
        "tail": tail[-25:],
        "started": first_line_time,
        "last_line_time": last_line_time,
    }


def fmt(value: Optional[float], width: int = 8, places: int = 3) -> str:
    if value is None:
        return "-".rjust(width)
    if is_bad(value):
        return ("NaN" if value != value else "Inf").rjust(width)
    if value != 0 and abs(value) < 1e-3:
        return f"{value:.2e}".rjust(width)
    return f"{value:.{places}f}".rjust(width)


def pick_rows(epochs: list[Epoch], rows: int) -> list[Epoch]:
    """Every recent epoch, plus an even sample of the earlier ones."""
    scored = [e for e in epochs if e.eval or e.train]
    if rows <= 0:
        return []
    if len(scored) <= rows:
        return scored
    # max(1, ...) matters: rows//2 == 0 would make scored[-0:] the whole list.
    n_recent = max(1, rows // 2)
    recent, earlier = scored[-n_recent:], scored[:-n_recent]
    n_earlier = rows - n_recent
    if n_earlier <= 0 or not earlier:
        return recent
    stride = max(1, len(earlier) // n_earlier)
    return earlier[::stride][:n_earlier] + recent


def series(epochs: list[Epoch], key: str) -> list[Optional[float]]:
    """One slot per epoch, None where the value is absent or NaN — epoch-aligned
    so the trend below compares the same stretch of wall-clock training, not
    whatever valid points happened to survive."""
    out: list[Optional[float]] = []
    for epoch in epochs:
        value = epoch.eval.get(key, epoch.train.get(key))
        out.append(None if value is None or is_bad(value) else value)
    return out


def trend(points: list[Optional[float]], window: int) -> Optional[str]:
    """Compare the mean of the last `window` epochs with the `window` before."""
    if len(points) < 2 * window:
        return None
    recent = points[-window:]
    earlier = points[-2 * window : -window]
    holes = sum(1 for v in recent if v is None)
    if holes:
        return f"{holes}/{window} of the most recent epochs are NaN or missing — no trend"
    recent_ok = [v for v in recent if v is not None]
    earlier_ok = [v for v in earlier if v is not None]
    if not earlier_ok:
        return None
    a, b = sum(earlier_ok) / len(earlier_ok), sum(recent_ok) / len(recent_ok)
    if a == 0:
        return None
    delta = (b - a) / abs(a) * 100
    arrow = "improving" if delta < -0.25 else ("worsening" if delta > 0.25 else "FLAT")
    gap = f" [{window - len(earlier_ok)} gaps in the earlier window]" if len(earlier_ok) < window else ""
    return f"{a:.3f} -> {b:.3f} ({delta:+.2f}% over {window} epochs) {arrow}{gap}"


def main(argv: Optional[list[str]] = None) -> int:
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("log", nargs="?", default=None, help="training log (default: newest run)")
    parser.add_argument("--rows", type=int, default=40, help="max epoch rows in the table")
    parser.add_argument("--window", type=int, default=10, help="epochs per side for the trend test")
    parser.add_argument("--tail", type=int, default=12, help="raw log lines to append")
    args = parser.parse_args(argv)

    path = args.log or default_log(repo_root)
    if not path or not os.path.isfile(path):
        print("No training log found. Pass one explicitly.", file=sys.stderr)
        return 1

    report = parse(path)
    epochs = report["epochs"]
    size_mb = os.path.getsize(path) / 1e6
    age_min = (time.time() - os.path.getmtime(path)) / 60

    print("=" * 78)
    print("VITS TRAINING REPORT")
    print("=" * 78)
    print(f"log        : {path}")
    print(f"run dir    : {report['run_dir'] or '-'}")
    print(f"log size   : {size_mb:.1f} MB   last written: {age_min:.1f} min ago")
    print(f"generated  : {time.strftime('%Y-%m-%d %H:%M:%S')}")
    if report["started"]:
        print(f"log spans  : {report['started']}  ->  {report['last_line_time']}")

    if not epochs:
        print("\nNo epochs parsed yet — the run is still starting up (dataset scan / DDP init).")
        print("\n--- last lines " + "-" * 62)
        for line in report["tail"][-args.tail :] if args.tail else []:
            print(line)
        return 0

    last = epochs[-1]
    print(f"\nepoch      : {last.index}/{report['max_epoch']}   global step: {report['last_step']}")
    timed = [e.seconds for e in epochs if e.seconds]
    if timed:
        mean_epoch = sum(timed[-10:]) / len(timed[-10:])
        deltas = [
            b.step - a.step
            for a, b in zip(epochs, epochs[1:])
            if a.step and b.step and b.step > a.step
        ]
        steps_per_epoch = sorted(deltas)[len(deltas) // 2] if deltas else 0
        print(f"epoch time : {mean_epoch / 60:.1f} min (mean of last {len(timed[-10:])})"
              f"   ~{steps_per_epoch} steps/epoch")
        remaining = (report["max_epoch"] or 0) - last.index
        if remaining > 0:
            print(f"eta to end : {remaining * mean_epoch / 3600:.1f} h for {remaining} more epochs")

    # ---------------- health ----------------
    print("\n" + "-" * 78)
    print("HEALTH")
    print("-" * 78)

    if report["bad"]:
        keys = sorted({k for _, k in report["bad"]})
        first_epoch = report["bad"][0][0]
        print(f"NaN/Inf    : *** {len(report['bad'])} hits, first at epoch {first_epoch}, keys: {', '.join(keys[:6])}")
    else:
        print("NaN/Inf    : none")

    scaler = report["diag_last"].get("amp_scaler", report["diag_last"].get("scaler"))
    if scaler is None:
        print("AMP scaler : not logged")
    elif scaler < 1.0:
        print(f"AMP scaler : *** {scaler:g} — collapsed; fp16 overflow is skipping optimizer steps")
    else:
        print(f"AMP scaler : {scaler:g}")

    for key in ("grad_norm_0", "grad_norm_1"):
        if key in report["diag_last"]:
            value = report["diag_last"][key]
            flag = "  *** zero/NaN — no gradient reaching the weights" if (value == 0 or is_bad(value)) else ""
            print(f"{key:<11}: {value:g}{flag}")

    for key in sorted(report["lr_last"]):
        start, now = report["lr_first"][key], report["lr_last"][key]
        ratio = now / start if start else 0
        flag = "  *** decayed >100x — the schedule, not the model, stopped training" if ratio < 0.01 else ""
        print(f"{key:<11}: {start:.3e} -> {now:.3e}  (x{ratio:.4f}){flag}")

    # Catch a runaway schedule on day 1 instead of at epoch 100. Coqui's VITS
    # gamma (0.999875) is meant to be applied once per EPOCH (~0.01%/epoch); if
    # it is being applied once per STEP the LR falls by roughly a fifth every
    # epoch and the run flatlines a hundred epochs later.
    lrs = [e.train.get("current_lr_1") for e in epochs]
    lrs = [v for v in lrs if v]
    if len(lrs) >= 3:
        ratios = sorted(b / a for a, b in zip(lrs, lrs[1:]) if a)
        per_epoch = ratios[len(ratios) // 2]
        note = f"LR decay   : x{per_epoch:.6f} per epoch"
        if per_epoch < 0.995:
            note += (
                f"  *** TOO FAST — at this rate the LR is"
                f" {per_epoch ** 100:.2e} of its start by epoch 100."
                "\n             Coqui's gamma is per-epoch; a per-step schedule looks like this."
                "\n             Fix: set scheduler_after_epoch=True in vits/config.py."
            )
        print(note)

    print(f"CUDA OOM   : {'*** ' + str(len(report['ooms'])) + ' hits' if report['ooms'] else 'none'}")
    print(f"tracebacks : {'*** ' + str(len(report['tracebacks'])) + ' hits' if report['tracebacks'] else 'none'}")

    if report["best_saves"]:
        b_epoch, b_step = report["best_saves"][-1]
        stale = last.index - b_epoch
        flag = "  *** eval loss has not improved in a long time" if stale >= 25 else ""
        print(f"best model : epoch {b_epoch} (step {b_step}) — {stale} epochs ago{flag}")
    else:
        print("best model : none saved yet (config save_best_after gates this)")

    # ---------------- curve ----------------
    rows = pick_rows(epochs, args.rows)
    print("\n" + "-" * 78)
    print("EVAL LOSS PER EPOCH  (avg_* from the EVAL PERFORMANCE block)")
    print("-" * 78)
    if rows:
        print(f"{'epoch':>6} {'step':>9}" + "".join(f"{name:>9}" for name, _ in COLUMNS)
              + f"{'lr_gen':>10}{'min/ep':>8}  best")
    for epoch in rows:
        source = epoch.eval or epoch.train
        row = f"{epoch.index:>6} {epoch.step or 0:>9}"
        for _, key in COLUMNS:
            row += fmt(source.get(key), 9)
        row += fmt(epoch.train.get("current_lr_1"), 10, 6)
        row += f"{epoch.seconds / 60:>8.1f}" if epoch.seconds else " " * 8
        row += "   *" if epoch.best else ""
        print(row)
    if not any(e.eval for e in epochs):
        print("(no EVAL block parsed — values above are in-epoch training running means)")

    # ---------------- verdict ----------------
    print("\n" + "-" * 78)
    print(f"IS IT STILL LEARNING?  (last {args.window} epochs vs the {args.window} before)")
    print("-" * 78)
    for name, key in COLUMNS:
        points = series(epochs, key)
        line = trend(points, args.window)
        if line:
            print(f"{name:>5}: {line}")
        elif any(v is not None for v in points):
            have = sum(1 for v in points if v is not None)
            print(f"{name:>5}: only {have} epochs — need {2 * args.window} for a trend")
    print("\nnote: loss_gen/loss_disc are adversarial and oscillate around a fixed")
    print("      point forever — a flat value there is normal. mel/dur/kl are the")
    print("      real progress signals.")

    print("\n--- last log lines " + "-" * 58)
    for line in report["tail"][-args.tail :] if args.tail else []:
        print(line)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

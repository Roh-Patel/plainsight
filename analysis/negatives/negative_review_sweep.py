"""Pooled sweep over both graded rounds: what would a stricter reranker cut buy?

Round 1: scratch_synth\\round1\\ (hardest retained negatives under t = 0, seed 20).
Round 2: scratch_synth\\ (hardest retained negatives under t = -5, seed 22).
Uses negative items only (120 per round). CPU only, writes nothing.
Run from the repo root: python -m analysis.negatives.negative_review_sweep [--thresholds -5 -6 -7 -8]

Limit: items are re-scored under each cut. After a real change the hardest retained
negative would be a different candidate, and round 1 items come from a different pool
than round 2 items, so the rates are approximations. Cuts are scanned on the same data
that measured the problem, so they are optimistic until confirmed on a fresh sample.
"""
import argparse
import json
import math
import random
from pathlib import Path

import numpy as np

ROUNDS = {
    "r1": (Path("scratch_synth/round1/neg_review.tsv"), Path("scratch_synth/round1/neg_review_key.json")),
    "r2": (Path("scratch_synth/neg_review.tsv"), Path("scratch_synth/neg_review_key.json")),
}
VOICES = ["layperson", "confused_undergrad", "half_remembered_term"]
SHORT = ["lay", "conf", "half"]
SEED = 21
N_BOOT = 2000


def wilson(k, n, z=1.96):
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    d = 1 + z * z / n
    c = p + z * z / (2 * n)
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return ((c - h) / d, (c + h) / d)


def auc(pos, neg):
    pos, neg = np.asarray(pos, float), np.asarray(neg, float)
    return (pos[:, None] > neg[None, :]).mean() + 0.5 * (pos[:, None] == neg[None, :]).mean()


def auc_ci(rows):
    fn = [r["ce"] for r in rows if r["b"] == 1]
    tn = [r["ce"] for r in rows if r["b"] == 0]
    if not fn or not tn:
        return float("nan"), float("nan"), float("nan")
    rng = random.Random(SEED)
    boots = []
    for _ in range(N_BOOT):
        s = [rng.choice(rows) for _ in rows]
        p = [r["ce"] for r in s if r["b"] == 1]
        n = [r["ce"] for r in s if r["b"] == 0]
        if p and n:
            boots.append(auc(p, n))
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return auc(fn, tn), lo, hi


def load():
    items = []
    for rnd, (tsv, key_path) in ROUNDS.items():
        key = json.loads(key_path.read_text(encoding="utf-8"))
        labels = {}
        with open(tsv, encoding="utf-8") as f:
            f.readline()
            for line in f:
                parts = line.rstrip("\n").split("\t")
                parts += [""] * (2 - len(parts))
                labels[parts[0]] = parts[1].strip()
        for item, r in key.items():
            if r["kind"] != "neg":
                continue
            if labels.get(item) not in ("0", "1"):
                raise SystemExit(f"{rnd} item {item} has no 0/1 label")
            items.append({"round": rnd, "voice": r["voice"], "ce": r["ce_score"],
                          "b": int(labels[item]), "rank": r["baseline_rank"]})
    return items


def cell(rows):
    k = sum(r["b"] for r in rows)
    return f"{k}/{len(rows)}" + (f" ({k / len(rows):.2f})" if rows else "")


def main(thresholds):
    items = load()
    print(f"{len(items)} graded negatives, {sum(r['b'] for r in items)} false negatives (B = 1)\n")

    print("Items and false negatives per voice and round")
    print(f"{'voice':<22}{'round 1':>14}{'round 2':>14}{'pooled':>16}")
    for v in VOICES:
        rows = [r for r in items if r["voice"] == v]
        print(f"{v:<22}" + "".join(
            f"{cell([r for r in rows if r['round'] == rnd]):>14}" for rnd in ("r1", "r2"))
            + f"{cell(rows):>16}")

    print("\nCan the logit separate false negatives from true negatives? AUC, pooled over both rounds")
    for v in VOICES + ["all"]:
        rows = [r for r in items if v in ("all", r["voice"])]
        a, lo, hi = auc_ci(rows)
        print(f"  {v:<22}n={len(rows):<4} AUC {a:.3f} (95% bootstrap CI {lo:.3f} to {hi:.3f})")

    print("\nFalse negatives among items with logit <= t (kept FN/kept), pooled over both rounds")
    print(f"{'t':>3} " + "".join(f"{s:>16}" for s in SHORT + ['all']))
    for t in thresholds:
        cells = []
        for v in VOICES + ["all"]:
            rows = [r for r in items if v in ("all", r["voice"]) and r["ce"] <= t]
            cells.append(cell(rows))
        print(f"{t:>3} " + "".join(f"{c:>16}" for c in cells))

    print("\nWilson 95% intervals for layperson and all voices at the same cuts")
    for t in thresholds:
        out = []
        for v in ("layperson", "all"):
            rows = [r for r in items if v in ("all", r["voice"]) and r["ce"] <= t]
            k = sum(r["b"] for r in rows)
            lo, hi = wilson(k, len(rows))
            out.append(f"{v}: {lo:.3f}-{hi:.3f}")
        print(f"  t={t:>3}  " + "   ".join(out))

    print("\nBy baseline rank, items with logit <= -5 only (comparable to the current pool), pooled")
    for label, lo_r, hi_r in (("10-12", 10, 12), ("13-14", 13, 14), ("15-19", 15, 19), ("20+", 20, 99)):
        rows = [r for r in items if r["ce"] <= -5 and lo_r <= r["rank"] <= hi_r]
        print(f"  rank {label:<6}{cell(rows)}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--thresholds", type=float, nargs="+", default=[-5, -6, -7, -8])
    main([int(t) if float(t).is_integer() else t for t in ap.parse_args().thresholds])
"""What an absolute reranker cut does to the hard-negative pool.

CPU only, reads hn_scored.jsonl and ranks.jsonl, writes nothing.
A candidate is dropped when its reranker logit is above t.
Run from the repo root: python -m analysis.negatives.threshold_impact [--thresholds -7 -6 -5 -4 -3]
"""
import argparse
import json
from pathlib import Path

import numpy as np

HN = Path("data/synthetic/hn_scored.jsonl")
SRC_IDS = Path("data/synthetic/ranks/ranks.jsonl")

VOICES = ["layperson", "confused_undergrad", "half_remembered_term"]
SHORT = ["lay", "conf", "half"]
DEFAULT_THRESHOLDS = (-4, -3, -2, -1, 0, 1, 2)
BANDS = ((10, 19), (20, 34), (35, 50))


def read_jsonl(path):
    with open(path, encoding="utf-8") as f:
        for line in f:
            yield json.loads(line)


def main(THRESHOLDS):
    info = {r["qid"]: (VOICES.index(r["voice"]), r["hybrid_base_src_rank"]) for r in read_jsonl(SRC_IDS)}
    qv, q_out = [], []
    fq, fr, fs = [], [], []
    for qi, row in enumerate(read_jsonl(HN)):
        v, sr = info[row["qid"]]
        qv.append(v)
        q_out.append(not (sr is not None and sr <= 50))
        for _, rank, score in row["cands"]:
            fq.append(qi)
            fr.append(rank)
            fs.append(score)
    qv = np.array(qv)
    q_out = np.array(q_out)
    fq = np.array(fq)
    fr = np.array(fr)
    fs = np.array(fs)
    nq = len(qv)
    cv = qv[fq]       # voice of each candidate
    co = q_out[fq]    # stratum of each candidate
    print(f"{nq} queries, {len(fs)} candidates in pool C (structural rules only; "
          "near-duplicate exclusion will remove a few more)")

    def pct(x):
        return f"{100 * x:6.1f}"

    print("\nShare of candidates dropped (%), by voice | by baseline rank band")
    print(f"{'t':>3} | " + " ".join(f"{s:>6}" for s in SHORT) + " | "
          + " ".join(f"{a}-{b}".rjust(6) for a, b in BANDS))
    for t in THRESHOLDS:
        drop = fs > t
        by_v = [drop[cv == v].mean() for v in range(3)]
        by_b = [drop[(fr >= a) & (fr <= b)].mean() for a, b in BANDS]
        print(f"{t:>3} | " + " ".join(pct(x) for x in by_v) + " | " + " ".join(pct(x) for x in by_b))

    print("\nShare of candidates dropped (%), out stratum | in stratum")
    print(f"{'t':>3} | " + " ".join(f"{s:>6}" for s in SHORT) + " | "
          + " ".join(f"{s:>6}" for s in SHORT))
    for t in THRESHOLDS:
        drop = fs > t
        o = [drop[(cv == v) & co].mean() for v in range(3)]
        i = [drop[(cv == v) & ~co].mean() for v in range(3)]
        print(f"{t:>3} | " + " ".join(pct(x) for x in o) + " | " + " ".join(pct(x) for x in i))

    print("\nShare of queries (%) left with fewer than 10 | fewer than 5 | zero negatives")
    print(f"{'t':>3} | " + " ".join(f"{s:>6}" for s in SHORT) + " | "
          + " ".join(f"{s:>6}" for s in SHORT) + " | " + " ".join(f"{s:>6}" for s in SHORT))
    for t in THRESHOLDS:
        left = np.bincount(fq[fs <= t], minlength=nq)
        cells = []
        for lim in (10, 5, 1):
            cells.append(" ".join(pct((left[qv == v] < lim).mean()) for v in range(3)))
        print(f"{t:>3} | " + " | ".join(cells))

    print("\nMean negatives left per query")
    print(f"{'t':>3} | " + " ".join(f"{s:>6}" for s in SHORT))
    for t in THRESHOLDS:
        left = np.bincount(fq[fs <= t], minlength=nq)
        print(f"{t:>3} | " + " ".join(f"{left[qv == v].mean():6.1f}" for v in range(3)))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--thresholds", type=float, nargs="+", default=list(DEFAULT_THRESHOLDS))
    main([int(t) if float(t).is_integer() else t for t in ap.parse_args().thresholds])
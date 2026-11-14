"""Blind hand check of retained hard negatives (residual false-negative rate).

  python -m train.make_negative_review sample [--seed N]
      writes scratch_synth\\neg_review.txt, neg_review.tsv (to fill in), neg_review_key.json
  python -m train.make_negative_review analyze
      reads the filled TSV and the key; prints rates with 95% Wilson intervals

Each voice gets N_NEG queries, each shown with its hardest retained negative (lowest
baseline rank), plus N_CTRL control queries shown with their own source passage.
All items are shuffled together. Label B for every item: 1 if the passage answers the
query, 0 if not.
"""
import argparse
import json
import math
import random
from pathlib import Path

QUERIES = Path("data/synthetic/queries.jsonl")
SRC_IDS = Path("data/synthetic/ranks/ranks.jsonl")
PASSAGES = Path("data/processed/passages.jsonl")
HN_OUT = Path("data/synthetic/hard_negatives.jsonl")
SCRATCH = Path("scratch_synth")
TXT = SCRATCH / "neg_review.txt"
TSV = SCRATCH / "neg_review.tsv"
KEY = SCRATCH / "neg_review_key.json"

VOICES = ["layperson", "confused_undergrad", "half_remembered_term"]
DEFAULT_SEED = 20  # round 1 used 20; use a different seed for a new round
ACCEPT_MAX = 0.10   
TIGHTEN_MIN = 0.20  


def read_jsonl(path):
    with open(path, encoding="utf-8") as f:
        for line in f:
            yield json.loads(line)


def wilson(k, n, z=1.96):
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    d = 1 + z * z / n
    c = p + z * z / (2 * n)
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return ((c - h) / d, (c + h) / d)


def sample(args):
    if TXT.exists() or TSV.exists() or KEY.exists():
        raise SystemExit("review files already exist in scratch_synth. Delete them by hand to resample "
                         "(a filled TSV would be lost).")
    SCRATCH.mkdir(exist_ok=True)
    queries = list(read_jsonl(QUERIES))
    qinfo = {q["qid"]: q for q in queries}
    sr = {r["qid"]: r["hybrid_base_src_rank"] for r in read_jsonl(SRC_IDS)}

    hardest = {}
    for row in read_jsonl(HN_OUT):
        if row["negatives"]:
            hardest[row["qid"]] = min(row["negatives"], key=lambda n: n["baseline_rank"])
    by_voice = {v: [] for v in VOICES}
    for q in queries:
        if q["kept"] and q["qid"] in hardest:
            by_voice[q["voice"]].append(q["qid"])

    rng = random.Random(args.seed)
    items = []
    for v in VOICES:
        picked = rng.sample(by_voice[v], args.n_neg + args.n_ctrl)
        for qid in picked[:args.n_neg]:
            n = hardest[qid]
            items.append({"qid": qid, "voice": v, "kind": "neg", "passage_id": n["passage_id"],
                          "baseline_rank": n["baseline_rank"], "ce_score": n["ce_score"]})
        for qid in picked[args.n_neg:]:
            items.append({"qid": qid, "voice": v, "kind": "control",
                          "passage_id": qinfo[qid]["source_passage_id"],
                          "baseline_rank": None, "ce_score": None})
    for it in items:
        s = sr[it["qid"]]
        it["stratum"] = "in" if (s is not None and s <= 50) else "out"
    rng.shuffle(items)
    for k, it in enumerate(items, 1):
        it["item"] = f"{k:03d}"

    pids = [r["passage_id"] for r in read_jsonl(PASSAGES)]
    index = {p: i for i, p in enumerate(pids)}
    wanted = {index[it["passage_id"]] for it in items}
    text = {}
    for i, r in enumerate(read_jsonl(PASSAGES)):
        if i in wanted:
            text[i] = (r["section_title"], r["text"])
            if len(text) == len(wanted):
                break

    with open(TXT, "w", encoding="utf-8", newline="\n") as f:
        f.write("Label B for every item in neg_review.tsv: 1 if the passage answers the query "
                "(same definition as the D045 B label), 0 if not.\n\n")
        for it in items:
            sec, body = text[index[it["passage_id"]]]
            f.write(f"=== ITEM {it['item']} ===\nQUERY: {queries_text(qinfo, it)}\n"
                    f"SECTION: {sec}\nPASSAGE:\n{body}\n\n")
    with open(TSV, "w", encoding="utf-8", newline="\n") as f:
        f.write("item\tB\n")
        for it in items:
            f.write(f"{it['item']}\t\n")
    KEY.write_text(json.dumps({it["item"]: it for it in items}, indent=1), encoding="utf-8")
    print(f"{len(items)} items ({args.n_neg} negatives + {args.n_ctrl} controls per voice), "
          f"seed {args.seed}. Grade {TSV} against {TXT}. Do not open {KEY}.")


def queries_text(qinfo, it):
    return qinfo[it["qid"]]["query"]


def analyze(_args):
    key = json.loads(KEY.read_text(encoding="utf-8"))
    labels = {}
    with open(TSV, encoding="utf-8") as f:
        f.readline()
        for line in f:
            parts = line.rstrip("\n").split("\t")
            parts += [""] * (2 - len(parts))
            labels[parts[0]] = parts[1].strip()
    missing = [k for k in key if labels.get(k, "") not in ("0", "1")]
    if missing:
        raise SystemExit(f"{len(missing)} items have no 0/1 label, first: {missing[:5]}")

    def rate(rows):
        k = sum(int(labels[r["item"]]) for r in rows)
        lo, hi = wilson(k, len(rows))
        return k, len(rows), lo, hi

    print("Residual false negatives among the hardest retained negative (B = 1)")
    print(f"{'voice':<22}{'k/n':>8}{'rate':>8}{'95% CI':>16}   rule")
    for v in VOICES:
        rows = [r for r in key.values() if r["voice"] == v and r["kind"] == "neg"]
        k, n, lo, hi = rate(rows)
        p = k / n
        verdict = ("accept" if p <= ACCEPT_MAX else
                   "tighten before T16" if p >= TIGHTEN_MIN else "grade 20 more for this voice")
        print(f"{v:<22}{f'{k}/{n}':>8}{p:>8.3f}{f'{lo:.3f}-{hi:.3f}':>16}   {verdict}")
    rows = [r for r in key.values() if r["kind"] == "neg"]
    k, n, lo, hi = rate(rows)
    print(f"{'all voices':<22}{f'{k}/{n}':>8}{k / n:>8.3f}{f'{lo:.3f}-{hi:.3f}':>16}")

    print("\nControls (source passage shown as the candidate; most should be B = 1)")
    for v in VOICES + ["all"]:
        rows = [r for r in key.values() if r["kind"] == "control" and v in ("all", r["voice"])]
        k, n, lo, hi = rate(rows)
        print(f"  {v:<22}{k}/{n}")

    print("\nDescriptive only (counts too small for rates): B = 1 negatives by stratum and rank")
    neg = [r for r in key.values() if r["kind"] == "neg"]
    for label, sel in (("stratum in", lambda r: r["stratum"] == "in"),
                       ("stratum out", lambda r: r["stratum"] == "out"),
                       ("rank 10-14", lambda r: r["baseline_rank"] <= 14),
                       ("rank 15+", lambda r: r["baseline_rank"] >= 15)):
        rows = [r for r in neg if sel(r)]
        k = sum(int(labels[r["item"]]) for r in rows)
        print(f"  {label:<14}{k}/{len(rows)}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("sample")
    s.add_argument("--n-neg", type=int, default=40)
    s.add_argument("--n-ctrl", type=int, default=4)
    s.add_argument("--seed", type=int, default=DEFAULT_SEED)
    sub.add_parser("analyze")
    a = ap.parse_args()
    sample(a) if a.cmd == "sample" else analyze(a)
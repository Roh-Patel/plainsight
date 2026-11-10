"""Blinded hand check of the source-rank question.
sample:  draws N queries per voice x stratum under hybrid_base, writes review files.
analyze: validates the graded TSV and prints the stratum comparison."""
import argparse
import csv
import json
import random
from pathlib import Path

import numpy as np

PASSAGES = Path("data/processed/passages.jsonl")
QUERIES = Path("data/synthetic/queries.jsonl")
RANKS = Path("data/synthetic/ranks/ranks.jsonl")
TOP = Path("data/synthetic/ranks/hybrid_base_top100.npy")
OUT = Path("scratch_synth")
TSV = OUT / "rank_review.tsv"
KEY = OUT / "rank_review_key.json"
SEED = 15
VOICES = ("layperson", "confused_undergrad", "half_remembered_term")


def read_jsonl(p):
    with open(p, encoding="utf-8") as f:
        return [json.loads(l) for l in f]


def block(p):
    return f"[{p['paper_id']} | {p['section_title'] or '(no heading)'}]\n{p['text']}\n"


def sample(args):
    if TSV.exists() and not args.force:
        raise SystemExit(f"{TSV} exists; refusing to overwrite graded work (use --force)")
    qs, rk, top = read_jsonl(QUERIES), read_jsonl(RANKS), np.load(TOP)
    if not (len(qs) == len(rk) == len(top)) or any(q["qid"] != r["qid"] for q, r in zip(qs, rk)):
        raise SystemExit("queries, ranks and top-100 file are not aligned")
    cells = {}
    for i, (q, r) in enumerate(zip(qs, rk)):
        if not q["kept"]:
            continue
        s = r["hybrid_base_src_rank"]
        cells.setdefault((q["voice"], "in" if s is not None and s <= 50 else "out"), []).append(i)
    rng = random.Random(SEED)
    picked = []
    for key in sorted(cells):
        picked += [(key, i) for i in rng.sample(cells[key], args.per_cell)]
    rng.shuffle(picked)

    comp = {}
    for (_, st), i in picked:
        if st == "out":
            src = rk[i]["src_point_id"]
            comp[i] = next(int(p) for p in top[i] if p >= 0 and p != src)
    need = {rk[i]["src_point_id"] for _, i in picked} | set(comp.values())
    texts = {}
    with open(PASSAGES, encoding="utf-8") as f:
        for n, line in enumerate(f):
            if n in need:
                texts[n] = json.loads(line)
    for _, i in picked:
        if texts[rk[i]["src_point_id"]]["passage_id"] != qs[i]["source_passage_id"]:
            raise SystemExit(f"source mismatch for {qs[i]['qid']}")

    OUT.mkdir(exist_ok=True)
    with open(OUT / "rank_review_AN.txt", "w", encoding="utf-8", newline="\n") as f:
        for _, i in picked:
            f.write(f"=== {qs[i]['qid']} ===\nQUERY: {qs[i]['query']}\n"
                    f"SOURCE PASSAGE:\n{block(texts[rk[i]['src_point_id']])}\n")
    with open(OUT / "rank_review_B.txt", "w", encoding="utf-8", newline="\n") as f:
        for (_, st), i in picked:
            if st == "out":
                f.write(f"=== {qs[i]['qid']} ===\nQUERY: {qs[i]['query']}\n"
                        f"OTHER PASSAGE:\n{block(texts[comp[i]])}\n")
    with open(TSV, "w", encoding="utf-8", newline="\n") as f:
        f.write("qid\tA\tN\tB\n")
        for (_, st), i in picked:
            f.write(f"{qs[i]['qid']}\t\t\t{'' if st == 'out' else '-'}\n")
    KEY.write_text(json.dumps(
        {qs[i]["qid"]: {"voice": v, "stratum": st, "hybrid_src_rank": rk[i]["hybrid_base_src_rank"],
                        "competitor_point_id": comp.get(i)} for (v, st), i in picked}, indent=1),
        encoding="utf-8")
    print("population per cell:", {f"{v}/{s}": len(x) for (v, s), x in sorted(cells.items())})
    print(f"sampled {len(picked)} (seed {SEED}), B file rows {len(comp)}")


def boot_diff(a, b, n_boot=10000, seed=0):
    rng = np.random.default_rng(seed)
    ia = rng.integers(0, len(a), (n_boot, len(a)))
    ib = rng.integers(0, len(b), (n_boot, len(b)))
    d = a[ia].mean(axis=1) - b[ib].mean(axis=1)
    return a.mean() - b.mean(), *np.percentile(d, [2.5, 97.5])


def analyze(args):
    key = json.loads(KEY.read_text(encoding="utf-8"))
    with open(TSV, encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f, delimiter="\t"))
    if len(rows) != len(key) or {r["qid"] for r in rows} != set(key):
        raise SystemExit("qids in the TSV do not match the key")
    data = {"in": [], "out": []}
    for r in rows:
        k = key[r["qid"]]
        a, n, b = r["A"].strip(), r["N"].strip(), r["B"].strip()
        if a not in ("0", "1", "2") or n not in ("0", "1"):
            raise SystemExit(f"bad A or N for {r['qid']}: {a!r} {n!r}")
        if k["stratum"] == "out" and b not in ("0", "1"):
            raise SystemExit(f"bad B for {r['qid']}: {b!r}")
        if k["stratum"] == "in" and b != "-":
            raise SystemExit(f"B must be '-' for {r['qid']}")
        data[k["stratum"]].append({"voice": k["voice"], "A": int(a), "N": int(n),
                                   "B": int(b) if b != "-" else None})
    for st in ("in", "out"):
        d = data[st]
        print(f"{st}: n={len(d)}  A=2 {sum(x['A'] == 2 for x in d)}  A=1 {sum(x['A'] == 1 for x in d)}  "
              f"A=0 {sum(x['A'] == 0 for x in d)}  N=1 {sum(x['N'] == 1 for x in d)}")
    for name, f in (("A=2", lambda x: float(x["A"] == 2)), ("N=1", lambda x: float(x["N"] == 1))):
        a = np.array([f(x) for x in data["out"]])
        b = np.array([f(x) for x in data["in"]])
        d, lo, hi = boot_diff(a, b)
        print(f"{name}: out {a.mean():.3f}  in {b.mean():.3f}  out-in {d:+.3f}  95% CI [{lo:+.3f}, {hi:+.3f}]")
    print("per voice (descriptive only), A=2 / n:")
    for v in VOICES:
        o = [x for x in data["out"] if x["voice"] == v]
        i = [x for x in data["in"] if x["voice"] == v]
        print(f"  {v}: out {sum(x['A'] == 2 for x in o)}/{len(o)}  in {sum(x['A'] == 2 for x in i)}/{len(i)}")
        print(f"  {v}: B=1 in out {sum(x['B'] for x in o)}/{len(o)}")
    bs = [x["B"] for x in data["out"]]
    print(f"B=1 (another passage also answers) in out: {sum(bs)}/{len(bs)}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=("sample", "analyze"))
    ap.add_argument("--per-cell", type=int, default=20)
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()
    sample(a) if a.mode == "sample" else analyze(a)
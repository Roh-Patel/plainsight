"""Can bge-reranker-base act as a false-negative filter for hard negatives?

Part 1 (CPU): positive-vs-candidate separation per voice and stratum, from hn_scored.jsonl.
Part 2 (GPU): the 120 hand-graded rank-review queries. Scores the best-ranked
non-source passage ("competitor", hand label B) and the source passage for each query.
Writes nothing. Run from the repo root: python -m analysis.negatives.calibrate_filter
"""
import json
import random
from collections import defaultdict
from pathlib import Path

import numpy as np

HN = Path("data/synthetic/hn_scored.jsonl")
QUERIES = Path("data/synthetic/queries.jsonl")
SRC_IDS = Path("data/synthetic/ranks/ranks.jsonl")
KEY = Path("scratch_synth/rank_review_key.json")
TSV = Path("scratch_synth/rank_review.tsv")

VOICES = ["layperson", "confused_undergrad", "half_remembered_term"]
THRESHOLDS = (-8, -6, -4, -2, 0, 2)
SEED = 19
N_BOOT = 2000


def read_jsonl(path):
    with open(path, encoding="utf-8") as f:
        for line in f:
            yield json.loads(line)


def auc(pos, neg):
    pos, neg = np.asarray(pos, float), np.asarray(neg, float)
    gt = (pos[:, None] > neg[None, :]).mean()
    eq = (pos[:, None] == neg[None, :]).mean()
    return gt + 0.5 * eq


def part1():
    rank_info = {r["qid"]: (r["voice"], r["hybrid_base_src_rank"]) for r in read_jsonl(SRC_IDS)}
    acc = defaultdict(lambda: {"pos": [], "cand": [], "pct": []})
    for row in read_jsonl(HN):
        voice, sr = rank_info[row["qid"]]
        stratum = "in" if (sr is not None and sr <= 50) else "out"
        a = acc[(voice, stratum)]
        a["pos"].append(row["pos_ce_score"])
        if row["cands"]:
            c = np.array([x[2] for x in row["cands"]], dtype=float)
            a["cand"].append(c)
            a["pct"].append(float((c < row["pos_ce_score"]).mean()))

    print("PART 1: positive vs pool-C candidates (all kept queries)")
    print("pct = mean share of a query's candidates scoring below its positive (0.5 if the reranker is blind)")
    print(f"{'voice':<22}{'strat':<6}{'n':>7}{'pos med':>9}{'cand med':>9}{'pct':>7}"
          f"{'c p90':>8}{'c p95':>8}{'c p99':>8}")
    for voice in VOICES:
        for stratum in ("in", "out"):
            a = acc.get((voice, stratum))
            if not a:
                continue
            cand = np.concatenate(a["cand"])
            p90, p95, p99 = np.percentile(cand, [90, 95, 99])
            print(f"{voice:<22}{stratum:<6}{len(a['pos']):>7}{np.median(a['pos']):>9.2f}"
                  f"{np.median(cand):>9.2f}{np.mean(a['pct']):>7.3f}"
                  f"{p90:>8.2f}{p95:>8.2f}{p99:>8.2f}")


def part2():
    key = json.loads(KEY.read_text(encoding="utf-8"))
    labels = {}
    with open(TSV, encoding="utf-8") as f:
        header = f.readline().rstrip("\n").split("\t")
        for line in f:
            parts = line.rstrip("\n").split("\t")
            parts += [""] * (len(header) - len(parts))
            d = dict(zip(header, parts))
            labels[d["qid"]] = d

    qtext = {r["qid"]: r["query"] for r in read_jsonl(QUERIES)}
    src = {r["qid"]: r["src_point_id"] for r in read_jsonl(SRC_IDS)}

    from retrieval.rerank import Reranker
    rr = Reranker()
    qids = list(key)
    qs, ps, tags = [], [], []
    for qid in qids:
        q = rr._cap_query(qtext[qid])
        cid = key[qid]["competitor_point_id"]
        if cid is not None:  # in-stratum rows have no competitor
            qs.append(q)
            ps.append(rr.texts[cid])
            tags.append((qid, "comp"))
        qs.append(q)
        ps.append(rr.texts[src[qid]])
        tags.append((qid, "src"))
    scores = rr.score_pairs(qs, ps, count_truncation=False)
    comp, srcs = {}, {}
    for (qid, kind), sc in zip(tags, scores):
        (comp if kind == "comp" else srcs)[qid] = sc

    print("\nPART 2: hand-graded rank-review queries (D045)")
    out = [q for q in qids if q in comp and labels[q].get("B", "") != ""]
    b = {q: int(labels[q]["B"]) for q in out}
    pos = [q for q in out if b[q] == 1]
    neg = [q for q in out if b[q] == 0]
    print(f"out-stratum queries with a B label: {len(out)}; B=1 (competitor answers the query): {len(pos)}")
    for v in VOICES:
        vq = [q for q in out if key[q]["voice"] == v]
        print(f"  {v}: {len(vq)} labelled, {sum(b[q] for q in vq)} with B=1")
    if not pos or not neg:
        print("need both B=1 and B=0 rows to continue")
        return

    sp = [comp[q] for q in pos]
    sn = [comp[q] for q in neg]
    point = auc(sp, sn)
    rng = random.Random(SEED)
    boots = []
    for _ in range(N_BOOT):
        sample = [rng.choice(out) for _ in out]
        p = [comp[q] for q in sample if b[q] == 1]
        n = [comp[q] for q in sample if b[q] == 0]
        if p and n:
            boots.append(auc(p, n))
    lo, hi = np.percentile(boots, [2.5, 97.5])
    print(f"\ncompetitor logit, B=1 vs B=0: AUC {point:.3f} (95% bootstrap CI {lo:.3f} to {hi:.3f}, "
          f"resampling queries, seed {SEED})")
    print(f"B=1 logits (sorted): {' '.join(f'{x:.1f}' for x in sorted(sp))}")
    print(f"B=0 logits: median {np.median(sn):.2f}, p75 {np.percentile(sn, 75):.2f}, "
          f"p90 {np.percentile(sn, 90):.2f}, max {max(sn):.2f}")

    print("\nabsolute cut: drop a candidate if its logit is above t")
    print(f"{'t':>5}{'B=1 above':>12}{'B=0 above':>12}")
    for t in THRESHOLDS:
        print(f"{t:>5}{sum(x > t for x in sp):>8}/{len(sp):<3}{sum(x > t for x in sn):>8}/{len(sn):<3}")

    print("\nsource logit by hand grade A (all 120 queries): A=2 vs A<2")
    for v in VOICES:
        a2 = [srcs[q] for q in qids if key[q]["voice"] == v and labels[q]["A"] == "2"]
        a_lt = [srcs[q] for q in qids if key[q]["voice"] == v and labels[q]["A"] != "2"]
        m2 = f"{np.mean(a2):.2f}" if a2 else "n/a"
        ml = f"{np.mean(a_lt):.2f}" if a_lt else "n/a"
        print(f"  {v}: A=2 n={len(a2)} mean {m2}; A<2 n={len(a_lt)} mean {ml}")


if __name__ == "__main__":
    part1()
    part2()
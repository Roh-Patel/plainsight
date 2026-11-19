import json
import random

SEED = 28
CAP = 7


def load_rows():
    kept = []
    with open("data/synthetic/queries.jsonl", encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            if r["kept"]:
                kept.append(r)
    with open("data/synthetic/hard_negatives.jsonl", encoding="utf-8") as f:
        hn = [json.loads(line) for line in f]
    assert len(hn) == len(kept), (len(hn), len(kept))
    rows = []
    for q, h in zip(kept, hn):
        assert h["qid"] == q["qid"]
        rows.append({"qid": q["qid"], "voice": q["voice"], "query": q["query"],
                     "pos": q["source_passage_id"],
                     "negs": [n["passage_id"] for n in h["negatives"]]})
    return rows


def epoch_groups(rows, epoch):
    """One (qid, positive, negatives) per query that has at least one negative."""
    out = []
    for r in rows:
        if not r["negs"]:
            continue
        rng = random.Random("%d-%d-%s" % (SEED, epoch, r["qid"]))
        negs = rng.sample(r["negs"], min(len(r["negs"]), CAP))
        out.append((r["qid"], r["pos"], negs))
    return out


if __name__ == "__main__":
    rows = load_rows()
    voices = sorted({r["voice"] for r in rows})
    print("queries", len(rows))
    print("no-negative by voice", {v: sum(1 for r in rows if r["voice"] == v and not r["negs"])
                                  for v in voices})
    g0 = epoch_groups(rows, 0)
    print("groups", len(g0), "| pairs per epoch", sum(1 + len(g[2]) for g in g0))
    assert all(g[1] not in g[2] and len(set(g[2])) == len(g[2]) for g in g0)
    assert g0 == epoch_groups(rows, 0), "not deterministic"
    seen, total = set(), sum(len(r["negs"]) for r in rows)
    over = [r for r in rows if len(r["negs"]) > CAP]
    for e in range(3):
        for qid, _, negs in epoch_groups(rows, e):
            seen.update((qid, n) for n in negs)
        if e == 0:
            g1 = {g[0]: set(g[2]) for g in epoch_groups(rows, 1)}
            same = sum(1 for g in g0 if g[0] in g1 and set(g[2]) == g1[g[0]] and len(set(g[2])) == CAP)
            print("queries with >7 negatives:", len(over), "| identical draw in epochs 0 and 1:", same)
    print("negatives seen at least once in 3 epochs: %.3f of %d" % (len(seen) / total, total))
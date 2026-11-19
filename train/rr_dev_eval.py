import json, math, os, sys, time, argparse
import numpy as np

from retrieval.rerank import Reranker
from retrieval.common import load_passages

CAND = "data/synthetic_val/rr_dev_candidates.jsonl"
OUTD = "data/synthetic_val/rr_dev"
DEPTH = 50


def ndcg(rank):
    return 1.0 / math.log2(rank + 1) if rank and rank <= 10 else 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=None)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()
    out = os.path.join(OUTD, a.tag + ".json")
    if os.path.exists(out) and not a.limit:
        sys.exit(out + " exists, refusing to overwrite")
    os.makedirs(OUTD, exist_ok=True)
    rows = [json.loads(l) for l in open(CAND, encoding="utf-8")]
    if a.limit:
        rows = rows[:a.limit]
    kw = {"model_name": a.model} if a.model else {}
    rr = Reranker(texts=[r["text"] for r in load_passages(("text",))], **kw)
    t = time.time()
    res = rr.rerank([r["query"] for r in rows],
                    [[(p, 0.0) for p in r["cands"]] for r in rows],
                    depth=DEPTH, batch_size=32)
    secs = time.time() - t
    ranks = []
    for r, ranked in zip(rows, res):
        ids = [p for p, _ in ranked]
        ranks.append(ids.index(r["src_point_id"]) + 1 if r["src_point_id"] in ids else None)
    m = {}
    for v in ("ALL", "layperson", "confused_undergrad", "half_remembered_term"):
        ix = [i for i, r in enumerate(rows) if v == "ALL" or r["voice"] == v]
        rk = [ranks[i] for i in ix]
        m[v] = {"ndcg@10": float(np.mean([ndcg(x) for x in rk])),
                "mrr@10": float(np.mean([1 / x if x and x <= 10 else 0 for x in rk])),
                "recall@10": float(np.mean([bool(x and x <= 10) for x in rk])),
                "in_top50": float(np.mean([x is not None for x in rk]))}
    print(a.tag, "seconds %.0f" % secs, "pairs", rr.stats["pairs"], rr.stats)
    for v, d in m.items():
        print(v, {k: round(x, 4) for k, x in d.items()})
    if a.limit:
        print("limit run, nothing written")
        return
    json.dump({"tag": a.tag, "model": a.model or "base", "n_queries": len(rows),
               "metrics": m, "qids": [r["qid"] for r in rows], "ranks": ranks,
               "seconds": secs}, open(out, "w", encoding="utf-8", newline="\n"))
    print("wrote", out)


if __name__ == "__main__":
    main()
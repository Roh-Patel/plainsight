"""Time bge-reranker-base on real pool-C pairs and project the full scoring run.

GPU. Writes nothing. Run from the repo root: python -m analysis.negatives.time_reranker
Pool C = hybrid_base ranks 10-50, train papers only, minus the source paper.
"""
import json
import random
import time
from pathlib import Path

import numpy as np
import torch

from retrieval.rerank import Reranker

PASSAGES = Path("data/processed/passages.jsonl")
SPLITS = Path("data/processed/splits.json")
QUERIES = Path("data/synthetic/queries.jsonl")
RANKS_MANIFEST = Path("data/synthetic/ranks/ranks.manifest.json")
TOP100 = Path("data/synthetic/ranks/hybrid_base_top100.npy")
SRC_IDS = Path("data/synthetic/ranks/ranks.jsonl")

LO, HI = 10, 50  # 1-based inclusive rank window
N_SAMPLE = 100   # queries in the timing sample
SEED = 18
BATCH_SIZES = (32, 64, 128)
CAPS = (10, 20, 30, None)  # candidates scored per query; None = all of pool C


def read_jsonl(path):
    with open(path, encoding="utf-8") as f:
        for line in f:
            yield json.loads(line)


def main():
    manifest = json.loads(RANKS_MANIFEST.read_text(encoding="utf-8"))
    splits = json.loads(SPLITS.read_text(encoding="utf-8"))

    paper_ids = [r["paper_id"] for r in read_jsonl(PASSAGES)]
    if len(paper_ids) != manifest["passages_lines"]:
        raise SystemExit("passages.jsonl line count does not match the ranks manifest")
    uniq = {p: i for i, p in enumerate(sorted(set(paper_ids)))}
    paper = np.array([uniq[p] for p in paper_ids], dtype=np.int32)
    paper_is_train = np.zeros(len(uniq), dtype=bool)
    for p in splits["train"]:
        paper_is_train[uniq[p]] = True

    queries = list(read_jsonl(QUERIES))
    src_ids = [r["src_point_id"] for r in read_jsonl(SRC_IDS)]
    top = np.load(TOP100)
    if not (len(queries) == len(src_ids) == top.shape[0] == manifest["n"]):
        raise SystemExit("queries, ranks and npy row counts disagree")

    def pool_c(i):
        raw = top[i, LO - 1:HI]
        win = raw[raw >= 0]
        p = paper[win]
        keep = paper_is_train[p] & (p != paper[src_ids[i]]) & (win != src_ids[i])
        return win[keep].tolist()  # rank order preserved

    kept = [i for i, q in enumerate(queries) if q["kept"]]
    pools = {i: pool_c(i) for i in kept}
    sizes = np.array([len(pools[i]) for i in kept])
    print(f"kept queries {len(kept)}; pool C size mean {sizes.mean():.2f}, "
          f"min {sizes.min()}, max {sizes.max()}")

    sample = random.Random(SEED).sample(kept, N_SAMPLE)

    rr = Reranker()
    rr.stats = {"pairs": 0, "passage_truncated": 0, "query_truncated": 0}
    qs, ps = [], []
    for i in sample:
        q = rr._cap_query(queries[i]["query"])
        for pid in pools[i]:
            qs.append(q)
            ps.append(rr.texts[pid])
    print(f"timing sample: {N_SAMPLE} queries, {len(qs)} pairs")

    # counting run (also the warm-up)
    rr.score_pairs(qs, ps, batch_size=32, count_truncation=True)
    stats = dict(rr.stats)
    print(f"truncation on the sample: passages {stats['passage_truncated']} of "
          f"{stats['pairs']} ({stats['passage_truncated'] / stats['pairs']:.4f}), "
          f"queries capped {stats['query_truncated']}")

    best = 0.0
    print("\nbatch  pairs/s  peak_GB")
    for bs in BATCH_SIZES:
        torch.cuda.reset_peak_memory_stats()
        t = time.perf_counter()
        rr.score_pairs(qs, ps, batch_size=bs, count_truncation=False)
        dt = time.perf_counter() - t
        rate = len(qs) / dt
        best = max(best, rate)
        peak = torch.cuda.max_memory_allocated() / 1e9
        print(f"{bs:>5}  {rate:>7.1f}  {peak:>7.2f}")

    print(f"\nprojection at {best:.1f} pairs/s (best batch size), "
          f"plus {len(kept)} positive pairs:")
    print(f"{'cap':>6}{'pairs':>12}{'hours':>8}")
    for cap in CAPS:
        n = int((sizes if cap is None else np.minimum(sizes, cap)).sum()) + len(kept)
        label = "all" if cap is None else str(cap)
        print(f"{label:>6}{n:>12}{n / best / 3600:>8.2f}")


if __name__ == "__main__":
    main()
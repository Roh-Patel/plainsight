import random

from retrieval.bm25 import BM25Searcher
from retrieval.common import load_passages
from retrieval.dense import DenseSearcher
from retrieval.hybrid import HybridSearcher
from retrieval.rerank import Reranker

QUERIES = ["how do astronomers find planets by watching a star get dimmer",
           "why are some gas giants orbiting so close to their star",
           "what is the atmosphere of a hot jupiter made of"]


def asc(s):
    return s.encode("ascii", "replace").decode()


rows = load_passages(("passage_id", "section_title", "text"))
bm, dn = BM25Searcher(), DenseSearcher(backend="numpy")
hy = HybridSearcher(bm, dn)
rr = Reranker(texts=[r["text"] for r in rows])

fused = hy.search(QUERIES)
reranked = rr.rerank(QUERIES, fused, depth=50)

for q, f, r in zip(QUERIES, fused, reranked):
    frank = {p: i + 1 for i, (p, _) in enumerate(f)}
    print("\nQ:", q)
    print("  results:", len(r))
    for p, s in r[:3]:
        print(f"    {s:7.3f} (fused rank {frank[p]:2d}) {rows[p]['passage_id']} | "
              f"{asc(rows[p]['section_title'])[:40]}")

print("\nstats:", rr.stats)

# sanity 1: top reranked logit should sit well above random passages
rng = random.Random(0)
rand = rng.sample(range(len(rows)), 20)
rand_scores = rr.score_pairs([QUERIES[0]] * 20, [rows[i]["text"] for i in rand])
print(f"sanity: q1 top logit {reranked[0][0][1]:.2f}, "
      f"random-passage logits min/mean/max "
      f"{min(rand_scores):.2f}/{sum(rand_scores)/20:.2f}/{max(rand_scores):.2f}")

# sanity 2: batch composition must not change the ranking
cands = fused[0][:50]
a = rr.rerank([QUERIES[0]], [cands], batch_size=50, count_truncation=False)[0]
b = rr.rerank([QUERIES[0]], [cands], batch_size=7, count_truncation=False)[0]
da = dict(a)
diff = max(abs(da[p] - s) for p, s in b)
print(f"batch 50 vs 7: max logit diff {diff:.4f}, "
      f"top10 identical order {[p for p, _ in a[:10]] == [p for p, _ in b[:10]]}")
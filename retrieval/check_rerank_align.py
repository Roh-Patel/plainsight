from retrieval.bm25 import BM25Searcher
from retrieval.common import load_passages
from retrieval.dense import DenseSearcher
from retrieval.hybrid import HybridSearcher
from retrieval.rerank import Reranker

Q = "how do astronomers find planets by watching a star get dimmer"


def asc(s):
    return s.encode("ascii", "replace").decode()


rows = load_passages(("passage_id", "section_title", "text"))
rr = Reranker(texts=[r["text"] for r in rows])
hy = HybridSearcher(BM25Searcher(), DenseSearcher(backend="numpy"))
fused = hy.search([Q])[0]
reranked = rr.rerank([Q], [fused], depth=50)[0]
rpos = {p: (i + 1, s) for i, (p, s) in enumerate(reranked)}

print("fused top 5 -> reranked rank, logit")
for i, (p, _) in enumerate(fused[:5], 1):
    r, s = rpos[p]
    print(f"  fused {i} {rows[p]['passage_id']} -> rank {r}, {s:.2f}")

for label, p in [("reranked top1", reranked[0][0]), ("fused top1", fused[0][0])]:
    solo = rr.score_pairs([Q], [rows[p]["text"]], batch_size=1, count_truncation=False)[0]
    print(f"\n{label} {rows[p]['passage_id']}: batched {rpos[p][1]:.3f}, solo {solo:.3f}")
    print("   ", asc(rows[p]["text"][:500]).replace("\n", " "))
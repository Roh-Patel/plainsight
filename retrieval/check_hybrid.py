from retrieval.bm25 import BM25Searcher
from retrieval.common import load_passages
from retrieval.dense import DenseSearcher
from retrieval.hybrid import HybridSearcher

QUERIES = ["how do astronomers find planets by watching a star get dimmer",
           "why are some gas giants orbiting so close to their star",
           "what is the atmosphere of a hot jupiter made of"]

rows = load_passages(("passage_id", "section_title"))
bm, dn = BM25Searcher(), DenseSearcher(backend="numpy")
hy = HybridSearcher(bm, dn)

for q, b, d, h in zip(QUERIES, bm.search(QUERIES), dn.search(QUERIES), hy.search(QUERIES)):
    bs, ds = {p for p, _ in b}, {p for p, _ in d}
    print("\nQ:", q)
    print(f"  bm25/dense top100 overlap: {len(bs & ds)}; fused size: {len(h)}")
    top10 = [p for p, _ in h[:10]]
    print(f"  fused top10 from: bm25 only {sum(p in bs and p not in ds for p in top10)}, "
          f"dense only {sum(p in ds and p not in bs for p in top10)}, "
          f"both {sum(p in bs and p in ds for p in top10)}")
    for p, s in h[:3]:
        print(f"    {s:.5f} {rows[p]['passage_id']} | "
              f"{rows[p]['section_title'].encode('ascii', 'replace').decode()[:45]}")
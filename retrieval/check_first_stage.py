import random

import numpy as np

from retrieval.bm25 import BM25Searcher
from retrieval.common import load_passages
from retrieval.dense import DenseSearcher

QUERIES = ["how do astronomers find planets by watching a star get dimmer",
           "why are some gas giants orbiting so close to their star",
           "what is the atmosphere of a hot jupiter made of"]


def asc(s):
    return s.encode("ascii", "replace").decode()


def show(name, rows, hits):
    print(f"  [{name}]")
    for pid, s in hits[:3]:
        r = rows[pid]
        print(f"    {s:.3f} {r['passage_id']} | {asc(r['section_title'])[:45]}")


rows = load_passages(("passage_id", "section_title", "text"))
bm = BM25Searcher()
dn = DenseSearcher(backend="numpy")
dq = DenseSearcher(backend="qdrant", model=dn.model)

for q, b, n, d in zip(QUERIES, bm.search(QUERIES), dn.search(QUERIES), dq.search(QUERIES)):
    print("\nQ:", q)
    show("bm25", rows, b)
    show("dense numpy", rows, n)
    show("dense qdrant", rows, d)

print("\nBM25 edge cases (expect [[], [], []]):",
      [len(x) for x in bm.search(["zzzqqqxx", "the of and", ""])])

rng = random.Random(0)
sample = rng.sample(range(len(rows)), 50)
qs = [rows[i]["text"][:200] for i in sample]
rn, rq = dn.search(qs), dq.search(qs)
o10 = [len({a for a, _ in x[:10]} & {a for a, _ in y[:10]}) / 10 for x, y in zip(rn, rq)]
o100 = [len({a for a, _ in x} & {a for a, _ in y}) / 100 for x, y in zip(rn, rq)]
same = [[a for a, _ in x[:10]] == [a for a, _ in y[:10]] for x, y in zip(rn, rq)]
diff = max(abs(s1 - dict(y)[a]) for x, y in zip(rn, rq) for a, s1 in x[:10] if a in dict(y))
print(f"numpy vs qdrant over 50 queries: top10 overlap {np.mean(o10):.3f}, "
      f"top100 overlap {np.mean(o100):.4f}, top10 identical order {np.mean(same):.2f}, "
      f"max top10 score diff {diff:.5f}")
import json
import random

from retrieval.dense import DenseSearcher

with open("data/processed/passages.jsonl", encoding="utf-8") as f:
    idx = {json.loads(l)["passage_id"]: i for i, l in enumerate(f)}

rows = []
with open("data/synthetic_val/queries.jsonl", encoding="utf-8") as f:
    for line in f:
        r = json.loads(line)
        if r["kept"]:
            rows.append((r["query"], idx[r["source_passage_id"]]))
random.Random(1).shuffle(rows)
rows = rows[:500]
queries = [q for q, _ in rows]

for name in ["dense_base", "dense_ft"]:
    s = DenseSearcher(name, backend="numpy")
    res = s.search(queries, k=100)
    r10 = sum(src in [p for p, _ in x[:10]] for x, (_, src) in zip(res, rows))
    r100 = sum(src in [p for p, _ in x] for x, (_, src) in zip(res, rows))
    print(f"{name}: source in top 10 {r10}/500, in top 100 {r100}/500")
    del s
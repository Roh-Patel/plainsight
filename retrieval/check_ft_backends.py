import json
import random

from retrieval.dense import DenseSearcher

rows = []
with open("data/synthetic_val/queries.jsonl", encoding="utf-8") as f:
    for line in f:
        r = json.loads(line)
        if r["kept"]:
            rows.append(r["query"])
random.Random(0).shuffle(rows)
queries = rows[:200]

num = DenseSearcher("dense_ft", backend="numpy")
qd = DenseSearcher("dense_ft", backend="qdrant", model=num.model)
a = num.search(queries, k=100)
b = qd.search(queries, k=100)

same_order10 = sum([p for p, _ in x[:10]] == [p for p, _ in y[:10]] for x, y in zip(a, b))
ov100 = [len({p for p, _ in x} & {p for p, _ in y}) / 100 for x, y in zip(a, b)]
maxdiff = max(abs(s1 - s2) for x, y in zip(a, b)
              for (p1, s1), (p2, s2) in zip(x, y) if p1 == p2)
print("queries:", len(queries))
print("top-10 identical order:", same_order10, "/", len(queries))
print("top-100 overlap mean/min: %.4f / %.2f" % (sum(ov100) / len(ov100), min(ov100)))
print("max score diff on shared ids: %.6f" % maxdiff)
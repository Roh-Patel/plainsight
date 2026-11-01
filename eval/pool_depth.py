import json
from statistics import mean

SYSTEMS = ["bm25", "dense_base", "hybrid_base", "hybrid_base_rerank_base"]
runs = {}
for s in SYSTEMS:
    runs[s] = {}
    for line in open(f"runs/{s}.jsonl", encoding="utf-8"):
        r = json.loads(line)
        runs[s][r["qid"]] = [x["passage_id"] for x in r["results"]]
qids = list(runs[SYSTEMS[0]])

for depth in (5, 10):
    pool = {q: set().union(*(runs[s][q][:depth] for s in SYSTEMS)) for q in qids}
    sizes = [len(p) for p in pool.values()]
    print(f"\npool depth {depth}: {sum(sizes)} pairs total, mean {mean(sizes):.1f}/question, "
          f"min {min(sizes)}, max {max(sizes)}")
    for s in SYSTEMS:
        j10 = mean(len(set(runs[s][q][:10]) & pool[q]) / 10 for q in qids)
        j20 = mean(len(set(runs[s][q][:20]) & pool[q]) / 20 for q in qids)
        print(f"  {s:26s} judged@10 {j10:.3f}  judged@20 {j20:.3f}")
"""Rank of each synthetic query's source passage under bm25, dense_base, hybrid_base."""
import json
import statistics as st
import sys
from pathlib import Path

from retrieval.bm25 import BM25Searcher
from retrieval.dense import DenseSearcher
from retrieval.hybrid import HybridSearcher
from train.query_prompt import VOICES

sys.stdout.reconfigure(errors="replace")
raw_path, out_path = sys.argv[1], Path(sys.argv[2])
if out_path.exists():
    sys.exit(f"{out_path} exists; refusing to overwrite")

rows = [json.loads(l) for l in open(raw_path, encoding="utf-8") if l.strip()]
items = []
for r in rows:
    if r["parsed"] and r["parsed"]["usable"]:
        for v in VOICES:
            items.append((r["passage_id"], v, r["parsed"]["queries"][v]))
print(f"{len(items)} queries from {len(items)//3} passages")

pid_of = []
with open("data/processed/passages.jsonl", encoding="utf-8") as f:
    for line in f:
        pid_of.append(json.loads(line)["passage_id"])
assert all("::" in p for p in pid_of[:100]) and len(pid_of) == 116356, len(pid_of)

queries = [q for _, _, q in items]
bm25 = BM25Searcher()
dense = DenseSearcher()
hybrid = HybridSearcher(bm25, dense)
res = {"bm25": bm25.search(queries, k=100),
       "dense_base": dense.search(queries, k=100),
       "hybrid_base": hybrid.search(queries)}
for s, r in res.items():
    assert len(r) == len(items), s


def rank_of(hits, target):
    for i, (pt, _) in enumerate(hits, 1):
        if pid_of[pt] == target:
            return i
    return None


out = []
ranks = {s: [] for s in res}
for i, (pid, v, q) in enumerate(items):
    row = {"source_passage_id": pid, "voice": v, "query": q}
    for s in res:
        rk = rank_of(res[s][i], pid)
        row[f"rank_{s}"] = rk
        ranks[s].append(rk)
    out.append(row)
out_path.parent.mkdir(parents=True, exist_ok=True)
with out_path.open("w", encoding="utf-8", newline="\n") as f:
    for row in out:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")

print("\nshare of queries whose source passage is within rank N (not found = beyond 100)")
print(f"{'system':12s} {'voice':22s} {'n':>4s} {'@1':>6s} {'@10':>6s} {'@50':>6s} {'@100':>6s} {'median rank (found)':>20s}")
for s in res:
    for v in list(VOICES) + ["ALL"]:
        idx = [i for i, it in enumerate(items) if v == "ALL" or it[1] == v]
        rk = [ranks[s][i] for i in idx]
        n = len(rk)
        def share(N):
            return sum(1 for x in rk if x is not None and x <= N) / n
        found = sorted(x for x in rk if x is not None)
        med = st.median(found) if found else float("nan")
        print(f"{s:12s} {v:22s} {n:4d} {share(1):6.2f} {share(10):6.2f} {share(50):6.2f} {share(100):6.2f} {med:20.1f}")
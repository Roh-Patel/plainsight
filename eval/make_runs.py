"""Run systems 1-4 on the frozen questions and write runs/{system}.jsonl.

Refuses to overwrite existing run files without --force. Dry runs (--limit)
must write to a scratch directory, never to runs/.
"""
import argparse
import json
import sys
from pathlib import Path

from retrieval.bm25 import BM25Searcher
from retrieval.dense import DenseSearcher
from retrieval.hybrid import HybridSearcher
from retrieval.rerank import Reranker

QUESTIONS = Path("data/eval/questions.jsonl")
PASSAGES = Path("data/processed/passages.jsonl")
# (system name, expected results per question)
SYSTEMS = [("bm25", 100), ("dense_base", 100), ("hybrid_base", 100),
           ("hybrid_base_rerank_base", 50)]


def load_questions():
    with open(QUESTIONS, encoding="utf-8") as f:
        rows = [json.loads(line) for line in f if line.strip()]
    qids = [r["qid"] for r in rows]
    assert len(set(qids)) == len(qids), "duplicate qids"
    return rows


def load_passage_ids():
    with open(PASSAGES, encoding="utf-8") as f:
        return [json.loads(line)["passage_id"] for line in f]


def check(system, expected, hits_per_query):
    for i, hits in enumerate(hits_per_query):
        assert len(hits) == expected, f"{system} q{i}: {len(hits)} != {expected}"
        ids = [p for p, _ in hits]
        assert len(set(ids)) == len(ids), f"{system} q{i}: duplicate point ids"
        sc = [float(s) for _, s in hits]
        assert all(a >= b for a, b in zip(sc, sc[1:])), f"{system} q{i}: not sorted"


def write_run(path, system, qids, hits_per_query, pid_of):
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        for qid, hits in zip(qids, hits_per_query):
            row = {"qid": qid, "system": system,
                   "results": [{"passage_id": pid_of[p], "score": float(s)}
                               for p, s in hits]}
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default="runs")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()

    out = Path(a.out_dir)
    if a.limit and out.resolve() == Path("runs").resolve():
        sys.exit("--limit runs must not write to runs/")
    out.mkdir(parents=True, exist_ok=True)
    paths = {s: out / f"{s}.jsonl" for s, _ in SYSTEMS}
    existing = [str(p) for p in paths.values() if p.exists()]
    if existing and not a.force:
        sys.exit("refusing to overwrite: " + ", ".join(existing))

    rows = load_questions()
    if a.limit:
        rows = rows[:a.limit]
    qids = [r["qid"] for r in rows]
    queries = [r["question"] for r in rows]
    pid_of = load_passage_ids()
    print(f"{len(queries)} questions, {len(pid_of)} passages")

    bm25 = BM25Searcher()
    dense = DenseSearcher(backend="qdrant")
    hybrid = HybridSearcher(bm25, dense)

    results = {}
    results["bm25"] = bm25.search(queries, k=100)
    results["dense_base"] = dense.search(queries, k=100)
    results["hybrid_base"] = hybrid.search(queries)
    rer = Reranker()
    results["hybrid_base_rerank_base"] = rer.rerank(
        queries, results["hybrid_base"], depth=50)

    for system, expected in SYSTEMS:
        check(system, expected, results[system])
    for system, _ in SYSTEMS:
        write_run(paths[system], system, qids, results[system], pid_of)
        print("wrote", paths[system])
    print("Reranker.stats:", rer.stats)


if __name__ == "__main__":
    main()
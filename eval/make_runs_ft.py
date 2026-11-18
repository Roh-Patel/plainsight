"""Run dense_ft and hybrid_ft_embed on the frozen questions and write runs/{system}.jsonl.

Refuses to overwrite existing run files without --force. Dry runs (--limit)
must write to a scratch directory, never to runs/. Do not use --force once
pooling and labelling have started.
"""

import argparse
import sys
from pathlib import Path

from eval.make_runs import (check, load_passage_ids, load_questions,
                            write_run)
from retrieval.bm25 import BM25Searcher
from retrieval.dense import DenseSearcher
from retrieval.hybrid import HybridSearcher

SYSTEMS = [("dense_ft", 100), ("hybrid_ft_embed", 100)]


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
    dense = DenseSearcher("dense_ft", backend="qdrant")
    hybrid = HybridSearcher(bm25, dense)

    results = {}
    results["dense_ft"] = dense.search(queries, k=100)
    results["hybrid_ft_embed"] = hybrid.search(queries)

    for system, expected in SYSTEMS:
        check(system, expected, results[system])
    for system, _ in SYSTEMS:
        write_run(paths[system], system, qids, results[system], pid_of)
        print("wrote", paths[system])


if __name__ == "__main__":
    main()
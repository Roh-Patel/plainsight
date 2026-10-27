"""Per-query metrics for one run.."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from eval.loaders import load_passage_ids, load_qrels, load_run
from eval.metrics import (judged_at_k, mrr_at_k, ndcg_at_k, paper_hit_at_k,
                          recall_at_k)

METRICS = ["ndcg@10", "recall@10", "recall@20", "mrr@10", "paper_hit@10",
           "judged@10", "judged@20"]


def evaluate_run(run, qrels, min_grade: int = 2) -> dict[str, dict[str, float]]:
    missing = sorted(set(qrels) - set(run))
    if missing:
        raise ValueError(f"{len(missing)} qrels qids missing from run: {missing[:5]}")
    extra = sorted(set(run) - set(qrels))
    if extra:
        raise ValueError(f"{len(extra)} run qids have no qrels: {extra[:5]}")
    empty = sorted(q for q, j in qrels.items() if not any(g >= min_grade for g in j.values()))
    if empty:
        raise ValueError(f"{len(empty)} questions have no passage with grade >= "
                         f"{min_grade}: {empty}")
    out = {}
    for qid in sorted(qrels):
        r, j = run[qid], qrels[qid]
        out[qid] = {
            "ndcg@10": ndcg_at_k(r, j, 10),
            "recall@10": recall_at_k(r, j, 10, min_grade),
            "recall@20": recall_at_k(r, j, 20, min_grade),
            "mrr@10": mrr_at_k(r, j, 10, min_grade),
            "paper_hit@10": paper_hit_at_k(r, j, 10, min_grade),
            "judged@10": judged_at_k(r, j, 10),
            "judged@20": judged_at_k(r, j, 20),
        }
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--system", required=True)
    ap.add_argument("--runs-dir", default="runs")
    ap.add_argument("--qrels", default="data/eval/qrels.tsv")
    ap.add_argument("--passages", default="data/processed/passages.jsonl")
    ap.add_argument("--min-grade", type=int, default=2, choices=[1, 2])
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    valid = load_passage_ids(a.passages)
    qrels = load_qrels(a.qrels, valid)
    run = load_run(Path(a.runs_dir) / f"{a.system}.jsonl", valid, a.system)
    res = evaluate_run(run, qrels, a.min_grade)

    out = Path(a.out or f"results/per_query/{a.system}.jsonl")
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8", newline="\n") as f:
        for qid, m in res.items():
            f.write(json.dumps({"qid": qid, "system": a.system,
                                "min_grade": a.min_grade, **m}) + "\n")
    print(f"{a.system}: {len(res)} questions, min_grade={a.min_grade}, wrote {out}")
    for k in METRICS:
        print(f"  {k:13s} {sum(m[k] for m in res.values()) / len(res):.4f}")


if __name__ == "__main__":
    main()
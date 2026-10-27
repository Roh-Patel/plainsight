"""One-off cross-check of eval.metrics against ranx on random data.
Needs ranx, so it is not part of the test suite or requirements."""
from __future__ import annotations

import random
import sys

from ranx import Qrels, Run, evaluate

from eval.metrics import mrr_at_k, ndcg_at_k, recall_at_k

TOL = 1e-6


def make_data(seed: int = 0, n_queries: int = 300):
    rng = random.Random(seed)
    pool = [f"P{i // 5}::000::{i % 5:03d}" for i in range(400)]
    qrels, runs = {}, {}
    for q in range(n_queries):
        qid = f"q{q:03d}"
        judged = rng.sample(pool, rng.randint(2, 25))
        grades = {p: rng.choice([0, 0, 1, 2]) for p in judged}
        grades[judged[0]] = 2  # at least one grade-2 passage
        cands = list(dict.fromkeys(judged + rng.sample(pool, 80)))
        rng.shuffle(cands)
        qrels[qid] = grades
        runs[qid] = cands[: rng.randint(5, 100)]
    return qrels, runs


def ranx_scores(qrels_d, runs, metrics):
    run_d = {q: {p: float(len(r) - i) for i, p in enumerate(r)} for q, r in runs.items()}
    run = Run.from_dict(run_d)
    evaluate(Qrels.from_dict(qrels_d), run, metrics, return_mean=False)
    return {m: {q: float(run.scores[m][q]) for q in runs} for m in metrics}


def main() -> None:
    qrels, runs = make_data()
    graded = ranx_scores(qrels, runs, ["ndcg_burges@10"])
    binary_q = {q: {p: 1 for p, g in j.items() if g >= 2} for q, j in qrels.items()}
    binary = ranx_scores(binary_q, runs, ["recall@10", "recall@20", "mrr@10"])

    ours = {
        "ndcg_burges@10": lambda r, j: ndcg_at_k(r, j, 10),
        "recall@10": lambda r, j: recall_at_k(r, j, 10),
        "recall@20": lambda r, j: recall_at_k(r, j, 20),
        "mrr@10": lambda r, j: mrr_at_k(r, j, 10),
    }
    theirs = {**graded, **binary}
    ok = True
    for m, fn in ours.items():
        diffs = [abs(fn(runs[q], qrels[q]) - theirs[m][q]) for q in runs]
        worst = max(diffs)
        mean_val = sum(theirs[m].values()) / len(runs)
        print(f"{m:15s} mean {mean_val:.4f}  max abs diff {worst:.2e}  n={len(diffs)}")
        ok &= worst < TOL
    print("PASS" if ok else "FAIL")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
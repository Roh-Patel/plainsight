"""Ranking metrics over graded qrels. Pure functions, no I/O.

ranked: passage_ids, best first (list position is the rank).
qrel:   passage_id -> grade (0/1/2) for judged passages only.
Unjudged passages count as grade 0.
"""
from __future__ import annotations

import math
from typing import Mapping, Sequence


def gain(grade: int) -> float:
    return 2.0 ** grade - 1.0


def paper_of(passage_id: str) -> str:
    return passage_id.split("::", 1)[0]


def _relevant(qrel: Mapping[str, int], min_grade: int) -> set[str]:
    rel = {p for p, g in qrel.items() if g >= min_grade}
    if not rel:
        raise ValueError(f"no passage with grade >= {min_grade} in qrel")
    return rel


def _dcg(grades: Sequence[int], k: int) -> float:
    return sum(gain(g) / math.log2(i + 2) for i, g in enumerate(grades[:k]))


def ndcg_at_k(ranked: Sequence[str], qrel: Mapping[str, int], k: int = 10) -> float:
    ideal = _dcg(sorted(qrel.values(), reverse=True), k)
    if ideal == 0.0:
        raise ValueError("no relevant passage in qrel, nDCG undefined")
    return _dcg([qrel.get(p, 0) for p in ranked], k) / ideal


def recall_at_k(ranked: Sequence[str], qrel: Mapping[str, int], k: int,
                min_grade: int = 2) -> float:
    rel = _relevant(qrel, min_grade)
    return len(rel & set(ranked[:k])) / len(rel)


def mrr_at_k(ranked: Sequence[str], qrel: Mapping[str, int], k: int = 10,
             min_grade: int = 2) -> float:
    rel = _relevant(qrel, min_grade)
    for rank, p in enumerate(ranked[:k], start=1):
        if p in rel:
            return 1.0 / rank
    return 0.0


def paper_hit_at_k(ranked: Sequence[str], qrel: Mapping[str, int], k: int = 10,
                   min_grade: int = 2) -> float:
    papers = {paper_of(p) for p in _relevant(qrel, min_grade)}
    return float(any(paper_of(p) in papers for p in ranked[:k]))


def judged_at_k(ranked: Sequence[str], qrel: Mapping[str, int], k: int) -> float:
    """Fraction of the top k that has a label. Missing ranks count as unjudged."""
    return sum(1 for p in ranked[:k] if p in qrel) / k
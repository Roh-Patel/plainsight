"""Reciprocal rank fusion over ranked lists of point IDs."""


def rrf(rankings, k=60, top_n=100):
    """Fuse ranked lists of point IDs, best first, equal weights.

    Score of an ID is the sum over lists of 1 / (k + rank), rank starting at 1.
    Ties break by lower ID. Returns [(id, score), ...] best first, length <= top_n.
    """
    scores = {}
    for ranking in rankings:
        if len(set(ranking)) != len(ranking):
            raise ValueError("duplicate ID within one ranking")
        for rank, pid in enumerate(ranking, start=1):
            scores[pid] = scores.get(pid, 0.0) + 1.0 / (k + rank)
    ordered = sorted(scores.items(), key=lambda kv: (-kv[1], kv[0]))
    return ordered[:top_n]
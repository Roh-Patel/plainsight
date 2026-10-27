def order_hits(hits):
    """Sort (point_id, score) pairs best first. Ties break by lower point ID."""
    return sorted(hits, key=lambda h: (-h[1], h[0]))

def order_by_score_stable(hits):
    """Sort (id, score) pairs best first. Ties keep input order."""
    return sorted(hits, key=lambda h: -h[1])
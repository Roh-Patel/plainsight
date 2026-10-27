"""Hybrid searcher (system 3): RRF over BM25 and dense top lists."""
from retrieval.fusion import rrf


class HybridSearcher:
    def __init__(self, bm25, dense, depth=100, top_n=100, k=60):
        self.bm25, self.dense = bm25, dense
        self.depth, self.top_n, self.k = depth, top_n, k

    def search(self, queries):
        """Return one [(point_id, rrf_score), ...] list per query, best first."""
        queries = list(queries)
        b = self.bm25.search(queries, k=self.depth)
        d = self.dense.search(queries, k=self.depth)
        return [rrf([[p for p, _ in x], [p for p, _ in y]], k=self.k, top_n=self.top_n)
                for x, y in zip(b, d)]
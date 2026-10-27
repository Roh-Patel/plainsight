"""BM25 searcher (system 1). Tokenization is imported from the indexer."""
import bm25s

from ingest.sparse import BM25_DIR, tok
from retrieval.common import check_manifest
from retrieval.hits import order_hits


class BM25Searcher:
    def __init__(self):
        self.manifest = check_manifest(BM25_DIR / "manifest.json")
        self.retriever = bm25s.BM25.load(str(BM25_DIR))

    def search(self, queries, k=100):
        """Return one [(point_id, score), ...] list per query, best first."""
        toks = tok(list(queries))
        out = [[] for _ in toks]
        live = [i for i, t in enumerate(toks) if t]
        if live:
            idx, sc = self.retriever.retrieve([toks[i] for i in live], k=k, show_progress=False)
            for j, i in enumerate(live):
                hits = [(int(a), float(b)) for a, b in zip(idx[j], sc[j]) if b > 0]
                out[i] = order_hits(hits)
        return out
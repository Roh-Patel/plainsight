"""Dense searcher (systems 2, 3, 5). Backends: 'numpy' (GPU matmul, bulk mining)
and 'qdrant' (exact search, eval and app). Both take the same fp32 query vector."""
import numpy as np
import torch
from qdrant_client import QdrantClient, models

from ingest.dense import load_model
from retrieval.common import INDEX_DIR, IndexMismatch, check_manifest
from retrieval.hits import order_hits

QDRANT_URL = "http://127.0.0.1:6333"


class DenseSearcher:
    def __init__(self, emb_name="dense_base", backend="numpy", model=None, url=QDRANT_URL):
        if backend not in ("numpy", "qdrant"):
            raise ValueError(backend)
        self.backend = backend
        self.manifest = check_manifest(INDEX_DIR / f"{emb_name}.manifest.json")
        self.prefix = self.manifest["query_prefix"]
        self.model = model if model is not None else load_model(self.manifest["model"])
        if backend == "numpy":
            emb = np.load(INDEX_DIR / f"{emb_name}.npy")
            if emb.shape[0] != self.manifest["passages_lines"]:
                raise IndexMismatch(f"{emb_name}.npy has {emb.shape[0]} rows")
            mat = torch.from_numpy(emb).to("cuda", dtype=torch.float32)
            self.mat = mat / mat.norm(dim=1, keepdim=True)
        else:
            self.collection = f"passages_{emb_name}"
            check_manifest(INDEX_DIR / f"qdrant_{self.collection}.manifest.json")
            self.client = QdrantClient(url=url, timeout=60)
            n = self.client.count(self.collection, exact=True).count
            if n != self.manifest["passages_lines"]:
                raise IndexMismatch(f"{self.collection} holds {n} points")

    def encode(self, queries, batch_size=64):
        e = self.model.encode([self.prefix + q for q in queries], batch_size=batch_size,
                              show_progress_bar=False, convert_to_numpy=True,
                              normalize_embeddings=False).astype(np.float32)
        return e / np.linalg.norm(e, axis=1, keepdims=True)

    def search(self, queries, k=100, batch_size=64):
        """Return one [(point_id, score), ...] list per query, best first."""
        queries = list(queries)
        out = []
        for i in range(0, len(queries), batch_size):
            qv = self.encode(queries[i:i + batch_size], batch_size)
            if self.backend == "numpy":
                scores = torch.from_numpy(qv).to("cuda") @ self.mat.T
                top = torch.topk(scores, k, dim=1)
                vals, ids = top.values.cpu().numpy(), top.indices.cpu().numpy()
                for v, ix in zip(vals, ids):
                    out.append(order_hits(zip(ix.tolist(), v.tolist())))
            else:
                for v in qv:
                    res = self.client.query_points(
                        collection_name=self.collection, query=v.tolist(), limit=k,
                        search_params=models.SearchParams(exact=True),
                        with_payload=False, with_vectors=False)
                    out.append(order_hits((int(p.id), float(p.score)) for p in res.points))
        return out
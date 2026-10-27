"""Cross-encoder reranker (systems 4 and 6). Input is `text`, pair truncation
only_second, max_length 512, fp16, raw logits. Ties keep the incoming (fused) order."""
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from retrieval.common import load_passages
from retrieval.hits import order_by_score_stable

RERANK_MODEL = "BAAI/bge-reranker-base"
MAX_LEN = 512
QUERY_MAX_TOKENS = 128


class Reranker:
    def __init__(self, model_name=RERANK_MODEL, texts=None, batch_size=32):
        self.tok = AutoTokenizer.from_pretrained(model_name)
        self.model = (AutoModelForSequenceClassification.from_pretrained(model_name)
                      .half().to("cuda").eval())
        self.texts = texts if texts is not None else [r["text"] for r in load_passages(("text",))]
        self.batch_size = batch_size
        self.stats = {"pairs": 0, "passage_truncated": 0, "query_truncated": 0}

    def _cap_query(self, q):
        ids = self.tok(q, add_special_tokens=False)["input_ids"]
        if len(ids) > QUERY_MAX_TOKENS:
            self.stats["query_truncated"] += 1
            return self.tok.decode(ids[:QUERY_MAX_TOKENS])
        return q

    def score_pairs(self, queries, passages, batch_size=None, count_truncation=True):
        """Raw logits for aligned (query, passage text) lists, in input order."""
        bs = batch_size or self.batch_size
        n = len(passages)
        if count_truncation and n:
            full = self.tok(queries, passages, truncation=False)["input_ids"]
            self.stats["passage_truncated"] += sum(len(x) > MAX_LEN for x in full)
        self.stats["pairs"] += n
        order = sorted(range(n), key=lambda i: len(passages[i]))
        scores = [0.0] * n
        for s in range(0, n, bs):
            idx = order[s:s + bs]
            enc = self.tok([queries[i] for i in idx], [passages[i] for i in idx],
                           truncation="only_second", max_length=MAX_LEN,
                           padding=True, return_tensors="pt").to("cuda")
            with torch.inference_mode():
                logits = self.model(**enc).logits[:, 0].float().cpu().tolist()
            for i, v in zip(idx, logits):
                scores[i] = v
        return scores

    def rerank(self, queries, candidate_lists, depth=50, batch_size=None, count_truncation=True):
        """candidate_lists: one [(point_id, score), ...] per query, best first (fused).
        Returns one [(point_id, logit), ...] per query, best first, length <= depth."""
        qs, ps, spans = [], [], []
        for q, cands in zip(queries, candidate_lists):
            q = self._cap_query(q)
            ids = [p for p, _ in cands[:depth]]
            spans.append(ids)
            qs.extend([q] * len(ids))
            ps.extend(self.texts[p] for p in ids)
        scores = self.score_pairs(qs, ps, batch_size, count_truncation)
        out, pos = [], 0
        for ids in spans:
            chunk = list(zip(ids, scores[pos:pos + len(ids)]))
            pos += len(ids)
            out.append(order_by_score_stable(chunk))
        return out
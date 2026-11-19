import hashlib, json, math, os, sys, collections
import numpy as np
import torch
import bm25s
import Stemmer
from sentence_transformers import SentenceTransformer

from ingest.dense import QUERY_PREFIX
from retrieval.fusion import rrf
from train.dev_eval import load_corpus, load_queries, PASSAGES

MODEL = "models/embedder/emb-v1/checkpoint-810"
OUT = "data/synthetic_val/rr_dev_candidates.jsonl"
MAN = "data/synthetic_val/rr_dev_candidates.manifest.json"
DENSE_MAN = "data/processed/index/dense_ft.manifest.json"
DEV_JSON = "data/synthetic_val/dev/emb-v1-810.json"
DEPTH = 50


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for c in iter(lambda: f.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()


def bm25_search(texts):
    st = Stemmer.Stemmer("english")
    r = bm25s.BM25(method="lucene", k1=1.5, b=0.75)
    r.index(bm25s.tokenize(texts, stopwords="en", stemmer=st, show_progress=False),
            show_progress=False)

    def search(queries, k):
        docs, _ = r.retrieve(bm25s.tokenize(queries, stopwords="en", stemmer=st,
                                            show_progress=False), k=k, show_progress=False)
        return docs
    return search


def recipe_check():
    pids, texts = [], []
    with PASSAGES.open(encoding="utf-8") as f:
        for line in f:
            p = json.loads(line)
            pids.append(p["passage_id"])
            texts.append(p["embed_text"])
    search = bm25_search(texts)
    with open("data/eval/questions.jsonl", encoding="utf-8") as f:
        qs = [json.loads(l) for l in f]
    docs = search([q["question"] for q in qs], 10)
    runs = {}
    with open("runs/bm25.jsonl", encoding="utf-8") as f:
        for l in f:
            r = json.loads(l)
            runs[r["qid"]] = [x["passage_id"] for x in r["results"][:10]]
    exact = ov = 0
    for q, d in zip(qs, docs):
        got = [pids[i] for i in d]
        exact += got == runs[q["qid"]]
        ov += len(set(got) & set(runs[q["qid"]])) / 10
    print("recipe check vs runs/bm25.jsonl: exact top-10 order %d/%d, mean overlap %.4f"
          % (exact, len(qs), ov / len(qs)))


def ndcg(rank):
    return 1.0 / math.log2(rank + 1) if rank and rank <= 10 else 0.0


def main():
    write = "--write" in sys.argv
    if write and (os.path.exists(OUT) or os.path.exists(MAN)):
        sys.exit("outputs exist, refusing to overwrite")
    sha = sha256_file(PASSAGES)
    with open(DENSE_MAN, encoding="utf-8") as f:
        if json.load(f)["passages_sha256"] != sha:
            sys.exit("passages.jsonl does not match the dense_ft manifest")
    recipe_check()

    idx, pids, texts = load_corpus()
    idx = np.array(idx)
    qs = load_queries()
    pid2local = {p: i for i, p in enumerate(pids)}
    src_local = np.array([pid2local[q["source_passage_id"]] for q in qs])
    print("val passages", len(pids), "| kept val queries", len(qs))

    E = np.load("data/processed/index/dense_ft.npy", mmap_mode="r")
    M = torch.tensor(np.asarray(E[idx]), device="cuda").float()
    M = M / M.norm(dim=1, keepdim=True)
    model = SentenceTransformer(MODEL, device="cuda")
    model.half()
    model.max_seq_length = 512
    Q = model.encode([QUERY_PREFIX + q["query"] for q in qs], batch_size=64,
                     convert_to_tensor=True).float()
    Q = Q / Q.norm(dim=1, keepdim=True)
    del model

    dense_top, dense_src_rank = [], []
    for s in range(0, len(qs), 512):
        S = Q[s:s + 512] @ M.T
        sr = torch.from_numpy(src_local[s:s + 512]).cuda()
        ss = S.gather(1, sr[:, None])
        dense_src_rank.extend((1 + (S > ss).sum(1)).cpu().tolist())
        dense_top.append(np.argsort(-S.cpu().numpy(), axis=1, kind="stable")[:, :100])
    dense_top = np.concatenate(dense_top)

    bm_docs = bm25_search(texts)([q["query"] for q in qs], 100)

    rows = []
    for n, q in enumerate(qs):
        b = [int(idx[i]) for i in bm_docs[n]]
        d = [int(idx[i]) for i in dense_top[n]]
        fused = [p for p, _ in rrf([b, d], k=60, top_n=100)][:DEPTH]
        src = int(idx[src_local[n]])
        rank = fused.index(src) + 1 if src in fused else None
        rows.append({"qid": q["qid"], "voice": q["voice"], "query": q["query"],
                     "src_point_id": src, "src_rank": rank, "cands": fused})

    voices = ["layperson", "confused_undergrad", "half_remembered_term", "ALL"]
    print("voice | n | source in fused top 50 | fused nDCG@10 | fused src@1")
    for v in voices:
        sub = [r for r in rows if v == "ALL" or r["voice"] == v]
        print(v, "|", len(sub), "|", round(sum(r["src_rank"] is not None for r in sub) / len(sub), 4),
              "|", round(sum(ndcg(r["src_rank"]) for r in sub) / len(sub), 4), "|",
              round(sum(r["src_rank"] == 1 for r in sub) / len(sub), 4))

    with open(DEV_JSON, encoding="utf-8") as f:
        dev = json.load(f)
    stored = dict(zip(dev["qids"], dev["ranks"]))
    mine = {q["qid"]: r for q, r in zip(qs, dense_src_rank)}
    assert set(stored) == set(mine)
    same = sum(stored[k] == mine[k] for k in stored)
    print("dense-only nDCG@10 here %.4f vs dev json %.4f | exact source-rank match %d/%d"
          % (np.mean([ndcg(r) for r in dense_src_rank]),
             dev["metrics"]["all"]["ndcg@10"], same, len(stored)))

    if not write:
        print("dry run, nothing written (use --write)")
        return
    with open(OUT, "w", encoding="utf-8", newline="\n") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    qh = hashlib.sha256("\n".join(r["qid"] for r in rows).encode("utf-8")).hexdigest()
    with open(MAN, "w", encoding="utf-8", newline="\n") as f:
        json.dump({"n": len(rows), "depth": DEPTH, "rrf_k": 60, "bm25": "val-only index, D020 settings",
                   "dense_rows": "dense_ft.npy, val rows", "embedder": MODEL,
                   "passages_sha256": sha, "qids_sha256": qh,
                   "created": __import__("datetime").datetime.now().isoformat(timespec="seconds")},
                  f, indent=1)
    print("wrote", OUT, "and", MAN)


if __name__ == "__main__":
    main()
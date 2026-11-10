"""Rank the source passage of every synthetic query under bm25,
dense_base and hybrid_base. Writes top-100 point IDs and a rank table."""
import argparse
import hashlib
import json
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np

from retrieval.bm25 import BM25Searcher
from retrieval.dense import DenseSearcher
from retrieval.fusion import rrf

PASSAGES = Path("data/processed/passages.jsonl")
QUERIES = Path("data/synthetic/queries.jsonl")
QUESTIONS = Path("data/eval/questions.jsonl")
OUT_DIR = Path("data/synthetic/ranks")
SYSTEMS = ("bm25", "dense_base", "hybrid_base")
K = 100
RRF_K = 60


def load_passages():
    h = hashlib.sha256()
    ids = []
    with open(PASSAGES, "rb") as f:
        for line in f:
            h.update(line)
            ids.append(json.loads(line)["passage_id"])
    return ids, h.hexdigest()


def section_codes(ids):
    uniq = {}
    return np.array([uniq.setdefault(p.rsplit("::", 1)[0], len(uniq)) for p in ids],
                    dtype=np.int32)


def to_array(lists):
    a = np.full((len(lists), K), -1, dtype=np.int32)
    for i, hits in enumerate(lists):
        ids = [p for p, _ in hits][:K]
        a[i, :len(ids)] = ids
    return a


def search_all(queries, bm25, dense, chunk, batch_size):
    parts = {s: [] for s in SYSTEMS}
    t0 = time.time()
    for i in range(0, len(queries), chunk):
        qs = queries[i:i + chunk]
        b = bm25.search(qs, k=K)
        d = dense.search(qs, k=K, batch_size=batch_size)
        h = [rrf([[p for p, _ in x], [p for p, _ in y]], k=RRF_K, top_n=K)
             for x, y in zip(b, d)]
        parts["bm25"].append(to_array(b))
        parts["dense_base"].append(to_array(d))
        parts["hybrid_base"].append(to_array(h))
        print(f"{min(i + chunk, len(queries))}/{len(queries)}  {time.time() - t0:.0f}s", flush=True)
    return {s: np.concatenate(v) for s, v in parts.items()}


def rank_columns(top, src, sec_of):
    n = len(top)
    src_rank = np.full(n, -1, dtype=np.int32)
    sec_rank = np.full(n, -1, dtype=np.int32)
    hit = top == src[:, None]
    has = hit.any(axis=1)
    src_rank[has] = hit[has].argmax(axis=1) + 1
    valid = top >= 0
    same = (sec_of[np.where(valid, top, 0)] == sec_of[src][:, None]) & valid
    has = same.any(axis=1)
    sec_rank[has] = same[has].argmax(axis=1) + 1
    return src_rank, sec_rank


def load_models():
    return BM25Searcher(), DenseSearcher("dense_base", backend="numpy")


def run(args):
    out_dir = Path(args.out_dir)
    if args.limit and out_dir == OUT_DIR:
        raise SystemExit("--limit runs must use a different --out-dir")
    if out_dir.exists() and any(out_dir.iterdir()) and not args.force:
        raise SystemExit(f"{out_dir} is not empty; use --force to overwrite")
    out_dir.mkdir(parents=True, exist_ok=True)

    ids, sha = load_passages()
    point_of = {p: i for i, p in enumerate(ids)}
    sec_of = section_codes(ids)

    with open(QUERIES, encoding="utf-8") as f:
        rows = [json.loads(l) for l in f]
    if args.limit:
        rows = rows[:args.limit]
    missing = [r["qid"] for r in rows if r["source_passage_id"] not in point_of]
    if missing:
        raise SystemExit(f"{len(missing)} source passages not in passages.jsonl, e.g. {missing[:3]}")
    src = np.array([point_of[r["source_passage_id"]] for r in rows], dtype=np.int32)

    bm25, dense = load_models()
    t0 = time.time()
    tops = search_all([r["query"] for r in rows], bm25, dense, args.chunk, args.batch_size)

    cols = {}
    for s in SYSTEMS:
        np.save(out_dir / f"{s}_top100.npy", tops[s])
        cols[s] = rank_columns(tops[s], src, sec_of)

    def val(x):
        return None if x < 0 else int(x)

    with open(out_dir / "ranks.jsonl", "w", encoding="utf-8", newline="\n") as f:
        for i, r in enumerate(rows):
            o = {"qid": r["qid"], "voice": r["voice"], "src_point_id": int(src[i])}
            for s in SYSTEMS:
                o[f"{s}_src_rank"] = val(cols[s][0][i])
                o[f"{s}_sec_rank"] = val(cols[s][1][i])
            f.write(json.dumps(o) + "\n")

    manifest = {
        "n": len(rows), "k": K, "rrf_k": RRF_K, "systems": list(SYSTEMS),
        "dense_backend": "numpy", "dense_batch_size": args.batch_size, "chunk": args.chunk,
        "limit": args.limit, "rank_base": 1,
        "qids_sha256": hashlib.sha256("\n".join(r["qid"] for r in rows).encode()).hexdigest(),
        "passages_sha256": sha, "passages_lines": len(ids),
        "seconds": round(time.time() - t0, 1), "created": datetime.now().isoformat(timespec="seconds"),
    }
    (out_dir / "ranks.manifest.json").write_text(json.dumps(manifest, indent=1), encoding="utf-8")
    print(json.dumps(manifest, indent=1))


def summary(args):
    path = Path(args.out_dir) / "ranks.jsonl"
    by_voice = defaultdict(list)
    with open(path, encoding="utf-8") as f:
        for l in f:
            r = json.loads(l)
            by_voice[r["voice"]].append(r)
            by_voice["ALL"].append(r)
    print("voice | n | system | src@10 | src@50 | src@100 | absent | absent but same-section in top 50")
    for v in ("layperson", "confused_undergrad", "half_remembered_term", "ALL"):
        rs = by_voice[v]
        n = len(rs)
        for s in SYSTEMS:
            a = [r[f"{s}_src_rank"] for r in rs]
            sec = [r[f"{s}_sec_rank"] for r in rs]
            in10 = sum(x is not None and x <= 10 for x in a) / n
            in50 = sum(x is not None and x <= 50 for x in a) / n
            in100 = sum(x is not None for x in a) / n
            out50 = [(x, y) for x, y in zip(a, sec) if x is None or x > 50]
            sec50 = (sum(y is not None and y <= 50 for _, y in out50) / len(out50)) if out50 else 0.0
            print(f"{v} | {n} | {s} | {in10:.3f} | {in50:.3f} | {in100:.3f} | {1 - in100:.3f} | {sec50:.3f}")


def check(args):
    ids, _ = load_passages()
    point_of = {p: i for i, p in enumerate(ids)}
    with open(QUESTIONS, encoding="utf-8") as f:
        qs = [json.loads(l) for l in f]
    bm25, dense = load_models()
    n = len(qs)
    res = search_all([q["question"] for q in qs], bm25, dense, chunk=n, batch_size=n)
    for s in SYSTEMS:
        ref = {}
        with open(f"runs/{s}.jsonl", encoding="utf-8") as f:
            for l in f:
                r = json.loads(l)
                ref[r["qid"]] = [point_of[x["passage_id"]] for x in r["results"]]
        same10, overlap = 0, []
        for q, row in zip(qs, res[s]):
            mine = [int(x) for x in row if x >= 0]
            theirs = ref[q["qid"]]
            same10 += mine[:10] == theirs[:10]
            overlap.append(len(set(mine) & set(theirs)) / max(len(theirs), 1))
        print(f"{s}: identical top-10 order {same10}/{n}, mean top-100 overlap {np.mean(overlap):.4f}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default=str(OUT_DIR))
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--chunk", type=int, default=4096)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--summary", action="store_true")
    a = ap.parse_args()
    if a.check:
        check(a)
    elif a.summary:
        summary(a)
    else:
        run(a)
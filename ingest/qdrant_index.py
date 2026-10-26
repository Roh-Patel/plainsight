import argparse, json, time

import numpy as np
from qdrant_client import QdrantClient, models

from ingest.dense import PASSAGES, load_passages, paths, sha256

URL = "http://localhost:6333"


def client():
    return QdrantClient(url=URL, timeout=120)


def build(args):
    npy, man = paths(args.emb_name)
    m = json.loads(man.read_text(encoding="utf-8"))
    rows = load_passages()
    assert not m["limit"], "refusing to index a --limit embedding file"
    assert m["passages_lines"] == len(rows) and m["passages_sha256"] == sha256(PASSAGES), \
        "passages.jsonl changed since embedding"
    emb = np.load(npy)
    assert emb.shape[0] == len(rows), emb.shape

    c = client()
    if c.collection_exists(args.collection):
        if not args.force:
            raise SystemExit(f"{args.collection} exists; use --force to recreate")
        c.delete_collection(args.collection)
    c.create_collection(
        args.collection,
        vectors_config=models.VectorParams(size=int(emb.shape[1]),
                                           distance=models.Distance.COSINE))
    t0 = time.time()
    for s in range(0, len(rows), args.batch):
        e = min(s + args.batch, len(rows))
        c.upsert(args.collection, wait=True, points=models.Batch(
            ids=list(range(s, e)),
            vectors=emb[s:e].astype(np.float32).tolist(),
            payloads=[{"passage_id": r["passage_id"], "paper_id": r["paper_id"]}
                      for r in rows[s:e]]))
        if (s // args.batch) % 20 == 0:
            print(f"  {e}/{len(rows)}")
    while c.get_collection(args.collection).status != models.CollectionStatus.GREEN:
        time.sleep(2)
    manifest = {"collection": args.collection, "emb_name": args.emb_name,
                "n_points": len(rows), "distance": "cosine", "id_scheme": "line index in passages.jsonl",
                "payload": ["passage_id", "paper_id"],
                "passages_lines": len(rows), "passages_sha256": m["passages_sha256"],
                "seconds": round(time.time() - t0, 1)}
    (npy.parent / f"qdrant_{args.collection}.manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"upserted {len(rows)} points in {manifest['seconds']}s")


def check(args):
    npy, man = paths(args.emb_name)
    mq = json.loads((npy.parent / f"qdrant_{args.collection}.manifest.json").read_text(encoding="utf-8"))
    rows = load_passages()
    assert mq["passages_sha256"] == sha256(PASSAGES), "passages.jsonl changed since indexing"
    c = client()
    n = c.count(args.collection, exact=True).count
    print("points:", n, "expected:", len(rows))
    assert n == len(rows)
    emb = np.load(npy).astype(np.float32)
    rng = np.random.default_rng(0)

    # payload alignment: point id must map to the same passage_id as line i
    bad = 0
    sample = rng.choice(len(rows), 300, replace=False).tolist()
    for p in c.retrieve(args.collection, ids=sample, with_payload=True):
        bad += p.payload["passage_id"] != rows[p.id]["passage_id"]
    print("payload misaligned in 300 sampled points:", bad)

    # exact Qdrant search vs numpy matmul, using stored vectors as queries
    overlaps, self_top1 = [], 0
    for i in rng.choice(len(rows), 50, replace=False).tolist():
        res = c.query_points(args.collection, query=emb[i].tolist(), limit=10,
                             search_params=models.SearchParams(exact=True)).points
        ref = np.argsort(-(emb @ emb[i]))[:10].tolist()
        overlaps.append(len({p.id for p in res} & set(ref)) / 10)
        self_top1 += res[0].id == i
    print(f"top-10 overlap with numpy (50 queries): mean={np.mean(overlaps):.3f} min={min(overlaps):.1f}")
    print(f"self is top-1: {self_top1}/50")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--collection", default="passages_dense_base")
    ap.add_argument("--emb-name", default="dense_base")
    ap.add_argument("--batch", type=int, default=1000)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--check", action="store_true")
    a = ap.parse_args()
    check(a) if a.check else build(a)
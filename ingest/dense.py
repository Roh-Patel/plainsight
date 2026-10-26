import argparse, hashlib, json, time
from pathlib import Path

import numpy as np
import sentence_transformers
import torch
from sentence_transformers import SentenceTransformer

PASSAGES = Path("data/processed/passages.jsonl")
INDEX_DIR = Path("data/processed/index")
BASE_MODEL = "BAAI/bge-base-en-v1.5"
QUERY_PREFIX = "Represent this sentence for searching relevant passages: "


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def load_passages():
    with open(PASSAGES, encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def load_model(name):
    model = SentenceTransformer(name, device="cuda")
    model.half()
    model.max_seq_length = 512
    return model


def encode(model, texts, batch_size=128):
    e = model.encode(texts, batch_size=batch_size, show_progress_bar=True,
                     convert_to_numpy=True, normalize_embeddings=False)
    e = e.astype(np.float32)
    e /= np.linalg.norm(e, axis=1, keepdims=True)
    return e.astype(np.float16)


def paths(name):
    return INDEX_DIR / f"{name}.npy", INDEX_DIR / f"{name}.manifest.json"


def build(args):
    name = args.out_name + (f"_limit{args.limit}" if args.limit else "")
    npy, man = paths(name)
    if npy.exists() and not args.force:
        raise SystemExit(f"{npy} exists; use --force to overwrite")
    INDEX_DIR.mkdir(parents=True, exist_ok=True)
    rows = load_passages()
    texts = [r["embed_text"] for r in rows]
    if args.limit:
        texts = texts[: args.limit]
    model = load_model(args.model)
    t0 = time.time()
    emb = encode(model, texts, args.batch_size)
    dt = time.time() - t0
    np.save(npy, emb)
    manifest = {
        "model": args.model, "dtype": "float16", "normalized": True,
        "n": int(emb.shape[0]), "dim": int(emb.shape[1]),
        "limit": args.limit, "text_field": "embed_text",
        "passages_lines": len(rows), "passages_sha256": sha256(PASSAGES),
        "query_prefix": QUERY_PREFIX, "max_seq_length": 512,
        "sentence_transformers": sentence_transformers.__version__,
        "torch": torch.__version__,
        "seconds": round(dt, 1),
        "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    man.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    f32 = emb.astype(np.float32)
    norms = np.linalg.norm(f32, axis=1)
    print(f"shape={emb.shape} dtype={emb.dtype} nan={bool(np.isnan(f32).any())}")
    print(f"norm min/max={norms.min():.4f}/{norms.max():.4f}")
    print(f"{len(texts)/dt:.0f} passages/s, {dt:.0f}s total")


def ascii_(s):
    return s.encode("ascii", "replace").decode()


def check(args):
    npy, man = paths(args.out_name)
    m = json.loads(man.read_text(encoding="utf-8"))
    rows = load_passages()
    assert m["passages_lines"] == len(rows), "passages.jsonl line count changed"
    assert m["passages_sha256"] == sha256(PASSAGES), "passages.jsonl content changed"
    emb = np.load(npy).astype(np.float32)
    assert emb.shape == (len(rows), m["dim"]), emb.shape
    print("manifest matches passages.jsonl:", emb.shape)
    model = load_model(m["model"])
    queries = [
        "how do astronomers find planets by watching a star get dimmer",
        "why are some gas giants orbiting so close to their star",
        "what is the atmosphere of a hot jupiter made of",
    ]
    for q in queries:
        qv = model.encode([m["query_prefix"] + q], convert_to_numpy=True,
                          normalize_embeddings=False).astype(np.float32)[0]
        qv /= np.linalg.norm(qv)
        scores = emb @ qv
        print("\nQ:", q)
        for i in np.argsort(-scores)[:3]:
            r = rows[i]
            print(f"  {scores[i]:.3f} {r['passage_id']} | {ascii_(r['section_title'])[:50]}")
            print("     ", ascii_(r["text"][:140]).replace("\n", " "))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=BASE_MODEL)
    ap.add_argument("--out-name", default="dense_base")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--check", action="store_true")
    a = ap.parse_args()
    check(a) if a.check else build(a)
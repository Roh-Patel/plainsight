import argparse, json, time
from pathlib import Path

import bm25s
import Stemmer

from ingest.dense import INDEX_DIR, PASSAGES, load_passages, sha256

BM25_DIR = INDEX_DIR / "bm25"
STEMMER = Stemmer.Stemmer("english")


def tok(texts):
    return bm25s.tokenize(texts, stopwords="en", stemmer=STEMMER,
                          return_ids=False, show_progress=False)


def build(args):
    if BM25_DIR.exists() and not args.force:
        raise SystemExit(f"{BM25_DIR} exists; use --force to overwrite")
    rows = load_passages()
    t0 = time.time()
    retriever = bm25s.BM25()  # defaults: lucene, k1=1.5, b=0.75
    retriever.index(bm25s.tokenize([r["embed_text"] for r in rows], stopwords="en",
                                   stemmer=STEMMER, show_progress=True))
    retriever.save(str(BM25_DIR))
    manifest = {"text_field": "embed_text", "tokenizer": "bm25s default regex",
                "stopwords": "en", "stemmer": "PyStemmer english",
                "method": "lucene", "k1": 1.5, "b": 0.75,
                "bm25s": bm25s.__version__, "passages_lines": len(rows),
                "passages_sha256": sha256(PASSAGES),
                "seconds": round(time.time() - t0, 1)}
    (BM25_DIR / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"indexed {len(rows)} passages in {manifest['seconds']}s")


def check(args):
    m = json.loads((BM25_DIR / "manifest.json").read_text(encoding="utf-8"))
    rows = load_passages()
    assert m["passages_lines"] == len(rows) and m["passages_sha256"] == sha256(PASSAGES), \
        "passages.jsonl changed since indexing"
    print("manifest matches passages.jsonl")
    retriever = bm25s.BM25.load(str(BM25_DIR))
    for q in ["how do astronomers find planets by watching a star get dimmer",
              "why are some gas giants orbiting so close to their star",
              "what is the atmosphere of a hot jupiter made of"]:
        idx, sc = retriever.retrieve(tok([q]), k=3, show_progress=False)
        print("\nQ:", q)
        for i, s in zip(idx[0], sc[0]):
            r = rows[int(i)]
            print(f"  {s:.2f} {r['passage_id']} | "
                  f"{r['section_title'].encode('ascii','replace').decode()[:50]}")
            print("     ", r["text"][:140].encode("ascii", "replace").decode().replace("\n", " "))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--check", action="store_true")
    a = ap.parse_args()
    check(a) if a.check else build(a)
"""T19 judgment pool and owner random sample (D036, D066).

Pool: union of the top 10 of the seven run files per question, deduplicated.
Sample: 250 pairs drawn uniformly with random.Random(29) from the pool sorted
by (qid, passage_id), before any grading call.

  python -m eval.build_pool          dry run: print counts, write nothing
  python -m eval.build_pool --write  write data/eval/pool.jsonl and manifest (refuses to overwrite)
  python -m eval.build_pool --check  rebuild in memory and compare with the written files
"""
import argparse
import hashlib
import json
import random
import sys
from datetime import datetime
from pathlib import Path

from eval.loaders import load_passage_ids, load_run

SYSTEMS = ["bm25", "dense_base", "dense_ft", "hybrid_base", "hybrid_base_rerank_base",
           "hybrid_ft_embed", "hybrid_ft_embed_rerank_ft"]
DEPTH = 10
SEED = 29
N_SAMPLE = 250
PASSAGES = Path("data/processed/passages.jsonl")
QUESTIONS = Path("data/eval/questions.jsonl")
OUT = Path("data/eval/pool.jsonl")
MANIFEST = Path("data/eval/pool.manifest.json")


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def build():
    valid = load_passage_ids(PASSAGES)
    qids = [json.loads(l)["qid"] for l in QUESTIONS.read_text(encoding="utf-8").splitlines() if l.strip()]
    runs = {s: load_run(f"runs/{s}.jsonl", valid, system=s) for s in SYSTEMS}
    for s, run in runs.items():
        if set(run) != set(qids):
            sys.exit(f"{s}: run qids do not match questions.jsonl")
        short = [q for q in qids if len(run[q]) < DEPTH]
        if short:
            sys.exit(f"{s}: {len(short)} questions with fewer than {DEPTH} results")
    pairs = sorted({(q, p) for s in SYSTEMS for q in qids for p in runs[s][q][:DEPTH]})
    sample = set(random.Random(SEED).sample(range(len(pairs)), N_SAMPLE))
    rows = [{"qid": q, "passage_id": p, "in_sample": i in sample} for i, (q, p) in enumerate(pairs)]
    return rows, qids


def summary(rows, qids):
    per_q = {q: 0 for q in qids}
    for r in rows:
        per_q[r["qid"]] += 1
    n = list(per_q.values())
    s_q = len({r["qid"] for r in rows if r["in_sample"]})
    print(f"pairs {len(rows)}, questions {len(per_q)}, per question mean {sum(n) / len(n):.1f} "
          f"min {min(n)} max {max(n)}")
    print(f"sample {sum(r['in_sample'] for r in rows)} pairs from {s_q} questions")


def main():
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group()
    g.add_argument("--write", action="store_true")
    g.add_argument("--check", action="store_true")
    args = ap.parse_args()
    rows, qids = build()
    summary(rows, qids)
    if args.check:
        on_disk = [json.loads(l) for l in OUT.read_text(encoding="utf-8").splitlines() if l.strip()]
        man = json.loads(MANIFEST.read_text(encoding="utf-8"))
        ok = on_disk == rows and man["runs_sha256"] == {s: sha256_file(f"runs/{s}.jsonl") for s in SYSTEMS}
        print("check", "OK" if ok else "MISMATCH")
        sys.exit(0 if ok else 1)
    if not args.write:
        print("dry run, nothing written (use --write)")
        return
    if OUT.exists() or MANIFEST.exists():
        sys.exit(f"{OUT} or its manifest exists; refusing to overwrite")
    with OUT.open("w", encoding="utf-8", newline="\n") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    man = {"systems": SYSTEMS, "depth": DEPTH, "n_pairs": len(rows), "n_questions": len(qids),
           "seed": SEED, "n_sample": N_SAMPLE,
           "sample_rule": "random.Random(seed).sample(range(n_pairs), n_sample) over pairs sorted by (qid, passage_id)",
           "runs_sha256": {s: sha256_file(f"runs/{s}.jsonl") for s in SYSTEMS},
           "questions_sha256": sha256_file(QUESTIONS), "passages_sha256": sha256_file(PASSAGES),
           "created": datetime.now().isoformat(timespec="seconds")}
    MANIFEST.write_text(json.dumps(man, indent=1) + "\n", encoding="utf-8", newline="\n")
    print(f"wrote {OUT} and {MANIFEST}")


if __name__ == "__main__":
    main()

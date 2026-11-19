"""Draw the passage sample for synthetic query generation and assign shards."""
import argparse
import hashlib
import json
import os
import random
from collections import Counter, defaultdict
from datetime import datetime

from train.eligibility import MIN_FW, MIN_TOKENS, PASSAGES, SPLITS, fw_fraction

DENSE_MANIFEST = "data/processed/index/dense_base.manifest.json"
OUT = "data/synthetic/gen_sample.jsonl"
OUT_MANIFEST = "data/synthetic/gen_sample.manifest.json"
CAP = 15
SEED = 14
PILOT = 500
SHARD = 5000


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def load_excluded(paths):
    ids = set()
    for path in paths:
        with open(path, encoding="utf-8") as f:
            for line in f:
                ids.add(json.loads(line)["passage_id"])
    return ids


def build(excluded):
    with open(SPLITS, encoding="utf-8") as f:
        train = set(json.load(f)["train"])
    by_paper = defaultdict(list)
    n_lines = 0
    with open(PASSAGES, encoding="utf-8") as f:
        for i, line in enumerate(f):
            n_lines += 1
            p = json.loads(line)
            if p["paper_id"] not in train or p["passage_id"] in excluded:
                continue
            if p["n_tokens"] >= MIN_TOKENS and fw_fraction(p["text"]) >= MIN_FW:
                by_paper[p["paper_id"]].append((i, p["passage_id"]))
    rng = random.Random(SEED)
    picked = []
    for paper_id in sorted(by_paper):
        items = by_paper[paper_id]
        rng.shuffle(items)
        for i, pid in items[:CAP]:
            picked.append((i, pid, paper_id))
    rng.shuffle(picked)
    rows = []
    for order, (i, pid, paper_id) in enumerate(picked):
        shard = 0 if order < PILOT else 1 + (order - PILOT) // SHARD
        rows.append({"order": order, "point_id": i, "passage_id": pid,
                     "paper_id": paper_id, "shard": shard})
    return rows, n_lines


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exclude", nargs="+", required=True)
    ap.add_argument("--verify", action="store_true")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    excluded = load_excluded(args.exclude)
    sha = sha256_file(PASSAGES)
    with open(DENSE_MANIFEST, encoding="utf-8") as f:
        if json.load(f)["passages_sha256"] != sha:
            raise SystemExit("passages.jsonl does not match the index manifest")
    rows, n_lines = build(excluded)

    if args.verify:
        with open(OUT, encoding="utf-8") as f:
            saved = [json.loads(line) for line in f]
        print("reproducible from seed:", saved == rows)
        return

    if os.path.exists(OUT) and not args.force:
        raise SystemExit(f"{OUT} exists; refusing to overwrite")
    os.makedirs("data/synthetic", exist_ok=True)
    with open(OUT, "w", encoding="utf-8", newline="\n") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")

    shard_sizes = dict(sorted(Counter(r["shard"] for r in rows).items()))
    per_paper = Counter(r["paper_id"] for r in rows)
    manifest = {
        "seed": SEED, "cap_per_paper": CAP, "min_tokens": MIN_TOKENS,
        "min_function_word_fraction": MIN_FW, "pilot_size": PILOT,
        "shard_size": SHARD, "excluded_files": args.exclude,
        "n_excluded": len(excluded), "n": len(rows), "n_papers": len(per_paper),
        "shard_sizes": shard_sizes, "passages_lines": n_lines,
        "passages_sha256": sha, "created": datetime.now().isoformat(timespec="seconds"),
    }
    with open(OUT_MANIFEST, "w", encoding="utf-8", newline="\n") as f:
        json.dump(manifest, f, indent=2)
        f.write("\n")

    print(f"rows {len(rows)}  papers {len(per_paper)}  max per paper {max(per_paper.values())}")
    print(f"shard sizes {shard_sizes}")
    print(f"excluded ids {len(excluded)}  overlap with sample "
          f"{sum(1 for r in rows if r['passage_id'] in excluded)}")


if __name__ == "__main__":
    main()
"""Draw the val-split passage sample for the dev set (queries generated later)."""
import argparse
import json
import os
import random
from collections import Counter, defaultdict
from datetime import datetime

from train.draw_gen_sample import DENSE_MANIFEST, sha256_file
from train.eligibility import MIN_FW, MIN_TOKENS, PASSAGES, SPLITS, fw_fraction

OUT_DIR = "data/synthetic_val"
OUT = f"{OUT_DIR}/gen_sample.jsonl"
OUT_MANIFEST = f"{OUT_DIR}/gen_sample.manifest.json"
CAP = 10
SEED = 18


def build():
    with open(SPLITS, encoding="utf-8") as f:
        val = set(json.load(f)["val"])
    by_paper = defaultdict(list)
    n_lines = 0
    with open(PASSAGES, encoding="utf-8") as f:
        for i, line in enumerate(f):
            n_lines += 1
            p = json.loads(line)
            if p["paper_id"] not in val:
                continue
            if p["n_tokens"] >= MIN_TOKENS and fw_fraction(p["text"]) >= MIN_FW:
                by_paper[p["paper_id"]].append((i, p["passage_id"]))
    n_eligible = sum(len(v) for v in by_paper.values())
    rng = random.Random(SEED)
    picked = []
    for paper_id in sorted(by_paper):
        items = by_paper[paper_id]
        rng.shuffle(items)
        for i, pid in items[:CAP]:
            picked.append((i, pid, paper_id))
    rng.shuffle(picked)
    rows = [{"order": order, "point_id": i, "passage_id": pid,
             "paper_id": paper_id, "shard": 0}
            for order, (i, pid, paper_id) in enumerate(picked)]
    return rows, n_lines, val, n_eligible


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--verify", action="store_true")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    sha = sha256_file(PASSAGES)
    with open(DENSE_MANIFEST, encoding="utf-8") as f:
        if json.load(f)["passages_sha256"] != sha:
            raise SystemExit("passages.jsonl does not match the index manifest")
    rows, n_lines, val, n_eligible = build()
    assert all(r["paper_id"] in val for r in rows), "non-val paper in sample"

    if args.verify:
        with open(OUT, encoding="utf-8") as f:
            saved = [json.loads(line) for line in f]
        print("reproducible from seed:", saved == rows)
        return

    if os.path.exists(OUT) and not args.force:
        raise SystemExit(f"{OUT} exists; refusing to overwrite")
    os.makedirs(OUT_DIR, exist_ok=True)
    with open(OUT, "w", encoding="utf-8", newline="\n") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")

    per_paper = Counter(r["paper_id"] for r in rows)
    manifest = {
        "split": "val", "seed": SEED, "cap_per_paper": CAP, "min_tokens": MIN_TOKENS,
        "min_function_word_fraction": MIN_FW, "n_val_papers": len(val),
        "n_eligible": n_eligible, "n": len(rows), "n_papers": len(per_paper),
        "shard_sizes": {"0": len(rows)}, "passages_lines": n_lines,
        "passages_sha256": sha, "created": datetime.now().isoformat(timespec="seconds"),
    }
    with open(OUT_MANIFEST, "w", encoding="utf-8", newline="\n") as f:
        json.dump(manifest, f, indent=2)
        f.write("\n")

    print(f"val papers {len(val)}  eligible passages {n_eligible}")
    print(f"rows {len(rows)}  papers {len(per_paper)}  max per paper {max(per_paper.values())}")


if __name__ == "__main__":
    main()
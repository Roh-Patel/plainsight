"""Draw a seeded sample of train-split passages for prompt testing."""
import argparse
import json
import random
import sys
from pathlib import Path

from ingest.chunk import STOP

PASSAGES = Path("data/processed/passages.jsonl")
SPLITS = Path("data/processed/splits.json")
STOP_SET = {w.lower() for w in STOP}
PUNCT = ".,;:()[]{}\"'!?"


def fw_fraction(text):
    words = text.split()
    if not words:
        return 0.0
    hits = sum(1 for w in words if w.lower().strip(PUNCT) in STOP_SET)
    return hits / len(words)


def main():
    sys.stdout.reconfigure(errors="replace")
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=50)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--min-tokens", type=int, default=100)
    ap.add_argument("--min-fw", type=float, default=0.15)
    ap.add_argument("--out", required=True)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    out = Path(args.out)
    if out.exists() and not args.force:
        sys.exit(f"{out} exists; refusing to overwrite without --force")

    splits = json.loads(SPLITS.read_text(encoding="utf-8"))
    train = set(splits["train"])

    n_train = n_short = n_lowfw = 0
    by_paper = {}
    with PASSAGES.open(encoding="utf-8") as f:
        for line in f:
            p = json.loads(line)
            if p["paper_id"] not in train:
                continue
            n_train += 1
            short = p["n_tokens"] < args.min_tokens
            fw = fw_fraction(p["text"])
            lowfw = fw < args.min_fw
            n_short += short
            n_lowfw += lowfw
            if short or lowfw:
                continue
            by_paper.setdefault(p["paper_id"], []).append(
                {"passage_id": p["passage_id"], "paper_id": p["paper_id"],
                 "section_title": p["section_title"],
                 "n_tokens": p["n_tokens"], "fw": round(fw, 3)})

    n_elig = sum(len(v) for v in by_paper.values())
    print(f"train passages: {n_train}")
    print(f"  n_tokens < {args.min_tokens}: {n_short}")
    print(f"  function-word fraction < {args.min_fw}: {n_lowfw}")
    print(f"eligible: {n_elig} in {len(by_paper)} papers")

    rng = random.Random(args.seed)
    papers = rng.sample(sorted(by_paper), args.n)
    rows = []
    for pid in papers:
        cands = sorted(by_paper[pid], key=lambda r: r["passage_id"])
        rows.append(rng.choice(cands))
    assert all(r["paper_id"] in train for r in rows)
    assert len({r["paper_id"] for r in rows}) == len(rows)

    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8", newline="\n") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    toks = sorted(r["n_tokens"] for r in rows)
    print(f"wrote {len(rows)} rows to {out}; n_tokens median {toks[len(toks)//2]}")
    for r in rows:
        print(f"{r['passage_id']}  {r['n_tokens']:>3}  {r['fw']:.2f}  {r['section_title'][:60]}")


if __name__ == "__main__":
    main()
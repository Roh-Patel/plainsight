"""Build data/synthetic/train_triples.jsonl for T16 (IDs only, no text).
Dry run by default; --write creates the file and never overwrites."""
import argparse
import json
import random
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path

from train.build_hard_negatives import (OUT as HARD, OUT_MANIFEST as HARD_MANIFEST, PASSAGES,
                                        QUERIES, SPLITS, VOICES, read_jsonl, sha256_file)

OUT = Path("data/synthetic/train_triples.jsonl")
OUT_MANIFEST = Path("data/synthetic/train_triples.manifest.json")
SEED = 24
K = 1


def main():
    sys.stdout.reconfigure(errors="replace")
    ap = argparse.ArgumentParser()
    ap.add_argument("--write", action="store_true")
    args = ap.parse_args()
    if args.write and (OUT.exists() or OUT_MANIFEST.exists()):
        sys.exit(f"{OUT} or its manifest exists; refusing to overwrite")

    hn_man = json.loads(HARD_MANIFEST.read_text(encoding="utf-8"))
    sha = sha256_file(PASSAGES)
    if hn_man["passages_sha256"] != sha:
        sys.exit("passages.jsonl does not match the hard-negatives manifest")
    train = set(json.loads(SPLITS.read_text(encoding="utf-8"))["train"])
    ids = {r["passage_id"] for r in read_jsonl(PASSAGES)}
    kept = [q for q in read_jsonl(QUERIES) if q["kept"]]
    hn = list(read_jsonl(HARD))
    if len(kept) != len(hn) or any(q["qid"] != h["qid"] for q, h in zip(kept, hn)):
        sys.exit("queries.jsonl (kept) and hard_negatives.jsonl are not aligned")

    rng = random.Random(SEED)
    rows, ranks = [], []
    for q, h in zip(kept, hn):
        pos = q["source_passage_id"]
        if pos not in ids or pos.split("::")[0] not in train:
            sys.exit(f"{q['qid']}: positive {pos} unknown or not a train passage")
        cands = h["negatives"]
        for c in cands:
            p = c["passage_id"]
            if p not in ids or p.split("::")[0] not in train or p.split("::")[0] == pos.split("::")[0]:
                sys.exit(f"{q['qid']}: bad negative {p}")
        pick = [rng.choice(cands)] if cands else []
        ranks += [c["baseline_rank"] for c in pick]
        rows.append({"qid": q["qid"], "query": q["query"], "positive_passage_id": pos,
                     "negative_passage_ids": [c["passage_id"] for c in pick]})

    voice = {q["qid"]: q["voice"] for q in kept}
    with_neg = Counter(voice[r["qid"]] for r in rows if r["negative_passage_ids"])
    no_neg = Counter(voice[r["qid"]] for r in rows if not r["negative_passage_ids"])
    used = Counter(p for r in rows for p in r["negative_passage_ids"])
    print(f"rows {len(rows)}  with negative {sum(with_neg.values())}  without {sum(no_neg.values())}")
    for v in VOICES:
        n = with_neg[v] + no_neg[v]
        print(f"  {v:22s} with {with_neg[v]:6d}  without {no_neg[v]:5d} ({no_neg[v] / n:.1%})")
    share = no_neg["confused_undergrad"] / max(sum(no_neg.values()), 1)
    print(f"confused_undergrad share of the no-negative rows: {share:.1%}")
    print(f"distinct positives {len({r['positive_passage_id'] for r in rows})}  "
          f"distinct negatives used {len(used)}  max uses of one negative {max(used.values())}")
    print(f"chosen negative baseline rank: mean {sum(ranks) / len(ranks):.1f}, "
          f"min {min(ranks)}, max {max(ranks)}")

    if not args.write:
        print("dry run: nothing written")
        return
    tmp = OUT.with_suffix(".tmp")
    with tmp.open("w", encoding="utf-8", newline="\n") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    tmp.replace(OUT)
    OUT_MANIFEST.write_text(json.dumps({
        "n": len(rows), "n_with_negative": sum(with_neg.values()),
        "n_without_negative": sum(no_neg.values()), "k": K, "seed": SEED,
        "rule": "one negative drawn uniformly from the retained negatives of hard_negatives.jsonl",
        "hard_negatives_sha256": sha256_file(HARD), "passages_sha256": sha,
        "created": datetime.now().isoformat(timespec="seconds")}, indent=1) + "\n",
        encoding="utf-8")
    print(f"wrote {OUT} and {OUT_MANIFEST}")


if __name__ == "__main__":
    main()
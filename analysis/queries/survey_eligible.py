"""Survey eligible train passages per paper to choose a per-paper sampling cap (D042)."""
import json
from collections import Counter

from train.eligibility import MIN_FW, MIN_TOKENS, PASSAGES, SPLITS, fw_fraction

TARGET = 30000


def main():
    with open(SPLITS, encoding="utf-8") as f:
        train = set(json.load(f)["train"])
    n_train = 0
    per_paper = Counter()
    with open(PASSAGES, encoding="utf-8") as f:
        for line in f:
            p = json.loads(line)
            if p["paper_id"] not in train:
                continue
            n_train += 1
            if p["n_tokens"] >= MIN_TOKENS and fw_fraction(p["text"]) >= MIN_FW:
                per_paper[p["paper_id"]] += 1
    counts = sorted(per_paper.values())
    total = sum(counts)
    n = len(counts)
    print(f"train passages {n_train} (expect 81209)")
    print(f"eligible {total} (expect 78042), papers {n} (expect 2106)")
    q = lambda f: counts[int(f * (n - 1))]
    print(f"per paper: min {counts[0]}  p10 {q(.1)}  median {q(.5)}  p90 {q(.9)}  max {counts[-1]}")
    top = sum(counts[-(n // 10):])
    print(f"top 10% of papers hold {top / total:.1%} of eligible passages")
    rate = TARGET / total
    print(f"uniform sampling at {rate:.1%}: largest 5 papers would give "
          f"{[round(c * rate) for c in counts[-5:]]} passages each")
    print("cap  sample_size  papers_capped")
    for k in (10, 15, 20, 25, 30, 40, 60):
        size = sum(min(c, k) for c in counts)
        capped = sum(1 for c in counts if c > k)
        print(f"{k:>3}  {size:>11}  {capped:>13}")


if __name__ == "__main__":
    main()
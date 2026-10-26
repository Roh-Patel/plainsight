import json, random, sys
from ingest.chunk import function_word_frac

random.seed(0)
if sys.argv[1] == "--band":
    lo, hi = float(sys.argv[2]), float(sys.argv[3])
    rows = []
    with open("data/processed/passages.jsonl", encoding="utf-8") as f:
        for line in f:
            p = json.loads(line)
            s = function_word_frac(p["text"])
            if lo <= s < hi:
                rows.append((s, p["passage_id"], p["text"]))
    print(f"kept passages with function-word fraction in [{lo}, {hi}): {len(rows)}")
    for s, pid, t in random.sample(rows, min(15, len(rows))):
        print(f"[{s:.2f}] {pid} | {t[:100]!r}")
else:
    with open("data/processed/chunk_drops.jsonl", encoding="utf-8") as f:
        rows = [r for r in map(json.loads, f) if r["reason"] == sys.argv[1]]
    print(f"{sys.argv[1]}: {len(rows)} drops in {len({r['paper_id'] for r in rows})} papers")
    for r in random.sample(rows, min(15, len(rows))):
        print(f"[{r['n_tokens']}] {r['paper_id']} u{r['section_idx']} c{r['chunk_idx']} | {r['preview'][:100]!r}")
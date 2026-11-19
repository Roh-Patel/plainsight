import json, random
from collections import defaultdict
from transformers import AutoTokenizer

tok = AutoTokenizer.from_pretrained("BAAI/bge-base-en-v1.5")
tok.model_max_length = 10**9
WATCH = {("2609.27569", 30), ("2609.09631", 20), ("2504.12541", 2)}


def nonletter_frac(texts):
    out = []
    for seq in tok(texts, add_special_tokens=False)["input_ids"]:
        toks = tok.convert_ids_to_tokens(seq)
        out.append(sum(not any(c.isalpha() for c in t) for t in toks) / max(1, len(toks)))
    return out


random.seed(0)
by = defaultdict(list)
with open("data/processed/chunk_drops.jsonl", encoding="utf-8") as f:
    for line in f:
        d = json.loads(line)
        if d["reason"] != "unit_lt_50_tokens":
            by[d["reason"]].append(d)
for reason, rows in by.items():
    n_papers = len({r["paper_id"] for r in rows})
    print(f"\n=== DROPPED {reason}: {len(rows)} in {n_papers} papers; 8 random ===")
    for r in random.sample(rows, min(8, len(rows))):
        pv = r["preview"][:100].replace("\n", " ")
        print(f"[{r['n_tokens']}] {r['paper_id']} u{r['section_idx']} c{r['chunk_idx']} | {pv}")

with open("data/processed/passages.jsonl", encoding="utf-8") as f:
    kept = [json.loads(line) for line in f]
fr = []
for i in range(0, len(kept), 2000):
    fr.extend(nonletter_frac([p["text"] for p in kept[i:i + 2000]]))
sp = [p["text"].count(" ") / len(p["text"]) for p in kept]

s = sorted(fr)
print(f"\n=== KEPT: non-letter wordpiece fraction (n={len(kept)}) ===")
print({q: round(s[min(len(s) - 1, int(len(s) * q / 100))], 3) for q in (50, 95, 99)}, "max", round(s[-1], 3))
print({t: sum(x > t for x in fr) for t in (0.2, 0.3, 0.4)}, "kept passages above each threshold")

print("\n=== KEPT: 15 highest non-letter fraction ===")
for i in sorted(range(len(kept)), key=lambda i: -fr[i])[:15]:
    p = kept[i]
    print(f"[{fr[i]:.2f}] {p['passage_id']} | {p['text'][:90]!r}")

print("\n=== KEPT: 10 fewest spaces per character (prose is about 0.15) ===")
for i in sorted(range(len(kept)), key=lambda i: sp[i])[:10]:
    p = kept[i]
    print(f"[{sp[i]:.3f}] {p['passage_id']} | {p['text'][:90]!r}")

print("\n=== KEPT: watched units ===")
for i, p in enumerate(kept):
    if (p["paper_id"], p["section_idx"]) in WATCH:
        print(f"{p['passage_id']} n={p['n_tokens']} nonletter={fr[i]:.2f} sp={sp[i]:.3f} | {p['text'][:80]!r}")
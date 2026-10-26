import json, random
from transformers import AutoTokenizer

tok = AutoTokenizer.from_pretrained("BAAI/bge-base-en-v1.5")
tok.model_max_length = 10**9

big, tiny, max_idx = [], [], 0
with open("data/processed/parsed.jsonl", encoding="utf-8") as f:
    for line in f:
        row = json.loads(line)
        for u in row["units"]:
            max_idx = max(max_idx, u["idx"])
            paras = [p for p in u["text"].split("\n\n") if p.strip()]
            counts = [len(x) for x in tok(paras, add_special_tokens=False)["input_ids"]] if paras else []
            for p, n in zip(paras, counts):
                if n > 1000:
                    big.append((n, row["paper_id"], u["idx"], u["title"], p))
            if sum(counts) < 50:
                tiny.append((sum(counts), row["paper_id"], u["idx"], u["title"], u["text"]))

print(f"max section idx: {max_idx}")
print(f"\n=== {len(big)} paragraphs > 1000 tokens (top 12 by size) ===")
for n, pid, idx, title, p in sorted(big, reverse=True)[:12]:
    print(f"\n[{n} tok] {pid} unit {idx} | {title[:60]}")
    print(f"  sentence-end marks: {p.count('. ')}  newlines: {p.count(chr(10))}")
    print("  START:", p[:160].replace("\n", " "))
    print("  END  :", p[-120:].replace("\n", " "))

random.seed(0)
print(f"\n=== 15 random units < 50 tokens (of {len(tiny)}) ===")
for n, pid, idx, title, t in random.sample(tiny, 15):
    print(f"\n[{n} tok] {pid} unit {idx} | {title[:60]}")
    print("  ", t.replace("\n", " ")[:200])
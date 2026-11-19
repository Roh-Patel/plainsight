import json
import numpy as np
from transformers import AutoTokenizer

tok = AutoTokenizer.from_pretrained("BAAI/bge-reranker-base")
QUERY_BUDGET = 64   # generous for a learner question
SPECIALS = 4        # <s> q </s></s> p </s>
LIMIT = 512 - SPECIALS - QUERY_BUDGET

lens = {"text": [], "embed_text": []}
batch = {"text": [], "embed_text": []}

def flush():
    for k in lens:
        enc = tok(batch[k], add_special_tokens=False)["input_ids"]
        lens[k].extend(len(x) for x in enc)
        batch[k].clear()

with open("data/processed/passages.jsonl", encoding="utf-8") as f:
    for line in f:
        r = json.loads(line)
        batch["text"].append(r["text"])
        batch["embed_text"].append(r["embed_text"])
        if len(batch["text"]) == 2000:
            flush()
if batch["text"]:
    flush()

for k, v in lens.items():
    a = np.array(v)
    print(k, "n", len(a), "median", int(np.median(a)), "p99", int(np.percentile(a, 99)),
          "max", int(a.max()), "over", LIMIT, ":", int((a > LIMIT).sum()),
          f"({(a > LIMIT).mean():.4%})")
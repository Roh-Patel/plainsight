import json, math, sys
from transformers import AutoTokenizer

MODEL = "BAAI/bge-base-en-v1.5"
tok = AutoTokenizer.from_pretrained(MODEL)
tok.model_max_length = 10**9  # suppress the length warning


def ntok(texts):
    if not texts:
        return []
    return [len(x) for x in tok(texts, add_special_tokens=False)["input_ids"]]


def pct(v):
    v = sorted(v)
    n = len(v)
    out = {q: v[min(n - 1, int(n * q / 100))] for q in (5, 25, 50, 75, 95, 99)}
    out["max"] = v[-1]
    return out


limit = int(sys.argv[1]) if len(sys.argv) > 1 else None

titles = {}
with open("data/processed/candidates.jsonl", encoding="utf-8") as f:
    for line in f:
        r = json.loads(line)
        titles[r["paper_id"]] = r["title"]

unit_toks, para_toks, prefix_toks = [], [], []
n_papers = n_units = idx_over_99 = est_passages = 0

with open("data/processed/parsed.jsonl", encoding="utf-8") as f:
    for line in f:
        row = json.loads(line)
        n_papers += 1
        if limit and n_papers > limit:
            break
        ttl = ntok([titles[row["paper_id"]]])[0]
        for u in row["units"]:
            n_units += 1
            if u["idx"] > 99:
                idx_over_99 += 1
            paras = [p for p in u["text"].split("\n\n") if p.strip()]
            pt = ntok(paras)
            para_toks.extend(pt)
            ut = sum(pt)
            unit_toks.append(ut)
            est_passages += max(1, math.ceil(ut / 400))
            sec = ntok([u["title"]])[0] if u["title"] else 0
            prefix_toks.append(ttl + sec + 6)  # ~6 for separators

print(f"papers={n_papers} units={n_units} units_with_idx>99={idx_over_99}")
print("unit tokens     :", pct(unit_toks))
print("paragraph tokens:", pct(para_toks))
print("prefix tokens   :", pct(prefix_toks))
print(f"units < 50 tokens : {sum(t < 50 for t in unit_toks)}")
print(f"units < 250 tokens: {sum(t < 250 for t in unit_toks)}")
print(f"paragraphs > 400  : {sum(t > 400 for t in para_toks)}")
print(f"paragraphs > 1000 : {sum(t > 1000 for t in para_toks)}")
print(f"units where prefix+400 > 512: {sum(p + 400 > 512 for p in prefix_toks)}")
print(f"rough passage count (ceil(unit/400)): {est_passages}")
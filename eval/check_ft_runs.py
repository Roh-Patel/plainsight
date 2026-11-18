import json
import subprocess


def load(s):
    with open(f"runs/{s}.jsonl", encoding="utf-8") as f:
        return {r["qid"]: [x["passage_id"] for x in r["results"]]
                for r in map(json.loads, f)}


def overlap10(a, b):
    return sum(len(set(a[q][:10]) & set(b[q][:10])) / 10 for q in a) / len(a)


def same_top1(a, b):
    return sum(a[q][0] == b[q][0] for q in a)


runs = {s: load(s) for s in ["bm25", "dense_base", "dense_ft",
                             "hybrid_base", "hybrid_ft_embed"]}
pairs = [("dense_base", "dense_ft"), ("hybrid_base", "hybrid_ft_embed"),
         ("dense_ft", "hybrid_ft_embed"), ("bm25", "hybrid_ft_embed")]
for x, y in pairs:
    print(f"{x} vs {y}: top-10 overlap {overlap10(runs[x], runs[y]):.3f}, "
          f"same top-1 {same_top1(runs[x], runs[y])}/133")

out = subprocess.run(["git", "status", "--short", "runs"],
                     capture_output=True, text=True).stdout
print("git status runs/:")
print(out if out else "(clean)")
"""Cross-passage near-duplicate queries and overlap with the eval questions.
Report-only. Writes samples to scratch_synth/."""
import json
import random
import re
import sys
import time
from collections import defaultdict
from pathlib import Path
 
import numpy as np
import torch

from retrieval.dense import DenseSearcher

sys.stdout.reconfigure(errors="replace")

QUERIES = Path("data/synthetic/queries.jsonl")
QUESTIONS = Path("data/eval/questions.jsonl")
OUT = Path("scratch_synth")
VOICES = ("layperson", "confused_undergrad", "half_remembered_term")
THRESHOLDS = (0.85, 0.90, 0.93, 0.95, 0.97, 0.99)
BANDS = ((0.90, 0.95), (0.95, 0.97), (0.97, 1.01))
BLOCK = 2048
SEED = 17


def norm(s):
    return " ".join(re.sub(r"[^a-z0-9 ]", " ", s.lower()).split())


def main():
    with open(QUERIES, encoding="utf-8") as f:
        rows = [json.loads(l) for l in f]
    with open(QUESTIONS, encoding="utf-8") as f:
        qs = [json.loads(l) for l in f]
    n = len(rows)
    codes = {}
    code = np.array([codes.setdefault(r["source_passage_id"], len(codes)) for r in rows])
    paper = np.array([r["source_passage_id"].split("::")[0] for r in rows])
    voice = np.array([r["voice"] for r in rows])

    groups = defaultdict(set)
    for r in rows:
        groups[norm(r["query"])].add(r["source_passage_id"])
    exact = {k: v for k, v in groups.items() if len(v) > 1}
    print(f"queries {n}; normalised texts shared by 2+ source passages: {len(exact)} "
          f"({sum(len(v) for v in exact.values())} queries)")
    eval_norm = {norm(q["question"]) for q in qs}
    print(f"eval questions whose normalised text equals a synthetic query: "
          f"{sum(1 for k in groups if k in eval_norm)}")

    dense = DenseSearcher("dense_base", backend="numpy")
    t0 = time.time()
    emb = torch.from_numpy(dense.encode([r["query"] for r in rows], batch_size=128)).to("cuda")
    ev = torch.from_numpy(dense.encode([q["question"] for q in qs], batch_size=128)).to("cuda")
    print(f"encoded {n} + {len(qs)} in {time.time() - t0:.0f}s")

    ct = torch.from_numpy(code).to("cuda")
    sim = np.empty(n, dtype=np.float32)
    nn = np.empty(n, dtype=np.int64)
    for i in range(0, n, BLOCK):
        s = emb[i:i + BLOCK] @ emb.T
        s.masked_fill_(ct[i:i + BLOCK, None] == ct[None, :], -2.0)
        v, ix = s.max(dim=1)
        sim[i:i + BLOCK] = v.cpu().numpy()
        nn[i:i + BLOCK] = ix.cpu().numpy()

    pct = (50, 90, 99, 99.9)
    ref = {p: round(float(np.percentile(sim, p)), 3) for p in pct}
    print(f"\nsynthetic nearest query from a different passage, percentiles: {ref}")
    print("\nthreshold | queries with a near-duplicate elsewhere | per voice | same-paper share")
    for t in THRESHOLDS:
        m = sim >= t
        per = [int((m & (voice == v)).sum()) for v in VOICES]
        same = float((paper[m] == paper[nn[m]]).mean()) if m.any() else 0.0
        print(f">= {t:.2f} | {int(m.sum())} ({m.mean():.2%}) | {per} | {same:.2f}")

    rng = random.Random(SEED)
    OUT.mkdir(exist_ok=True)
    with open(OUT / "dup_samples.txt", "w", encoding="utf-8", newline="\n") as f:
        for lo, hi in BANDS:
            pool = np.flatnonzero((sim >= lo) & (sim < hi))
            f.write(f"== similarity {lo:.2f} to {hi:.2f}: {len(pool)} queries, 8 sampled. "
                    f"Do the two ask the same thing? ==\n")
            for i in rng.sample(list(pool), min(8, len(pool))):
                j = int(nn[i])
                f.write(f"sim {sim[i]:.3f} same_paper={paper[i] == paper[j]}\n"
                        f"  A {rows[i]['qid']} {voice[i]}: {rows[i]['query']}\n"
                        f"  B {rows[j]['qid']} {voice[j]}: {rows[j]['query']}\n")
            f.write("\n")

    s = ev @ emb.T
    v, ix = s.max(dim=1)
    v, ix = v.cpu().numpy(), ix.cpu().numpy()
    print(f"\neval question -> nearest synthetic query, percentiles: "
          f"{ {p: round(float(np.percentile(v, p)), 3) for p in pct} }")
    for t in THRESHOLDS:
        print(f"eval questions with a synthetic query >= {t:.2f}: {int((v >= t).sum())}")
    order = np.argsort(-v)
    with open(OUT / "eval_overlap.txt", "w", encoding="utf-8", newline="\n") as f:
        f.write("== 20 eval questions with the closest synthetic query. Paraphrase of the eval question? ==\n")
        for k in order[:20]:
            f.write(f"sim {v[k]:.3f}\n  EVAL {qs[k]['qid']} {qs[k]['style']}: {qs[k]['question']}\n"
                    f"  SYN  {rows[ix[k]]['qid']} {rows[ix[k]]['voice']}: {rows[ix[k]]['query']}\n")
    print(f"\nsamples written to {OUT / 'dup_samples.txt'} and {OUT / 'eval_overlap.txt'}")


if __name__ == "__main__":
    main()
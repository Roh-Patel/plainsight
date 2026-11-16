"""Run NoDuplicatesBatchSampler on the real triples and measure batch composition.
No model, no GPU, writes nothing."""
import argparse
import json
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from datasets import Dataset

try:
    import sentence_transformers.base.sampler as S
except ImportError:
    import sentence_transformers.sampler as S

TRIPLES = Path("data/synthetic/train_triples.jsonl")
QUERIES = Path("data/synthetic/queries.jsonl")
VOICES = ("layperson", "confused_undergrad", "half_remembered_term")


def paper(pid):
    return pid.split("::")[0]


def read(path):
    with path.open(encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


def run(name, rows, voice, with_neg, bs, seed):
    cols = {"query": [r["query"] for r in rows],
            "positive": [r["positive_passage_id"] for r in rows]}
    if with_neg:
        cols["negative"] = [r["negative_passage_ids"][0] for r in rows]
    ds = Dataset.from_dict(cols)
    gen = torch.Generator().manual_seed(seed)
    sampler = S.NoDuplicatesBatchSampler(ds, batch_size=bs, drop_last=True,
                                         generator=gen, seed=seed)
    t = time.time()
    batches = list(sampler)
    secs = time.time() - t
    seen = Counter(i for b in batches for i in b)
    print(f"\n== {name}: {len(rows)} rows, batch size {bs}, {secs:.0f}s")
    print(f"batches {len(batches)} (expected {len(sampler)}); rows yielded at least once "
          f"{len(seen)} ({len(seen) / len(rows):.1%}); rows yielded more than once "
          f"{sum(v > 1 for v in seen.values())}; short batches "
          f"{sum(len(b) != bs for b in batches)}")

    bad = 0
    any_hit, n_hit, pos_hit, conf = [], [], [], []
    for b in batches:
        vals = []
        for i in b:
            vals += [cols[c][i] for c in cols]
        bad += len(vals) != len(set(vals))
        pos_p = Counter(paper(cols["positive"][i]) for i in b)
        all_p = Counter(pos_p)
        if with_neg:
            all_p.update(paper(cols["negative"][i]) for i in b)
        for i in b:
            p = paper(cols["positive"][i])
            own = 1 + (with_neg and paper(cols["negative"][i]) == p)
            others = all_p[p] - own
            n_hit.append(others)
            any_hit.append(others > 0)
            pos_hit.append(pos_p[p] - 1)
        conf.append(np.mean([voice[rows[i]["qid"]] == "confused_undergrad" for i in b]))
    print(f"batches with a repeated value across columns: {bad} (expected 0)")
    print(f"same-paper candidates per query (positives + negatives of other rows): "
          f"mean {np.mean(n_hit):.3f}, max {max(n_hit)}; queries with at least one: "
          f"{np.mean(any_hit):.1%}")
    print(f"same-paper positives only: mean {np.mean(pos_hit):.3f}")
    print(f"confused_undergrad share per batch: mean {np.mean(conf):.1%}, "
          f"min {np.min(conf):.1%}, max {np.max(conf):.1%}")


def main():
    sys.stdout.reconfigure(errors="replace")
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch-size", type=int, required=True)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    print("sampler module:", S.__name__)
    print("_EXCLUDE_DATASET_COLUMNS:", getattr(S, "_EXCLUDE_DATASET_COLUMNS", "NOT FOUND"),
          " _MAX_RESAMPLE_PASSES:", getattr(S, "_MAX_RESAMPLE_PASSES", "NOT FOUND"))
    rows = read(TRIPLES)
    voice = {q["qid"]: q["voice"] for q in read(QUERIES)}
    a = [r for r in rows if r["negative_passage_ids"]]
    b = [r for r in rows if not r["negative_passage_ids"]]
    print(f"with negative {len(a)}  without {len(b)}")
    run("with-negative", a, voice, True, args.batch_size, args.seed)
    run("no-negative", b, voice, False, args.batch_size, args.seed)


if __name__ == "__main__":
    main()
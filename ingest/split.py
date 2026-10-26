"""Paper-level train/val/test split -> data/processed/splits.json"""
import argparse
import json
import random
import sys
from collections import Counter
from pathlib import Path

PASSAGES = Path("data/processed/passages.jsonl")
OUT = Path("data/processed/splits.json")
RATIOS = {"train": 0.7, "val": 0.1, "test": 0.2}


def paper_passage_counts(path=PASSAGES):
    counts = Counter()
    with open(path, encoding="utf-8") as f:
        for line in f:
            counts[json.loads(line)["paper_id"]] += 1
    return counts


def make_split(paper_ids, seed, ratios=RATIOS):
    ids = sorted(paper_ids)
    random.Random(seed).shuffle(ids)
    n = len(ids)
    n_train = round(n * ratios["train"])
    n_val = round(n * ratios["val"])
    return {
        "train": sorted(ids[:n_train]),
        "val": sorted(ids[n_train:n_train + n_val]),
        "test": sorted(ids[n_train + n_val:]),
    }


def build(seed, force):
    if OUT.exists() and not force:
        sys.exit(f"{OUT} exists. Splits are frozen after splitting; use --force only if you accept rebuilding everything downstream.")
    counts = paper_passage_counts()
    parts = make_split(counts.keys(), seed)
    manifest = {"seed": seed, "ratios": RATIOS, **parts}
    OUT.write_text(json.dumps(manifest), encoding="utf-8")
    print(f"wrote {OUT}: {len(counts)} papers")


def verify():
    m = json.loads(OUT.read_text(encoding="utf-8"))
    counts = paper_passage_counts()
    sets = {k: set(m[k]) for k in ("train", "val", "test")}
    assert not (sets["train"] & sets["val"]), "train/val overlap"
    assert not (sets["train"] & sets["test"]), "train/test overlap"
    assert not (sets["val"] & sets["test"]), "val/test overlap"
    assert set().union(*sets.values()) == set(counts), "split does not cover exactly the papers with passages"
    assert sum(len(m[k]) for k in sets) == len(counts), "duplicate IDs within a split"
    total_p = sum(counts.values())
    print(f"seed={m['seed']} papers={len(counts)} passages={total_p}")
    for k in ("train", "val", "test"):
        p = sum(counts[i] for i in sets[k])
        print(f"  {k:5s} papers={len(sets[k]):5d} ({len(sets[k])/len(counts):.3f})  passages={p:7d} ({p/total_p:.3f})")
    again = make_split(counts.keys(), m["seed"])
    assert all(again[k] == m[k] for k in sets), "re-running with the saved seed does not reproduce the manifest"
    print("OK: disjoint, complete, reproducible from seed")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--verify", action="store_true")
    a = ap.parse_args()
    verify() if a.verify else build(a.seed, a.force)
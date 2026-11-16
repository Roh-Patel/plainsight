"""Embedder fine-tune (cached MNRL, fp16). --smoke N runs N steps and saves nothing."""
import argparse
import json
import sys
import time
from pathlib import Path

import torch
from datasets import Dataset, DatasetDict
from sentence_transformers import (SentenceTransformer, SentenceTransformerTrainer,
                                   SentenceTransformerTrainingArguments)

try:
    from sentence_transformers.sentence_transformer.losses import CachedMultipleNegativesRankingLoss
except ImportError:
    from sentence_transformers.losses import CachedMultipleNegativesRankingLoss
try:
    from sentence_transformers.base.sampler import BatchSamplers
except ImportError:
    from sentence_transformers.sampler import BatchSamplers

from train.build_hard_negatives import DENSE_MANIFEST, PASSAGES, read_jsonl, sha256_file

TRIPLES = Path("data/synthetic/train_triples.jsonl")
TRIPLES_MANIFEST = Path("data/synthetic/train_triples.manifest.json")
SEED, LR, SCALE, EPOCHS, WARMUP, BATCH = 42, 2e-5, 50.0, 3, 81, 256


def build_datasets(prefix):
    man = json.loads(TRIPLES_MANIFEST.read_text(encoding="utf-8"))
    if man["passages_sha256"] != sha256_file(PASSAGES):
        sys.exit("passages.jsonl does not match the triples manifest")
    rows = list(read_jsonl(TRIPLES))
    need = {r["positive_passage_id"] for r in rows} | {
        p for r in rows for p in r["negative_passage_ids"]}
    text = {}
    for p in read_jsonl(PASSAGES):
        if p["passage_id"] in need:
            text[p["passage_id"]] = p["embed_text"]
    if len(text) != len(need):
        sys.exit("some passages are missing from passages.jsonl")
    a = [r for r in rows if r["negative_passage_ids"]]
    b = [r for r in rows if not r["negative_passage_ids"]]
    ds_a = Dataset.from_dict({"query": [prefix + r["query"] for r in a],
                              "positive": [text[r["positive_passage_id"]] for r in a],
                              "negative": [text[r["negative_passage_ids"][0]] for r in a]})
    ds_b = Dataset.from_dict({"query": [prefix + r["query"] for r in b],
                              "positive": [text[r["positive_passage_id"]] for r in b]})
    return DatasetDict({"with_negative": ds_a, "no_negative": ds_b})


def main():
    sys.stdout.reconfigure(errors="replace")
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", type=int, default=0)
    ap.add_argument("--mini-batch", type=int, default=64)
    ap.add_argument("--tag", default="emb-v1")
    ap.add_argument("--wandb", action="store_true")
    ap.add_argument("--only", default=None, choices=["with_negative", "no_negative"])
    args = ap.parse_args()

    man = json.loads(DENSE_MANIFEST.read_text(encoding="utf-8"))
    out_dir = Path("models/smoke") if args.smoke else Path("models/embedder") / args.tag
    if not args.smoke and out_dir.exists() and any(out_dir.iterdir()):
        sys.exit(f"{out_dir} is not empty; refusing to overwrite")

    data = build_datasets(man["query_prefix"])
    if args.only:
        data = DatasetDict({args.only: data[args.only]})
    print({k: len(v) for k, v in data.items()}, "columns",
          {k: v.column_names for k, v in data.items()})

    model = SentenceTransformer(man["model"], device="cuda")
    model.max_seq_length = man["max_seq_length"]
    loss = CachedMultipleNegativesRankingLoss(model, scale=SCALE, mini_batch_size=args.mini_batch)
    targs = SentenceTransformerTrainingArguments(
        output_dir=str(out_dir), num_train_epochs=EPOCHS, max_steps=args.smoke or -1,
        per_device_train_batch_size=BATCH, learning_rate=LR, warmup_steps=WARMUP,
        lr_scheduler_type="linear", fp16=True, batch_sampler=BatchSamplers.NO_DUPLICATES,
        dataloader_drop_last=True, seed=SEED, logging_steps=1 if args.smoke else 20,
        save_strategy="no" if args.smoke else "steps", save_steps=135, save_only_model=True,
        report_to="wandb" if args.wandb else "none", run_name=args.tag)
    trainer = SentenceTransformerTrainer(model=model, args=targs, train_dataset=data, loss=loss)

    dl = trainer.get_train_dataloader()
    print("batch sampler:", type(dl.batch_sampler).__name__, "-> per-dataset:",
          [type(s).__name__ for s in dl.batch_sampler.batch_samplers],
          " batches per epoch:", len(dl))
    for i, batch in enumerate(dl):
        if i == 6:
            break
        print("batch", i, {k: tuple(v.shape) for k, v in batch.items() if hasattr(v, "shape")})

    torch.cuda.reset_peak_memory_stats()
    t = time.time()
    trainer.train()
    secs = time.time() - t
    steps = args.smoke or int(trainer.state.max_steps)
    print(f"{steps} steps in {secs:.0f}s ({secs / steps:.1f} s/step); peak VRAM allocated "
          f"{torch.cuda.max_memory_allocated() / 1024 ** 3:.2f} GB; mini_batch {args.mini_batch}")
    if not args.smoke:
        trainer.save_model(str(out_dir / "final"))


if __name__ == "__main__":
    main()
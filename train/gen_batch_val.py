"""Run train.gen_batch against the val sample. Writes only under data/synthetic_val."""
import json
import sys
from pathlib import Path

import train.gen_batch as gb

VAL = Path("data/synthetic_val")
TRAIN_BATCHES = Path("data/synthetic/batches.jsonl")

gb.SAMPLE = VAL / "gen_sample.jsonl"
gb.SAMPLE_MANIFEST = VAL / "gen_sample.manifest.json"
gb.BATCHES = VAL / "batches.jsonl"
gb.RAW_DIR = VAL / "raw"


def check_config():
    for p in (gb.SAMPLE, gb.SAMPLE_MANIFEST, gb.BATCHES, gb.RAW_DIR):
        if VAL not in p.parents:
            sys.exit(f"path {p} is outside {VAL}")
    lines = TRAIN_BATCHES.read_text(encoding="utf-8").splitlines()
    ref = next(r for r in (json.loads(l) for l in lines if l.strip()) if r["shard"] == 0)
    now = (gb.PROMPT_VERSION, gb.PROMPT_SHA, gb.MODEL, gb.MAX_TOKENS)
    then = (ref["prompt_version"], ref["prompt_sha256"], ref["model"], ref["max_tokens"])
    if now != then:
        sys.exit(f"config differs from train shard 0:\nnow  {now}\nthen {then}")
    print("config identical to train shard 0:", gb.PROMPT_VERSION, gb.MODEL, gb.MAX_TOKENS,
          gb.PROMPT_SHA[:16])


if __name__ == "__main__":
    check_config()
    gb.main()
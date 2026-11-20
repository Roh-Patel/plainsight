"""T19 grading batches (D066): claude-sonnet-5-5 grades pool pairs under prompts P1 and P2.

  python -m eval.grade_batch dry-run --stage STAGE   build requests, print counts, no API call
  python -m eval.grade_batch submit  --stage STAGE   submit (refuses if the stage was submitted)
  python -m eval.grade_batch status  --stage STAGE
  python -m eval.grade_batch collect --stage STAGE   save verbatim results, then parse

Stages: pilot and pilot2 = the same 20 non-sample pool pairs (seed 30) x 2 prompts, format check
only, grades never stored or printed. pilot ran without a response schema; pilot2 and main use
OUTPUT_FORMAT (structured outputs, D066 amendment). main = all 4,770 pairs x 2 prompts. One user
message per request, no system prompt, default thinking and effort, max_tokens 4000. Results live in data/eval/raw/
(gitignored, never delete *.api.jsonl). The API key is read from the environment or .env.
"""
import argparse
import hashlib
import json
import os
import random
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path

import anthropic
from dotenv import load_dotenv

from eval.grade_prompt import PROMPT_SHA256, build, parse

MODEL = "claude-sonnet-5-5"
MAX_TOKENS = 4000
VERSIONS = ("p1", "p2")
FROZEN_SHA256 = {"p1": "ae16a6acfbd4a78317602c783fa579a59e0e6167591d85e0112d18275bd65725",
                 "p2": "4730926c27991757ecb121c9ef10ec80984b9a5df8d857933272b523453f2165"}
PILOT_SEED, PILOT_N = 30, 20
STAGES = ("pilot", "pilot2", "main")
OUTPUT_FORMAT = {"type": "json_schema",
                 "schema": {"type": "object",
                            "properties": {"grade": {"type": "integer", "enum": [0, 1, 2]}},
                            "required": ["grade"], "additionalProperties": False}}
BATCH_IN, BATCH_OUT = 1.00, 5.00  # USD per million tokens, Sonnet 5.5 batch (50% of $2 / $10)
POOL = Path("data/eval/pool.jsonl")
POOL_MANIFEST = Path("data/eval/pool.manifest.json")
PASSAGES = Path("data/processed/passages.jsonl")
CANDIDATES = Path("data/processed/candidates.jsonl")
QUESTIONS = Path("data/eval/questions.jsonl")
RAW = Path("data/eval/raw")


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def checks():
    if PROMPT_SHA256 != FROZEN_SHA256:
        sys.exit("grading prompts differ from the frozen hashes (D066); refusing")
    man = json.loads(POOL_MANIFEST.read_text(encoding="utf-8"))
    if sha256_file(PASSAGES) != man["passages_sha256"]:
        sys.exit("passages.jsonl does not match the pool manifest")
    if sha256_file(QUESTIONS) != man["questions_sha256"]:
        sys.exit("questions.jsonl does not match the pool manifest")
    return man


def load_pool():
    return [json.loads(l) for l in POOL.read_text(encoding="utf-8").splitlines() if l.strip()]


def stage_rows(stage, pool):
    """(row index, pool row) pairs for a stage."""
    if stage == "main":
        return list(enumerate(pool))
    candidates = [i for i, r in enumerate(pool) if not r["in_sample"]]
    return [(i, pool[i]) for i in sorted(random.Random(PILOT_SEED).sample(candidates, PILOT_N))]


def cid(stage, version, i):
    return f"{stage}-{version}-{i:04d}"


def build_requests(stage, rows):
    need = {r["passage_id"] for _, r in rows}
    pas = {}
    with PASSAGES.open(encoding="utf-8") as f:
        for line in f:
            p = json.loads(line)
            if p["passage_id"] in need:
                pas[p["passage_id"]] = p
    papers = {p["paper_id"] for p in pas.values()}
    title = {}
    with CANDIDATES.open(encoding="utf-8") as f:
        for line in f:
            c = json.loads(line)
            if c["paper_id"] in papers:
                title[c["paper_id"]] = c["title"]
    qs = {}
    for line in QUESTIONS.read_text(encoding="utf-8").splitlines():
        if line.strip():
            q = json.loads(line)
            qs[q["qid"]] = q["question"]
    if len(pas) != len(need) or len(title) != len(papers):
        sys.exit("missing passages or paper titles")
    reqs = []
    for v in VERSIONS:
        for i, r in rows:
            p = pas[r["passage_id"]]
            prompt = build(v, qs[r["qid"]], title[p["paper_id"]], p["section_title"], p["text"])
            params = {"model": MODEL, "max_tokens": MAX_TOKENS,
                      "messages": [{"role": "user", "content": prompt}]}
            if stage != "pilot":
                params["output_config"] = {"format": OUTPUT_FORMAT}
            reqs.append({"custom_id": cid(stage, v, i), "params": params})
    return reqs


def rec_path(stage):
    return RAW / f"{stage}.batch.json"


def client():
    load_dotenv()
    if not os.environ.get("ANTHROPIC_API_KEY"):
        sys.exit("ANTHROPIC_API_KEY is not set (environment or .env)")
    return anthropic.Anthropic()


def cmd_dry_run(args):
    checks()
    rows = stage_rows(args.stage, load_pool())
    reqs = build_requests(args.stage, rows)
    chars = sum(len(r["params"]["messages"][0]["content"]) for r in reqs)
    size = sum(len(json.dumps(r).encode("utf-8")) for r in reqs)
    print(f"stage {args.stage}: {len(rows)} pairs, {len(reqs)} requests, {chars / len(reqs):.0f} chars "
          f"per prompt, {size / 1024 ** 2:.1f} MB")
    print(f"in sample: {sum(r['in_sample'] for _, r in rows)}; ids {reqs[0]['custom_id']} .. {reqs[-1]['custom_id']}")
    print("prompt sha256", PROMPT_SHA256)


def cmd_submit(args):
    man = checks()
    if rec_path(args.stage).exists():
        sys.exit(f"{rec_path(args.stage)} exists; stage already submitted")
    if args.stage == "main" and not (RAW / "pilot2.jsonl").exists():
        sys.exit("collect pilot2 before submitting main")
    reqs = build_requests(args.stage, stage_rows(args.stage, load_pool()))
    c = client()
    RAW.mkdir(parents=True, exist_ok=True)
    batch = c.messages.batches.create(requests=reqs)
    rec = {"stage": args.stage, "batch_id": batch.id, "n_requests": len(reqs), "model": MODEL,
           "max_tokens": MAX_TOKENS, "thinking": "api default", "effort": "api default",
           "output_format": None if args.stage == "pilot" else OUTPUT_FORMAT,
           "prompt_sha256": PROMPT_SHA256, "pool_manifest_sha256": sha256_file(POOL_MANIFEST),
           "passages_sha256": man["passages_sha256"], "created": datetime.now().isoformat(timespec="seconds")}
    rec_path(args.stage).write_text(json.dumps(rec, indent=1) + "\n", encoding="utf-8", newline="\n")
    print(f"submitted {batch.id}: {len(reqs)} requests")


def cmd_status(args):
    rec = json.loads(rec_path(args.stage).read_text(encoding="utf-8"))
    b = client().messages.batches.retrieve(rec["batch_id"])
    print(rec["batch_id"], b.processing_status, b.request_counts)


def cmd_collect(args):
    rec = json.loads(rec_path(args.stage).read_text(encoding="utf-8"))
    api_path, out_path = RAW / f"{args.stage}.api.jsonl", RAW / f"{args.stage}.jsonl"
    if out_path.exists():
        sys.exit(f"{out_path} exists; refusing to overwrite")
    if not api_path.exists():
        c = client()
        b = c.messages.batches.retrieve(rec["batch_id"])
        if b.processing_status != "ended":
            print(f"not ended yet: {b.processing_status}, {b.request_counts}")
            return
        tmp = api_path.with_suffix(".tmp")
        with tmp.open("w", encoding="utf-8", newline="\n") as f:
            for item in c.messages.batches.results(rec["batch_id"]):
                f.write(item.model_dump_json() + "\n")
        tmp.replace(api_path)
        print(f"saved verbatim results to {api_path}")
    results = {}
    for line in api_path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            d = json.loads(line)
            results[d["custom_id"]] = d
    pool = load_pool()
    out = []
    for v in VERSIONS:
        for i, r in stage_rows(args.stage, pool):
            d = results.get(cid(args.stage, v, i))
            typ = "missing" if d is None else d["result"]["type"]
            row = {"custom_id": cid(args.stage, v, i), "qid": r["qid"], "passage_id": r["passage_id"],
                   "version": v, "result_type": typ, "stop_reason": None, "input_tokens": None,
                   "output_tokens": None, "grade": None}
            if typ == "succeeded":
                msg = d["result"]["message"]
                raw = "".join(b.get("text", "") for b in msg["content"] if b["type"] == "text")
                g = parse(raw) if msg["stop_reason"] == "end_turn" else None
                row.update(stop_reason=msg["stop_reason"], input_tokens=msg["usage"]["input_tokens"],
                           output_tokens=msg["usage"]["output_tokens"], grade=g)
            row["ok"] = row["grade"] is not None
            if args.stage != "main":
                row["grade"] = None  # D066: pilot grades are not read
            out.append(row)
    with out_path.open("w", encoding="utf-8", newline="\n") as f:
        for r in out:
            f.write(json.dumps(r) + "\n")
    n = len(out)
    tin = sum(r["input_tokens"] or 0 for r in out)
    tout = sum(r["output_tokens"] or 0 for r in out)
    print(f"wrote {out_path}: {n} rows")
    print(f"result types {dict(Counter(r['result_type'] for r in out))}")
    print(f"stop reasons {dict(Counter(r['stop_reason'] for r in out))}")
    print(f"parsed ok {sum(r['ok'] for r in out)}/{n} (p1 {sum(r['ok'] for r in out if r['version'] == 'p1')}, "
          f"p2 {sum(r['ok'] for r in out if r['version'] == 'p2')})")
    outs = sorted(r["output_tokens"] for r in out if r["output_tokens"] is not None)
    if outs:
        print(f"tokens per request: in mean {tin / n:.0f}; out mean {tout / n:.0f}, median "
              f"{outs[len(outs) // 2]}, max {outs[-1]}")
    print(f"cost at batch rates ${tin * BATCH_IN / 1e6 + tout * BATCH_OUT / 1e6:.3f} (check against the Console)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["dry-run", "submit", "status", "collect"])
    ap.add_argument("--stage", choices=STAGES, required=True)
    args = ap.parse_args()
    {"dry-run": cmd_dry_run, "submit": cmd_submit, "status": cmd_status, "collect": cmd_collect}[args.cmd](args)


if __name__ == "__main__":
    main()

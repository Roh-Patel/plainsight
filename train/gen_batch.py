"""Submit and collect Message Batches for synthetic query generation."""
import argparse
import hashlib
import json
import os
import re
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path

import anthropic

from train.draw_gen_sample import sha256_file
from train.query_prompt import PROMPT_VERSION, SYSTEM, VOICES, user_message

PASSAGES = Path("data/processed/passages.jsonl")
DENSE_MANIFEST = Path("data/processed/index/dense_base.manifest.json")
SAMPLE = Path("data/synthetic/gen_sample.jsonl")
SAMPLE_MANIFEST = Path("data/synthetic/gen_sample.manifest.json")
BATCHES = Path("data/synthetic/batches.jsonl")
RAW_DIR = Path("data/synthetic/raw")
MAX_TOKENS = 4000
MODEL = "claude-haiku-5-5"
ID_RE = re.compile(r"^[a-zA-Z0-9_-]{1,64}$")
MAX_BATCH_BYTES = 200 * 1024 * 1024
PARSE_FAIL_MAX = 0.01   # gate: parse failures (not truncated) must be below this rate
TRUNC_MAX = 0.02        # gate: stop_reason == max_tokens must be below this rate
BATCH_IN, BATCH_OUT = 0.05, 0.25  # USD per million tokens, assumed from public pricing
PROMPT_SHA = hashlib.sha256(SYSTEM.encode("utf-8")).hexdigest()


def parse(raw):
    """Return (obj or None, error string)."""
    s = raw.strip()
    i, j = s.find("{"), s.rfind("}")
    if i < 0 or j < i:
        return None, "no json object"
    try:
        obj = json.loads(s[i:j + 1])
    except json.JSONDecodeError as e:
        return None, f"json error: {e}"
    if not isinstance(obj, dict) or not isinstance(obj.get("usable"), bool):
        return None, "missing boolean usable"
    if obj["usable"]:
        q = obj.get("queries")
        if not isinstance(q, dict) or set(q) != set(VOICES):
            return None, "queries keys wrong"
        if not all(isinstance(q[v], str) and q[v].strip() for v in VOICES):
            return None, "empty or non-string query"
    return obj, ""


def cid(point_id):
    return f"p{point_id:06d}"


def need_key():
    if not os.environ.get("ANTHROPIC_API_KEY"):
        sys.exit("ANTHROPIC_API_KEY is not set in this cmd session")


def load_sample(shard):
    rows = [json.loads(l) for l in SAMPLE.read_text(encoding="utf-8").splitlines() if l.strip()]
    return [r for r in rows if r["shard"] == shard]


def check_hashes():
    sha = sha256_file(PASSAGES)
    dense = json.loads(DENSE_MANIFEST.read_text(encoding="utf-8"))["passages_sha256"]
    samp = json.loads(SAMPLE_MANIFEST.read_text(encoding="utf-8"))["passages_sha256"]
    if not (sha == dense == samp):
        sys.exit("passages.jsonl hash differs from the index manifest or the sample manifest")
    return sha


def load_passages(rows):
    need = {r["point_id"]: r for r in rows}
    out = {}
    with PASSAGES.open(encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i in need:
                p = json.loads(line)
                if p["passage_id"] != need[i]["passage_id"]:
                    sys.exit(f"line {i} is {p['passage_id']}, sample says {need[i]['passage_id']}")
                out[i] = p
    if len(out) != len(need):
        sys.exit("some sample lines are missing from passages.jsonl")
    return out


def build_params(p):
    return {"model": MODEL, "max_tokens": MAX_TOKENS, "system": SYSTEM,
            "messages": [{"role": "user",
                          "content": user_message(p["section_title"], p["text"])}]}


def build_requests(rows, passages):
    reqs = [{"custom_id": cid(r["point_id"]), "params": build_params(passages[r["point_id"]])}
            for r in rows]
    ids = [q["custom_id"] for q in reqs]
    if len(set(ids)) != len(ids) or not all(ID_RE.match(i) for i in ids):
        sys.exit("custom_id problem")
    return reqs


def submitted():
    if not BATCHES.exists():
        return {}
    recs = (json.loads(l) for l in BATCHES.read_text(encoding="utf-8").splitlines() if l.strip())
    return {r["shard"]: r for r in recs}


def cmd_dry_run(args):
    check_hashes()
    rows = load_sample(args.shard)
    total = len(rows)
    rows = rows[: args.n]
    reqs = build_requests(rows, load_passages(rows))
    sizes = [len(json.dumps(q).encode("utf-8")) for q in reqs]
    show = json.loads(json.dumps(reqs[0]))
    show["params"]["system"] = show["params"]["system"][:80] + "..."
    show["params"]["messages"][0]["content"] = show["params"]["messages"][0]["content"][:200] + "..."
    print(json.dumps(show, indent=1, ensure_ascii=False))
    print("param keys:", sorted(reqs[0]["params"]))
    print("custom_ids:", [q["custom_id"] for q in reqs])
    mean = sum(sizes) / len(sizes)
    print(f"mean request {mean / 1024:.1f} KB; shard {args.shard} has {total} requests, "
          f"projected {mean * total / 1024 ** 2:.1f} MB (limit 256 MB)")
    print("prompt sha256", PROMPT_SHA[:16], "version", PROMPT_VERSION, "model", MODEL,
          "max_tokens", MAX_TOKENS)


def cmd_submit(args):
    need_key()
    if args.shard in submitted():
        sys.exit(f"shard {args.shard} is already in {BATCHES}; refusing to submit again")
    if args.shard != 0:
        gate_path = RAW_DIR / "shard_0.gate.json"
        if not gate_path.exists():
            sys.exit("run collect and gate for shard 0 first")
        g = json.loads(gate_path.read_text(encoding="utf-8"))
        cfg = (PROMPT_VERSION, MODEL, MAX_TOKENS, PROMPT_SHA)
        if not g["passed"] or tuple(g["config"]) != cfg:
            sys.exit("shard 0 gate not passed, or the prompt/model/max_tokens changed since")
    sha = check_hashes()
    rows = load_sample(args.shard)
    if not rows:
        sys.exit("empty shard")
    reqs = build_requests(rows, load_passages(rows))
    size = sum(len(json.dumps(q).encode("utf-8")) for q in reqs)
    if size > MAX_BATCH_BYTES:
        sys.exit(f"batch is {size / 1024 ** 2:.0f} MB, over the self-imposed 200 MB limit")
    client = anthropic.Anthropic()
    batch = client.messages.batches.create(requests=reqs)
    rec = {"shard": args.shard, "batch_id": batch.id, "n_requests": len(reqs),
           "prompt_version": PROMPT_VERSION, "prompt_sha256": PROMPT_SHA, "model": MODEL,
           "max_tokens": MAX_TOKENS, "passages_sha256": sha,
           "created": datetime.now().isoformat(timespec="seconds")}
    with BATCHES.open("a", encoding="utf-8", newline="\n") as f:
        f.write(json.dumps(rec) + "\n")
    print(f"submitted shard {args.shard}: {batch.id}, {len(reqs)} requests, {size / 1024 ** 2:.1f} MB")


def cmd_status(args):
    need_key()
    client = anthropic.Anthropic()
    for shard, rec in sorted(submitted().items()):
        b = client.messages.batches.retrieve(rec["batch_id"])
        c = b.request_counts
        print(f"shard {shard} {rec['batch_id']} {b.processing_status} processing={c.processing} "
              f"succeeded={c.succeeded} errored={c.errored} canceled={c.canceled} expired={c.expired}")


def make_row(s, c, d, batch_id, shard):
    row = {"passage_id": s["passage_id"], "prompt_version": PROMPT_VERSION, "model": MODEL,
           "sampling": "api default", "max_tokens": MAX_TOKENS, "custom_id": c,
           "batch_id": batch_id, "shard": shard}
    typ = "missing" if d is None else d["result"]["type"]
    if typ != "succeeded":
        row.update(result_type=typ, stop_reason=None, input_tokens=None, output_tokens=None,
                   raw="", parsed=None, parse_error=f"batch result {typ}", thinking_tokens=None)
        return row
    msg = d["result"]["message"]
    usage = msg["usage"]
    raw = "".join(b.get("text", "") for b in msg["content"] if b["type"] == "text")
    obj, err = parse(raw)
    row.update(result_type=typ, stop_reason=msg["stop_reason"],
               input_tokens=usage["input_tokens"], output_tokens=usage["output_tokens"],
               raw=raw, parsed=obj, parse_error=err,
               thinking_tokens=(usage.get("output_tokens_details") or {}).get("thinking_tokens"))
    return row


def summarize(rows):
    n = len(rows)
    tin = sum(r["input_tokens"] or 0 for r in rows)
    tout = sum(r["output_tokens"] or 0 for r in rows)
    usable = sum(1 for r in rows if r["parsed"] and r["parsed"]["usable"])
    print(f"rows {n}  result types {dict(Counter(r['result_type'] for r in rows))}")
    print(f"stop reasons {dict(Counter(r['stop_reason'] for r in rows))}")
    print(f"parsed {sum(r['parsed'] is not None for r in rows)}  usable {usable} ({usable / n:.1%})")
    print(f"mean tokens in {tin / n:.0f} out {tout / n:.0f}; thinking max "
          f"{max((r['thinking_tokens'] or 0 for r in rows), default=0)}")
    cost = tin * BATCH_IN / 1e6 + tout * BATCH_OUT / 1e6
    print(f"tokens in {tin} out {tout}; cost at assumed batch rates ${cost:.3f} "
          f"(check against the Console)")


def cmd_collect(args):
    rec = submitted().get(args.shard)
    if rec is None:
        sys.exit(f"shard {args.shard} was never submitted")
    api_path = RAW_DIR / f"shard_{args.shard}.api.jsonl"
    out_path = RAW_DIR / f"shard_{args.shard}.jsonl"
    if out_path.exists():
        sys.exit(f"{out_path} exists; refusing to overwrite")
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    if not api_path.exists():
        need_key()
        client = anthropic.Anthropic()
        b = client.messages.batches.retrieve(rec["batch_id"])
        if b.processing_status != "ended":
            print(f"not ended yet: {b.processing_status}, {b.request_counts}")
            return
        tmp = api_path.with_suffix(".tmp")
        n = 0
        with tmp.open("w", encoding="utf-8", newline="\n") as f:
            for item in client.messages.batches.results(rec["batch_id"]):
                f.write(item.model_dump_json() + "\n")
                n += 1
        tmp.replace(api_path)
        print(f"saved {n} api results to {api_path}")
    results = {}
    for line in api_path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            d = json.loads(line)
            results[d["custom_id"]] = d
    sample = sorted(load_sample(args.shard), key=lambda r: r["order"])
    if len(results) != len(sample):
        print(f"WARNING: {len(results)} results for {len(sample)} requests")
    rows = [make_row(s, cid(s["point_id"]), results.get(cid(s["point_id"])),
                     rec["batch_id"], args.shard) for s in sample]
    with out_path.open("w", encoding="utf-8", newline="\n") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"wrote {out_path}")
    summarize(rows)


def cmd_gate(args):
    path = RAW_DIR / f"shard_{args.shard}.jsonl"
    rows = [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]
    n = len(rows)
    errors = sum(r["result_type"] != "succeeded" for r in rows)
    trunc = sum(r["stop_reason"] == "max_tokens" for r in rows)
    pfail = sum(r["result_type"] == "succeeded" and r["stop_reason"] != "max_tokens"
                and r["parsed"] is None for r in rows)
    checks = {
        "request errors == 0": errors == 0,
        f"parse failures < {PARSE_FAIL_MAX:.0%}": pfail / n < PARSE_FAIL_MAX,
        f"truncated < {TRUNC_MAX:.0%}": trunc / n < TRUNC_MAX,
    }
    for k, v in checks.items():
        print(("PASS " if v else "FAIL ") + k)
    print(f"n {n}  errors {errors}  parse_fail {pfail}  truncated {trunc}")
    passed = all(checks.values())
    gate = {"shard": args.shard, "passed": passed, "n": n, "errors": errors,
            "parse_fail": pfail, "truncated": trunc,
            "config": [PROMPT_VERSION, MODEL, MAX_TOKENS, PROMPT_SHA],
            "created": datetime.now().isoformat(timespec="seconds")}
    (RAW_DIR / f"shard_{args.shard}.gate.json").write_text(
        json.dumps(gate, indent=2) + "\n", encoding="utf-8")
    print("gate", "PASSED" if passed else "FAILED",
          "(form checks via analysis.queries.check_dev are separate)")


def main():
    sys.stdout.reconfigure(errors="replace")
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("dry-run")
    p.add_argument("--shard", type=int, required=True)
    p.add_argument("--n", type=int, default=5)
    p.set_defaults(func=cmd_dry_run)
    for name, fn in (("submit", cmd_submit), ("collect", cmd_collect), ("gate", cmd_gate)):
        p = sub.add_parser(name)
        p.add_argument("--shard", type=int, required=True)
        p.set_defaults(func=fn)
    sub.add_parser("status").set_defaults(func=cmd_status)
    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
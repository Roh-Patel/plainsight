"""Assemble queries.jsonl and gen_log.jsonl from the collected batch shards."""
import argparse
import json
import sys
from collections import Counter
from pathlib import Path

from train.gen_batch import MODEL, RAW_DIR, SAMPLE, check_hashes, cid, submitted
from train.query_prompt import VOICES
from train.survey_eligible import SPLITS

OUT_Q = Path("data/synthetic/queries.jsonl")
OUT_LOG = Path("data/synthetic/gen_log.jsonl")


def read_jsonl(path):
    return [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]


def write_jsonl(path, rows):
    tmp = path.with_suffix(".tmp")
    with tmp.open("w", encoding="utf-8", newline="\n") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    tmp.replace(path)


def build():
    check_hashes()
    sample = sorted(read_jsonl(SAMPLE), key=lambda r: r["order"])
    done = submitted()
    rows = {}
    for shard in sorted({r["shard"] for r in sample}):
        path = RAW_DIR / f"shard_{shard}.jsonl"
        if shard not in done or not path.exists():
            sys.exit(f"shard {shard} is not submitted or not collected")
        for r in read_jsonl(path):
            if r["passage_id"] in rows:
                sys.exit(f"duplicate row for {r['passage_id']}")
            rows[r["passage_id"]] = r
    if set(rows) != {s["passage_id"] for s in sample}:
        sys.exit("collected rows do not match the sample")
    train = set(json.loads(Path(SPLITS).read_text(encoding="utf-8"))["train"])

    queries, log = [], []
    for s in sample:
        r = rows[s["passage_id"]]
        if r["shard"] != s["shard"] or r["custom_id"] != cid(s["point_id"]):
            sys.exit(f"row {s['passage_id']} does not match the sample")
        if s["paper_id"] not in train:
            sys.exit(f"{s['passage_id']} is not from a train paper")
        parsed, reason = r["parsed"], ""
        if r["stop_reason"] == "refusal":
            status = "refusal"
        elif r["result_type"] != "succeeded":
            status, reason = "request_error", r["result_type"]
        elif parsed is None:
            status, reason = "parse_error", r["parse_error"]
        elif not parsed["usable"]:
            status, reason = "unusable", parsed.get("reason", "")
        else:
            status = "ok"
            for v in VOICES:
                queries.append({"qid": f"syn-{len(queries) + 1:06d}",
                                "source_passage_id": s["passage_id"], "voice": v,
                                "query": parsed["queries"][v].strip(),
                                "generator": MODEL, "kept": True, "drop_reason": None})
        log.append({"passage_id": s["passage_id"], "paper_id": s["paper_id"],
                    "shard": s["shard"], "custom_id": r["custom_id"], "status": status,
                    "reason": reason, "stop_reason": r["stop_reason"],
                    "input_tokens": r["input_tokens"], "output_tokens": r["output_tokens"],
                    "thinking_tokens": r["thinking_tokens"],
                    "n_queries": len(VOICES) if status == "ok" else 0})
    return queries, log


def main():
    sys.stdout.reconfigure(errors="replace")
    ap = argparse.ArgumentParser()
    ap.add_argument("--verify", action="store_true")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    queries, log = build()
    if args.verify:
        same = read_jsonl(OUT_Q) == queries and read_jsonl(OUT_LOG) == log
        print("reproducible from raw:", same)
        return
    for p in (OUT_Q, OUT_LOG):
        if p.exists() and not args.force:
            sys.exit(f"{p} exists; refusing to overwrite")
    write_jsonl(OUT_Q, queries)
    write_jsonl(OUT_LOG, log)
    status = Counter(l["status"] for l in log)
    assert sum(status.values()) == len(log)
    assert status["ok"] * len(VOICES) == len(queries)
    print(f"sample rows {len(log)}  status {dict(status)}")
    print(f"queries {len(queries)}  per voice "
          f"{dict(Counter(q['voice'] for q in queries))}")
    print(f"qids {queries[0]['qid']} to {queries[-1]['qid']}")


if __name__ == "__main__":
    main()
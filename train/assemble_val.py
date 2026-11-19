"""Assemble and filter the val dev queries. Writes only under data/synthetic_val."""
import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import train.gen_batch as gb
import train.gen_batch_val as gbv  # importing it redirects gen_batch paths to data/synthetic_val
from train.apply_filters import ORDER, form_reasons, norm
from train.assemble_queries import read_jsonl, write_jsonl
from train.query_prompt import VOICES
from train.eligibility import SPLITS

VAL = Path("data/synthetic_val")
OUT_Q = VAL / "queries.jsonl"
OUT_LOG = VAL / "gen_log.jsonl"


def build():
    gbv.check_config()
    gb.check_hashes()
    if set(gb.submitted()) != {0}:
        sys.exit("expected exactly shard 0 in the val batches file")
    sample = sorted(read_jsonl(gb.SAMPLE), key=lambda r: r["order"])
    rows = {}
    for r in read_jsonl(gb.RAW_DIR / "shard_0.jsonl"):
        if r["passage_id"] in rows:
            sys.exit(f"duplicate row for {r['passage_id']}")
        rows[r["passage_id"]] = r
    if set(rows) != {s["passage_id"] for s in sample}:
        sys.exit("collected rows do not match the sample")
    splits = json.loads(Path(SPLITS).read_text(encoding="utf-8"))
    val, train = set(splits["val"]), set(splits["train"])

    need = {s["point_id"]: s["passage_id"] for s in sample}
    text = {}
    with gb.PASSAGES.open(encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i in need:
                p = json.loads(line)
                if p["passage_id"] != need[i]:
                    sys.exit(f"line {i} is {p['passage_id']}, sample says {need[i]}")
                text[p["passage_id"]] = p["text"]
    if len(text) != len(need):
        sys.exit("some sample passages are missing from passages.jsonl")

    queries, log = [], []
    for s in sample:
        r = rows[s["passage_id"]]
        if r["custom_id"] != gb.cid(s["point_id"]):
            sys.exit(f"row {s['passage_id']} does not match the sample")
        if s["paper_id"] not in val or s["paper_id"] in train:
            sys.exit(f"{s['passage_id']} is not from a val-only paper")
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
                queries.append({"qid": f"val-{len(queries) + 1:06d}",
                                "source_passage_id": s["passage_id"], "voice": v,
                                "query": parsed["queries"][v].strip(),
                                "generator": gb.MODEL, "kept": True, "drop_reason": None})
        log.append({"passage_id": s["passage_id"], "paper_id": s["paper_id"],
                    "custom_id": r["custom_id"], "status": status, "reason": reason,
                    "stop_reason": r["stop_reason"], "input_tokens": r["input_tokens"],
                    "output_tokens": r["output_tokens"],
                    "thinking_tokens": r["thinking_tokens"],
                    "n_queries": len(VOICES) if status == "ok" else 0})

    reasons = [form_reasons(q["query"], text[q["source_passage_id"]]) for q in queries]
    seen = {}
    for i, q in enumerate(queries):
        if reasons[i]:
            continue
        k = norm(q["query"])
        if k in seen and queries[seen[k]]["source_passage_id"] != q["source_passage_id"]:
            reasons[i].append("duplicate_text")
        else:
            seen.setdefault(k, i)
    for q, rs in zip(queries, reasons):
        q["kept"] = not rs
        q["drop_reason"] = rs[0] if rs else None
    return queries, log, reasons


def main():
    sys.stdout.reconfigure(errors="replace")
    ap = argparse.ArgumentParser()
    ap.add_argument("--write", action="store_true")
    args = ap.parse_args()
    queries, log, reasons = build()

    status = Counter(l["status"] for l in log)
    print(f"sample rows {len(log)}  status {dict(status)}")
    print(f"queries {len(queries)}  qids {queries[0]['qid']} to {queries[-1]['qid']}")
    print("independent rule counts: " +
          ", ".join(f"{r} {sum(r in rs for rs in reasons)}" for r in ORDER[:-1]))
    first = Counter(q["drop_reason"] for q in queries if not q["kept"])
    print("drop_reason (first rule): " + ", ".join(f"{r} {first.get(r, 0)}" for r in ORDER[:-1]))
    kept = [q for q in queries if q["kept"]]
    print(f"kept {len(kept)} of {len(queries)} ({len(kept) / len(queries):.1%}); per voice "
          f"{ {v: sum(q['voice'] == v for q in kept) for v in VOICES} }")
    print(f"papers with at least one kept query: {len({q['source_passage_id'].split('::')[0] for q in kept})}")

    if not args.write:
        print("dry run: nothing written")
        return
    for p in (OUT_Q, OUT_LOG):
        if p.exists():
            sys.exit(f"{p} exists; refusing to overwrite")
    write_jsonl(OUT_Q, queries)
    write_jsonl(OUT_LOG, log)
    print(f"wrote {OUT_Q} and {OUT_LOG}")


if __name__ == "__main__":
    main()
"""Run the query-generation prompt on a sample, synchronously. Saves raw responses."""
import argparse
import json
import os
import sys
from pathlib import Path

import anthropic

from train.gen_batch import MODEL, parse
from train.query_prompt import PROMPT_VERSION, SYSTEM, user_message

PASSAGES = Path("data/processed/passages.jsonl")


def main():
    sys.stdout.reconfigure(errors="replace")
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-tokens", type=int, default=4000)
    args = ap.parse_args()

    out = Path(args.out)
    if out.exists():
        sys.exit(f"{out} exists; refusing to overwrite. Use a new file name.")
    if not os.environ.get("ANTHROPIC_API_KEY"):
        sys.exit("ANTHROPIC_API_KEY is not set in this cmd session")

    ids = [json.loads(l)["passage_id"] for l in
           Path(args.sample).read_text(encoding="utf-8").splitlines() if l.strip()]
    need = set(ids)
    pas = {}
    with PASSAGES.open(encoding="utf-8") as f:
        for line in f:
            p = json.loads(line)
            if p["passage_id"] in need:
                pas[p["passage_id"]] = p
    missing = need - set(pas)
    if missing:
        sys.exit(f"{len(missing)} sample ids not in passages.jsonl")

    client = anthropic.Anthropic()
    rows = []
    n_ok = n_usable = tin = tout = 0
    for k, pid in enumerate(ids, 1):
        p = pas[pid]
        resp = client.messages.create(
            model=MODEL, max_tokens=args.max_tokens,
            system=SYSTEM,
            messages=[{"role": "user",
                       "content": user_message(p["section_title"], p["text"])}])
        raw = "".join(b.text for b in resp.content if b.type == "text")
        obj, err = parse(raw)
        row = {"passage_id": pid, "prompt_version": PROMPT_VERSION, "model": MODEL,
               "sampling": "api default", "stop_reason": resp.stop_reason,
               "input_tokens": resp.usage.input_tokens,
               "output_tokens": resp.usage.output_tokens,
               "raw": raw, "parsed": obj, "parse_error": err,
               "max_tokens": args.max_tokens,
               "thinking_tokens": (resp.usage.model_dump()
                                   .get("output_tokens_details") or {}).get("thinking_tokens"),}
        rows.append(row)
        n_ok += obj is not None
        n_usable += bool(obj and obj["usable"])
        tin += resp.usage.input_tokens
        tout += resp.usage.output_tokens
        print(f"{k}/{len(ids)} {pid} parsed={obj is not None} "
              f"usable={bool(obj and obj['usable'])}", flush=True)

    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8", newline="\n") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"\nprompt {PROMPT_VERSION}, model {MODEL}")
    print(f"parsed {n_ok}/{len(ids)}, usable {n_usable}/{len(ids)}")
    print(f"tokens in {tin}, out {tout} (mean per call {tin//len(ids)} / {tout//len(ids)})")
    print(f"stop_reason max_tokens: {sum(r['stop_reason'] == 'max_tokens' for r in rows)}")
    tt = [r["thinking_tokens"] for r in rows if r["thinking_tokens"] is not None]
    if tt:
        print(f"thinking tokens mean/max: {sum(tt)//len(tt)} / {max(tt)}")
    print(f"output tokens max: {max(r['output_tokens'] for r in rows)}")


if __name__ == "__main__":
    main()
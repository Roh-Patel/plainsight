"""List generation rows with no parsed output, with a passage preview."""
import json
import sys
from pathlib import Path

PASSAGES = Path("data/processed/passages.jsonl")
RAW = Path("data/synthetic/raw")


def main():
    sys.stdout.reconfigure(errors="replace")
    bad = []
    for path in sorted(RAW.glob("shard_[0-9].jsonl")):
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                r = json.loads(line)
                if r["parsed"] is None:
                    bad.append(r)
    need = {int(r["custom_id"][1:]): r for r in bad}
    with PASSAGES.open(encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i in need:
                p = json.loads(line)
                if p["passage_id"] != need[i]["passage_id"]:
                    sys.exit(f"line {i} does not match {need[i]['passage_id']}")
                need[i]["_p"] = p
    print(f"{len(bad)} rows without parsed output")
    for r in sorted(bad, key=lambda r: (r["shard"], r["passage_id"])):
        p = r["_p"]
        print(f"shard {r['shard']} {r['passage_id']} stop={r['stop_reason']} "
              f"out_tokens={r['output_tokens']} err={r['parse_error']!r}")
        print(f"   section: {p['section_title'][:80]}")
        print(f"   text: {p['text'][:160]!r}")
        print(f"   raw: {r['raw'][:300]!r}")


if __name__ == "__main__":
    main()
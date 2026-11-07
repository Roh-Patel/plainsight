"""Summarise a raw generation file: why rows failed to parse."""
import json
import sys
from collections import Counter

rows = [json.loads(l) for l in open(sys.argv[1], encoding="utf-8") if l.strip()]
print("stop_reason:", dict(Counter(r["stop_reason"] for r in rows)))
print("parse_error:", dict(Counter(r["parse_error"][:40] for r in rows)))
print("raw empty:", sum(1 for r in rows if not r["raw"].strip()))
print("output_tokens min/median/max:",
      min(r["output_tokens"] for r in rows),
      sorted(r["output_tokens"] for r in rows)[len(rows) // 2],
      max(r["output_tokens"] for r in rows))
print("raw chars  min/median/max:",
      min(len(r["raw"]) for r in rows),
      sorted(len(r["raw"]) for r in rows)[len(rows) // 2],
      max(len(r["raw"]) for r in rows))
bad = [r for r in rows if r["parse_error"]][:2]
for r in bad:
    print("\n---", r["passage_id"], r["stop_reason"], r["output_tokens"], "tokens")
    print("HEAD:", r["raw"][:400].encode("ascii", "replace").decode())
    print("TAIL:", r["raw"][-200:].encode("ascii", "replace").decode())
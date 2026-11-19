"""Blind review sheet: each passage with its three queries. No ranks, no overlap."""
import json
import random
import sys
from pathlib import Path

from train.query_prompt import VOICES

raw_path, txt_out, tsv_out = sys.argv[1], Path(sys.argv[2]), Path(sys.argv[3])
for p in (txt_out, tsv_out):
    if p.exists():
        sys.exit(f"{p} exists; refusing to overwrite")

rows = [json.loads(l) for l in open(raw_path, encoding="utf-8") if l.strip()]
rows = [r for r in rows if r["parsed"] and r["parsed"]["usable"]]
need = {r["passage_id"] for r in rows}
pas = {}
with open("data/processed/passages.jsonl", encoding="utf-8") as f:
    for line in f:
        p = json.loads(line)
        if p["passage_id"] in need:
            pas[p["passage_id"]] = p

random.Random(0).shuffle(rows)
tag = {"layperson": "L", "confused_undergrad": "U", "half_remembered_term": "H"}

txt, tsv = [], ["n\tpassage_id\tvoice\tA\tS\tV\tnote"]
for k, r in enumerate(rows, 1):
    p = pas[r["passage_id"]]
    txt.append(f"=== #{k:02d}   section: {p['section_title'] or '(none)'}")
    txt.append("")
    txt.append(p["text"])
    txt.append("")
    for v in VOICES:
        txt.append(f"  {tag[v]}: {r['parsed']['queries'][v]}")
        tsv.append(f"{k}\t{r['passage_id']}\t{v}\t\t\t\t")
    txt.append("")
    txt.append("")

txt_out.parent.mkdir(parents=True, exist_ok=True)
txt_out.write_text("\n".join(txt), encoding="utf-8", newline="\n")
tsv_out.write_text("\n".join(tsv) + "\n", encoding="utf-8", newline="\n")
print(f"{len(rows)} passages, {len(rows) * 3} queries")
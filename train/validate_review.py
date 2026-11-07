"""Validate a filled review TSV and score it against the rubric thresholds."""
import csv
import json
import sys
from collections import Counter

from train.query_prompt import VOICES

sys.stdout.reconfigure(errors="replace")
raw_path, tsv_path = sys.argv[1], sys.argv[2]

try:
    text = open(tsv_path, encoding="utf-8-sig").read()
except UnicodeDecodeError:
    sys.exit("TSV is not UTF-8; re-save it as UTF-8 from your editor")
rows = list(csv.DictReader(text.splitlines(), delimiter="\t"))
for col in ("passage_id", "voice", "A", "V", "N"):
    if col not in rows[0]:
        sys.exit(f"missing column {col}; found {list(rows[0])}")

raw = [json.loads(l) for l in open(raw_path, encoding="utf-8") if l.strip()]
qtext = {(r["passage_id"], v): r["parsed"]["queries"][v]
         for r in raw if r["parsed"] and r["parsed"]["usable"] for v in VOICES}

errs = []
if len(rows) != len(qtext):
    errs.append(f"{len(rows)} rows in TSV, {len(qtext)} queries in raw file")
keys = [(r["passage_id"], r["voice"]) for r in rows]
if len(set(keys)) != len(keys):
    errs.append("duplicate (passage_id, voice) rows")
if set(keys) != set(qtext):
    errs.append("TSV rows do not match the raw file's queries")
allowed = {"A": {"0", "1", "2"}, "V": {"0", "1"}, "N": {"0", "1"}}
for i, r in enumerate(rows, 2):
    for c, ok in allowed.items():
        if r[c].strip() not in ok:
            errs.append(f"line {i} ({r['passage_id']} {r['voice']}): {c}={r[c]!r}")
if errs:
    print("\n".join(errs[:30]))
    sys.exit(f"{len(errs)} validation errors; fix the TSV and rerun")

for r in rows:
    r["q"] = qtext[(r["passage_id"], r["voice"])]
    for c in allowed:
        r[c] = int(r[c])
n = len(rows)
print(f"{n} rows valid; S column ignored (withdrawn as a hand check)\n")


def pct(k, d):
    return f"{k}/{d} = {k/d:.1%}"


print(f"A = 0: {pct(sum(r['A']==0 for r in rows), n)}   (threshold <= 5%)")
print(f"A = 2: {pct(sum(r['A']==2 for r in rows), n)}   (threshold >= 80%)")
print(f"N = 1: {pct(sum(r['N']==1 for r in rows), n)}   (threshold >= 80%)")
print("V = 1 per voice (threshold >= 90%):")
for v in VOICES:
    vr = [r for r in rows if r["voice"] == v]
    k = sum(r["V"] == 1 for r in vr)
    print(f"   {v:22s} {pct(k, len(vr))}  {'OK' if k/len(vr) >= 0.9 else 'FAIL'}")

print("\nper voice (report-only): A=2 share, N=1 share")
for v in VOICES:
    vr = [r for r in rows if r["voice"] == v]
    print(f"   {v:22s} A=2 {sum(r['A']==2 for r in vr)}/{len(vr)}   "
          f"N=1 {sum(r['N']==1 for r in vr)}/{len(vr)}")

print("\nN x A:", dict(sorted(Counter((r['N'], r['A']) for r in rows).items())))

print("\nqueries failing any criterion (A<2, V=0 or N=0):")
for r in rows:
    if r["A"] < 2 or r["V"] == 0 or r["N"] == 0:
        print(f"- {r['passage_id']} {r['voice'][:6]} A{r['A']} V{r['V']} N{r['N']}: {r['q']}")
        if r.get("note", "").strip():
            print(f"    note: {r['note'].strip()}")
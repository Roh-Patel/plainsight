"""Automatic checks on a raw generation file against the pre-declared thresholds."""
import json
import re
import statistics as st
import sys
from collections import Counter
from pathlib import Path

from ingest.chunk import STOP
from train.query_prompt import VOICES

sys.stdout.reconfigure(errors="replace")
raw_path = sys.argv[1]
rows = [json.loads(l) for l in open(raw_path, encoding="utf-8") if l.strip()]
need = {r["passage_id"] for r in rows}
pas = {}
with open("data/processed/passages.jsonl", encoding="utf-8") as f:
    for line in f:
        p = json.loads(line)
        if p["passage_id"] in need:
            pas[p["passage_id"]] = p

BANNED = ["the paper", "the authors", "this study", "the passage", "the text",
          "the article", "the researchers", "the section", "the figure", "the table"]
STOPS = {w.lower() for w in STOP} | {
    "what", "why", "how", "does", "do", "did", "is", "are", "was", "were", "can",
    "could", "would", "will", "which", "who", "when", "where", "about", "than",
    "that", "this", "these", "those", "there", "their", "they", "them", "it", "its",
    "i", "me", "my", "you", "your", "we", "our", "be", "been", "being", "have",
    "has", "had", "so", "if", "then", "also", "just", "like", "such", "into",
    "over", "out", "up", "down", "very", "much", "many", "more", "most", "some",
    "any", "each", "both", "other", "not", "no", "but", "or", "and", "the", "a", "an"}
TOK = re.compile(r"[a-z0-9]+(?:[-'][a-z0-9]+)*")


def content_words(s):
    return {w for w in TOK.findall(s.lower()) if w not in STOPS and len(w) > 1}


def jaccard(a, b):
    a, b = content_words(a), content_words(b)
    return len(a & b) / len(a | b) if a | b else 0.0


n = len(rows)
parsed = [r for r in rows if r["parsed"] is not None]
usable = [r for r in parsed if r["parsed"]["usable"]]
print(f"file {raw_path}  prompt {rows[0]['prompt_version']}")
print(f"parsed {len(parsed)}/{n} (threshold >= 49/50)")
print(f"usable {len(usable)}/{n} = {len(usable)/n:.0%} (threshold >= 50%)")

queries = []
for r in usable:
    for v in VOICES:
        queries.append((r["passage_id"], v, r["parsed"]["queries"][v]))

banned = [(pid, v, q) for pid, v, q in queries if any(b in q.lower() for b in BANNED)]
print(f"banned phrase {len(banned)}/{len(queries)} = {len(banned)/len(queries):.1%} (threshold <= 3%)")
for pid, v, q in banned:
    print(f"   {pid} {v}: {q}")

lens = [len(q.split()) for _, _, q in queries]
ok_len = sum(6 <= L <= 40 for L in lens)
print(f"length 6-40 words {ok_len}/{len(lens)} = {ok_len/len(lens):.1%} (threshold >= 95%); "
      f"min/median/max {min(lens)}/{int(st.median(lens))}/{max(lens)}")

print("overlap (share of query content words found in source passage), report-only except median <= 0.75:")
for v in VOICES:
    vals = []
    for pid, vv, q in queries:
        if vv != v:
            continue
        qw = content_words(q)
        vals.append(len(qw & content_words(pas[pid]["text"])) / len(qw) if qw else 0.0)
    vals.sort()
    med = st.median(vals)
    flag = "OK" if med <= 0.75 else "FAIL"
    print(f"   {v:22s} median {med:.2f}  p25 {vals[len(vals)//4]:.2f}  p75 {vals[3*len(vals)//4]:.2f}  {flag}")

dups = 0
for r in usable:
    qs = [r["parsed"]["queries"][v] for v in VOICES]
    if any(jaccard(qs[i], qs[j]) > 0.6 for i in range(3) for j in range(i + 1, 3)):
        dups += 1
        print(f"   near-duplicate within passage: {r['passage_id']}")
print(f"near-duplicate passages {dups}/{len(usable)} = {dups/len(usable):.1%} (threshold <= 5%)")

def ends_q(q):
    return q.strip().rstrip("\"')").endswith("?")


nq = [(pid, v, q) for pid, v, q in queries if not ends_q(q)]
print(f"ends with '?' {len(queries) - len(nq)}/{len(queries)} = "
      f"{(len(queries) - len(nq)) / len(queries):.1%} (threshold >= 95%)")
for v in VOICES:
    nv = sum(1 for _, vv, _ in queries if vv == v)
    mv = sum(1 for _, vv, _ in nq if vv == v)
    print(f"   {v:22s} missing '?': {mv}/{nv}")
for pid, v, q in nq[:6]:
    print(f"   e.g. {v}: {q[:120]}")

print("most common first three words per voice (threshold: top share <= 35%):")
for v in VOICES:
    c = Counter(" ".join(q.lower().split()[:3]) for _, vv, q in queries if vv == v)
    top, k = c.most_common(1)[0]
    nv = sum(c.values())
    print(f"   {v:22s} '{top}' {k}/{nv} = {k / nv:.0%}  {'OK' if k / nv <= 0.35 else 'FAIL'}")

DEIXIS = re.compile(r"\b(this|these)\b|\bsuch\b(?! as\b)", re.I)
dx = [(pid, v, q) for pid, v, q in queries if DEIXIS.search(q)]
print(f"deictic words {len(dx)}/{len(queries)} = {len(dx)/len(queries):.1%} (threshold <= 3%)")
for pid, v, q in dx:
    print(f"   {pid} {v}: {q}")

NUMW = {"two", "three", "four", "five", "six", "seven", "eight", "nine", "ten",
        "eleven", "twelve", "hundred", "thousand", "million", "billion"}


def numtoks(s):
    return {t for t in TOK.findall(s.lower()) if any(c.isdigit() for c in t) or t in NUMW}


nm = [(pid, v, q) for pid, v, q in queries if numtoks(q) & numtoks(pas[pid]["text"])]
print(f"numbers also in source passage {len(nm)}/{len(queries)} = {len(nm)/len(queries):.1%} (report-only; list read by hand)")
for pid, v, q in nm:
    print(f"   {pid} {v}: {q}")

COMP = re.compile(r"\b(and|while)\s+(how|why|what|is|are|does|do|can|might|would)\b", re.I)
cp = [(pid, v, q) for pid, v, q in queries if COMP.search(q)]
print(f"compound clause {len(cp)}/{len(queries)} = {len(cp)/len(queries):.1%} (threshold <= 5%)")
for pid, v, q in cp:
    print(f"   {pid} {v}: {q}")

print("unusable reasons:")
for r in parsed:
    if not r["parsed"]["usable"]:
        print("  ", r["passage_id"], "|", r["parsed"]["reason"][:200])
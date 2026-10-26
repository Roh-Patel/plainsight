import json, re
import ingest.chunk as C

STOP = set("""the of and a an in to is are was were be been this that these those for with
on at by from as it its we our which can has have had not or but also than then their there
such may more most into between using used use""".split())
WORD = re.compile(r"[A-Za-z']+")
LONG = re.compile(r"[A-Za-z]{25,}")


def feats(text):
    n = max(1, len(text.split()))
    stop = sum(t.lower() in STOP for t in WORD.findall(text)) / n
    return stop, len(LONG.findall(text)), text.count(",") / n


def pct(v):
    v = sorted(v)
    return {q: round(v[min(len(v) - 1, int(len(v) * q / 100))], 3) for q in (1, 5, 25, 50, 95)}


orig = C.junk_reason
C.junk_reason = lambda t: None  # keep everything so dropped chunks have full text

tk = C.BgeTokenizer()
titles = {}
with open(C.CANDIDATES, encoding="utf-8") as f:
    for line in f:
        r = json.loads(line)
        titles[r["paper_id"]] = " ".join(r["title"].split())

rows = []
with open(C.PARSED, encoding="utf-8") as f:
    for line in f:
        row = json.loads(line)
        ps, _ = C.process_paper(row["paper_id"], titles[row["paper_id"]], row["units"], tk)
        for p in ps:
            t = p["text"]
            s, lr, cm = feats(t)
            rows.append({"id": p["passage_id"], "cur": orig(t) or "kept", "stop": s,
                         "long": lr, "comma": cm,
                         "head": t[:110].replace("\n", " "),
                         "mid": t[len(t) // 2:len(t) // 2 + 110].replace("\n", " ")})

groups = {}
for r in rows:
    groups.setdefault(r["cur"], []).append(r)
print("passages:", len(rows), {k: len(v) for k, v in groups.items()})

print("\n=== function-word fraction by current label ===")
for k, v in groups.items():
    print(f"{k:14s}", pct([r["stop"] for r in v]))

kept = groups["kept"]
print("\nkept with stop < 0.10 / 0.15 / 0.20:",
      [sum(r["stop"] < t for r in kept) for t in (0.10, 0.15, 0.20)])
print("kept with a >=25-letter run:", sum(r["long"] > 0 for r in kept))
for k in ("table_residue", "comma_density", "missing_spaces"):
    print(f"{k} dropped with stop >= 0.25 (prose-like):", sum(r["stop"] >= 0.25 for r in groups[k]))

print("\n=== KEPT: 20 lowest function-word fraction ===")
for r in sorted(kept, key=lambda r: r["stop"])[:20]:
    print(f"[{r['stop']:.2f}] {r['id']} | {r['head']!r}")

print("\n=== DROPPED but prose-like: 15 highest function-word fraction ===")
dropped = [r for r in rows if r["cur"] != "kept"]
for r in sorted(dropped, key=lambda r: -r["stop"])[:15]:
    print(f"[{r['stop']:.2f}] {r['cur']} {r['id']} comma={r['comma']:.2f}")
    print(f"     {r['head']!r}")
    print(f"     ...{r['mid']!r}")

print("\n=== KEPT with a >=25-letter run (10 examples) ===")
for r in [r for r in kept if r["long"] > 0][:10]:
    print(f"[{r['long']}] {r['id']} | {r['head']!r}")
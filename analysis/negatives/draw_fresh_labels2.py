import json, os, random

OUT = "scratch_synth"
PATHS = {k: os.path.join(OUT, "fresh_review2." + k) for k in ("txt", "tsv", "json")}
if any(os.path.exists(p) for p in PATHS.values()):
    raise SystemExit("fresh_review2.* already exists, refusing to overwrite")

with open(os.path.join(OUT, "fresh_review.json"), encoding="utf-8") as f:
    used = {v["qid"] for v in json.load(f).values()}

qinfo = {}
with open("data/synthetic/queries.jsonl", encoding="utf-8") as f:
    for line in f:
        r = json.loads(line)
        if r["kept"]:
            qinfo[r["qid"]] = (r["voice"], r["query"])

pool = {v: [] for v in ("layperson", "confused_undergrad", "half_remembered_term")}
with open("data/synthetic/hn_scored.jsonl", encoding="utf-8") as f:
    for line in f:
        r = json.loads(line)
        if len(r["cands"]) >= 4 and r["qid"] not in used:
            pool[qinfo[r["qid"]][0]].append(r)

rng = random.Random(27)
picked = []
for voice, rows in pool.items():
    for r in rng.sample(rows, 50):
        top4 = sorted(r["cands"], key=lambda c: -c[2])[:4]
        picked.append((r["qid"], voice, top4[rng.randrange(4)]))
rng.shuffle(picked)
assert not ({q for q, _, _ in picked} & used)

need = {c[0] for _, _, c in picked}
text = {}
with open("data/processed/passages.jsonl", encoding="utf-8") as f:
    for i, line in enumerate(f):
        if i in need:
            r = json.loads(line)
            text[i] = (r["passage_id"], r["text"])

key = {}
with open(PATHS["txt"], "w", encoding="utf-8", newline="\n") as ft, \
     open(PATHS["tsv"], "w", encoding="utf-8", newline="\n") as fs:
    ft.write("Label B for every item in fresh_review2.tsv: 1 if the passage answers the query "
             "(same definition as the D045 B label), 0 if not.\n\n")
    fs.write("item\tB\n")
    for n, (qid, voice, c) in enumerate(picked, 1):
        item = "%03d" % n
        pid, body = text[c[0]]
        ft.write("=== %s ===\nQUERY: %s\nPASSAGE: %s\n\n" % (item, qinfo[qid][1], body))
        fs.write(item + "\t\n")
        key[item] = {"qid": qid, "voice": voice, "kind": "neg", "point_id": c[0],
                     "passage_id": pid, "baseline_rank": c[1], "ce_score": c[2], "item": item}
with open(PATHS["json"], "w", encoding="utf-8", newline="\n") as f:
    json.dump(key, f, indent=1)

print("items", len(picked), {v: sum(1 for _, x, _ in picked if x == v) for v in pool})
print("candidates at ce > -6:", sum(1 for _, _, c in picked if c[2] > -6))
print("qids shared with set 1:", len({q for q, _, _ in picked} & used))
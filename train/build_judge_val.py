import csv, json, os, random, collections

D = "scratch_synth"
OUT = os.path.join(D, "judge_val_items.jsonl")
if os.path.exists(OUT):
    raise SystemExit(OUT + " exists, refusing to overwrite")

def grades(tsv):
    with open(os.path.join(D, tsv), encoding="utf-8") as f:
        rows = list(csv.reader(f, delimiter="\t"))[1:]
    return {r[0].strip().zfill(3): r[1].strip() for r in rows if len(r) > 1}

pid2idx = {}
with open("data/processed/passages.jsonl", encoding="utf-8") as f:
    for i, line in enumerate(f):
        pid2idx[json.loads(line)["passage_id"]] = i

items = []  # (source, qid, voice, point_id, B)
kinds = collections.Counter()
for src, tsv, keyf in (("fresh1", "fresh_review.tsv", "fresh_review.json"),
                       ("fresh2", "fresh_review2.tsv", "fresh_review2.json"),
                       ("round3", "neg_review.tsv", "neg_review_key.json")):
    g = grades(tsv)
    with open(os.path.join(D, keyf), encoding="utf-8") as f:
        key = json.load(f)
    for it, k in sorted(key.items()):
        kinds[(src, k["kind"])] += 1
        if k["kind"] != "neg":
            continue
        b = g.get(it, "")
        assert b in ("0", "1"), (src, it, b)
        pidx = k["point_id"] if "point_id" in k else pid2idx[k["passage_id"]]
        items.append((src, k["qid"], k["voice"], pidx, int(b)))

g = grades("rank_review.tsv") if False else None
with open(os.path.join(D, "rank_review.tsv"), encoding="utf-8") as f:
    rr = {r[0]: r[3].strip() for r in list(csv.reader(f, delimiter="\t"))[1:] if len(r) > 3}
with open(os.path.join(D, "rank_review_key.json"), encoding="utf-8") as f:
    rkey = json.load(f)
for qid, k in sorted(rkey.items()):
    if rr.get(qid) in ("0", "1"):
        items.append(("d045", qid, k["voice"], k["competitor_point_id"], int(rr[qid])))

need = {q for _, q, _, _, _ in items}
cands = {}
with open("data/synthetic/hn_scored.jsonl", encoding="utf-8") as f:
    for line in f:
        r = json.loads(line)
        if r["qid"] in need:
            cands[r["qid"]] = sorted(r["cands"], key=lambda c: -c[2])

rng = random.Random(25)
rows, skipped = [], collections.Counter()
for src, qid, voice, pidx, b in items:
    comp = [c[0] for c in cands.get(qid, []) if c[0] != pidx][:3]
    if not comp:
        skipped[src] += 1
        continue
    order = comp[:]
    pos = rng.randrange(len(order) + 1)
    order.insert(pos, pidx)
    rows.append({"source": src, "qid": qid, "voice": voice, "item_point_id": pidx,
                 "owner_B": b, "position": pos, "point_ids": order})

with open(OUT, "w", encoding="utf-8", newline="\n") as f:
    for r in rows:
        f.write(json.dumps(r) + "\n")

print("kinds", dict(kinds))
print("skipped (no companions)", dict(skipped))
print("source | n | B=1 | B=0 | n_companions<3")
for s in ("fresh1", "fresh2", "round3", "d045"):
    sub = [r for r in rows if r["source"] == s]
    print(s, "|", len(sub), "|", sum(r["owner_B"] for r in sub), "|",
          sum(1 - r["owner_B"] for r in sub), "|", sum(1 for r in sub if len(r["point_ids"]) < 4))
fr = [r for r in rows if r["source"].startswith("fresh")]
print("fresh combined |", len(fr), "|", sum(r["owner_B"] for r in fr), "|", sum(1 - r["owner_B"] for r in fr))
print("position counts", dict(collections.Counter(r["position"] for r in rows)))
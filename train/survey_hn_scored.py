import json

T = -6.0
KS = (2, 3, 4, 5)

voice = {}
with open("data/synthetic/queries.jsonl", encoding="utf-8") as f:
    for line in f:
        r = json.loads(line)
        if r["kept"]:
            voice[r["qid"]] = r["voice"]

agg = {}
n_rows = 0
with open("data/synthetic/hn_scored.jsonl", encoding="utf-8") as f:
    for line in f:
        r = json.loads(line)
        n_rows += 1
        for key in (voice[r["qid"]], "ALL"):
            a = agg.setdefault(key, {"n": 0, "cands": 0, "above": 0,
                                     "no_above": 0, "gt_pos": 0,
                                     "pairs": {k: 0 for k in KS}})
            ces = [c[2] for c in r["cands"]]
            above = sum(1 for c in ces if c > T)
            a["n"] += 1
            a["cands"] += len(ces)
            a["above"] += above
            a["no_above"] += (above == 0)
            a["gt_pos"] += bool(ces) and max(ces) > r["pos_ce_score"]
            for k in KS:
                a["pairs"][k] += min(k, above)

print("rows", n_rows, "kept queries", len(voice), "match", n_rows == len(voice))
print("voice | queries | cands/q | above -6 /q | no above | cand>pos | pairs K=" + ",".join(map(str, KS)))
for key, a in agg.items():
    n = a["n"]
    print(key, "|", n, "|", round(a["cands"] / n, 1), "|", round(a["above"] / n, 1), "|",
          round(a["no_above"] / n, 3), "|", round(a["gt_pos"] / n, 3), "|",
          ", ".join(str(a["pairs"][k]) for k in KS))
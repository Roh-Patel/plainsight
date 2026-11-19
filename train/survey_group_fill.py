import json, collections

voice = {}
with open("data/synthetic/queries.jsonl", encoding="utf-8") as f:
    for line in f:
        r = json.loads(line)
        if r["kept"]:
            voice[r["qid"]] = r["voice"]

agg = {}
with open("data/synthetic/hard_negatives.jsonl", encoding="utf-8") as f:
    for line in f:
        r = json.loads(line)
        n = len(r["negatives"])
        for key in (voice[r["qid"]], "ALL"):
            a = agg.setdefault(key, collections.Counter())
            a["q"] += 1
            a["neg"] += n
            a["zero"] += n == 0
            for t in (1, 3, 5, 7):
                a["ge%d" % t] += n >= t
            a["pairs7"] += 1 + min(n, 7)

print("voice | queries | neg total | mean neg/q | n=0 | n<3 | n<5 | n<7 | pairs/epoch at cap 7")
for key, a in agg.items():
    q = a["q"]
    print(key, "|", q, "|", a["neg"], "|", round(a["neg"] / q, 1), "|",
          round(a["zero"] / q, 3), "|", round(1 - a["ge3"] / q, 3), "|",
          round(1 - a["ge5"] / q, 3), "|", round(1 - a["ge7"] / q, 3), "|", a["pairs7"])
import json, os, sys, collections
import numpy as np
import anthropic

from train.judge_prompt import parse

D = "scratch_synth"
ITEMS = os.path.join(D, "judge_val_items.jsonl")
REC = os.path.join(D, "judge_val_batch.json")
RAW = os.path.join(D, "judge_val_raw.api.jsonl")


def collect():
    if os.path.exists(RAW):
        sys.exit(RAW + " exists, refusing to overwrite")
    with open(REC, encoding="utf-8") as f:
        bid = json.load(f)["batch_id"]
    n = 0
    with open(RAW, "w", encoding="utf-8", newline="\n") as f:
        for r in anthropic.Anthropic().messages.batches.results(bid):
            f.write(r.model_dump_json() + "\n")
            n += 1
    print("saved", n, "results to", RAW)


def ci(x, rng, B=10000):
    x = np.asarray(x, dtype=float)
    if len(x) == 0:
        return (float("nan"),) * 3
    m = x[rng.integers(0, len(x), (B, len(x)))].mean(axis=1)
    return x.mean(), np.percentile(m, 2.5), np.percentile(m, 97.5)


def score():
    with open(ITEMS, encoding="utf-8") as f:
        rows = [json.loads(line) for line in f]
    res = {}
    with open(RAW, encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            res[r["custom_id"]] = r["result"]
    stops, fails = collections.Counter(), collections.Counter()
    tok_in = tok_out = n_ok = 0
    for n, r in enumerate(rows):
        x = res.get("v%04d" % n)
        pred = None
        if x is None or x["type"] != "succeeded":
            stops["<" + (x["type"] if x else "missing") + ">"] += 1
        else:
            m = x["message"]
            stops[m["stop_reason"]] += 1
            tok_in += m["usage"]["input_tokens"]
            tok_out += m["usage"]["output_tokens"]
            n_ok += 1
            text = "".join(b.get("text", "") for b in m["content"] if b["type"] == "text")
            if m["stop_reason"] == "end_turn":
                p = parse(text, len(r["point_ids"]))
                if p is not None:
                    pred = p[r["position"]]
        if pred is None:
            fails[r["source"]] += 1
            pred = 1 - r["owner_B"]  # declared rule: failure counts as wrong
        r["pred"] = pred
        r["failed"] = int(x is None or pred is None)
    print("stop reasons", dict(stops), "| failures by source", dict(fails))
    if n_ok:
        print("mean tokens per call: input", round(tok_in / n_ok), "output", round(tok_out / n_ok))

    rng = np.random.default_rng(0)

    def report(name, sub):
        pos = [r["pred"] == 1 for r in sub if r["owner_B"] == 1]
        neg = [r["pred"] == 0 for r in sub if r["owner_B"] == 0]
        a, b, c = ci(pos, rng)
        d, e, f = ci(neg, rng)
        print("%-18s recall %.3f [%.3f, %.3f] (n=%d) | specificity %.3f [%.3f, %.3f] (n=%d)"
              % (name, a, b, c, len(pos), d, e, f, len(neg)))
        return a, d

    print("== bar: recall >= 0.85 and specificity >= 0.60, pooled and combined fresh ==")
    pooled = report("pooled", rows)
    fresh = report("fresh combined", [r for r in rows if r["source"].startswith("fresh")])
    print("PASS pooled", pooled[0] >= 0.85 and pooled[1] >= 0.60,
          "| PASS fresh", fresh[0] >= 0.85 and fresh[1] >= 0.60)
    print("== report only ==")
    for s in ("fresh1", "fresh2", "round3", "d045"):
        report(s, [r for r in rows if r["source"] == s])
    for v in ("layperson", "confused_undergrad", "half_remembered_term"):
        report("fresh " + v[:10], [r for r in rows if r["source"].startswith("fresh") and r["voice"] == v])
    for p in range(4):
        report("position %d" % p, [r for r in rows if r["position"] == p])
    flagged = [r["pred"] for r in rows if r["source"].startswith("fresh")]
    print("fresh items judged B=1: %.3f (owner %.3f)" % (np.mean(flagged),
          np.mean([r["owner_B"] for r in rows if r["source"].startswith("fresh")])))


if __name__ == "__main__":
    {"collect": collect, "score": score}[sys.argv[1]]()
"""Report-only form checks over data/synthetic/queries.jsonl.
Regexes and helpers are copied from analysis/queries/check_dev.py. Writes nothing but a sample file."""
import json
import random
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.stdout.reconfigure(errors="replace")

PASSAGES = Path("data/processed/passages.jsonl")
QUERIES = Path("data/synthetic/queries.jsonl")
RANKS = Path("data/synthetic/ranks/ranks.jsonl")
SAMPLES = Path("scratch_synth/form_samples.txt")
VOICES = ("layperson", "confused_undergrad", "half_remembered_term")
SEED = 16

# copied from analysis/queries/check_dev.py
BANNED = ["the paper", "the authors", "this study", "the passage", "the text",
          "the article", "the researchers", "the section", "the figure", "the table"]
DEIXIS = re.compile(r"\b(this|these)\b|\bsuch\b(?! as\b)", re.I)
COMP = re.compile(r"\b(and|while)\s+(how|why|what|is|are|does|do|can|might|would)\b", re.I)
TOK = re.compile(r"[a-z0-9]+(?:[-'][a-z0-9]+)*")
NUMW = {"two", "three", "four", "five", "six", "seven", "eight", "nine", "ten",
        "eleven", "twelve", "hundred", "thousand", "million", "billion"}

# new: numeric values with decimals kept, not embedded in a word (H2O) or a longer number
NUM = re.compile(r"(?<![A-Za-z\d.])\d+(?:\.\d+)?")

# T13 full-set figures from the handoff, for the drift check
EXPECTED = {"banned": 8, "no_qmark": 1, "length": 3, "compound": 3484,
            "deictic": 279, "old_numbers": 2897}


def ends_q(q):
    return q.strip().rstrip("\"')").endswith("?")


def numtoks_old(s):
    return {t for t in TOK.findall(s.lower()) if any(c.isdigit() for c in t) or t in NUMW}


def nums(s):
    return set(NUM.findall(s))


def single_digit(t):
    return len(t) == 1


def main():
    with open(QUERIES, encoding="utf-8") as f:
        rows = [json.loads(l) for l in f]
    need = {r["source_passage_id"] for r in rows}
    text = {}
    with open(PASSAGES, encoding="utf-8") as f:
        for line in f:
            p = json.loads(line)
            if p["passage_id"] in need:
                text[p["passage_id"]] = p["text"]
    with open(RANKS, encoding="utf-8") as f:
        ranks = [json.loads(l) for l in f]
    if len(ranks) != len(rows) or any(a["qid"] != b["qid"] for a, b in zip(rows, ranks)):
        raise SystemExit("queries.jsonl and ranks.jsonl are not aligned")
    n = len(rows)
    print(f"queries {n}, kept {sum(r['kept'] for r in rows)}")

    flags = {k: set() for k in ("banned", "no_qmark", "length", "compound", "deictic",
                                "old_numbers", "num_any", "num_multi")}
    has_num = set()
    for i, r in enumerate(rows):
        q, src = r["query"], text[r["source_passage_id"]]
        if any(b in q.lower() for b in BANNED):
            flags["banned"].add(i)
        if not ends_q(q):
            flags["no_qmark"].add(i)
        if not 6 <= len(q.split()) <= 40:
            flags["length"].add(i)
        if COMP.search(q):
            flags["compound"].add(i)
        if DEIXIS.search(q):
            flags["deictic"].add(i)
        if numtoks_old(q) & numtoks_old(src):
            flags["old_numbers"].add(i)
        qn = nums(q)
        if qn:
            has_num.add(i)
        shared = qn & nums(src)
        if shared:
            flags["num_any"].add(i)
        if any(not single_digit(t) for t in shared):
            flags["num_multi"].add(i)

    print("\ndrift check against the T13 full-set report (check_dev):")
    for k, exp in EXPECTED.items():
        got = len(flags[k])
        print(f"  {k:12s} expected {exp:5d}  got {got:5d}  {'OK' if got == exp else 'MISMATCH'}")

    voice_n = Counter(r["voice"] for r in rows)
    print("\ncount per rule and voice (share of that voice):")
    print("rule | " + " | ".join(VOICES) + " | max/min share")
    for k in flags:
        c = Counter(rows[i]["voice"] for i in flags[k])
        sh = [c[v] / voice_n[v] for v in VOICES]
        ratio = (max(sh) / min(sh)) if min(sh) > 0 else float("inf")
        print(f"{k} | " + " | ".join(f"{c[v]} ({s:.1%})" for v, s in zip(VOICES, sh)) + f" | {ratio:.2f}")
    print(f"queries containing any numeric token: {len(has_num)} ({len(has_num) / n:.1%})")

    drop_rules = ("banned", "no_qmark", "length", "compound")
    drop = set().union(*(flags[k] for k in drop_rules))
    hard = set().union(*(flags[k] for k in ("banned", "no_qmark", "length")))
    print(f"\nhard failures (banned, no '?', length): {len(hard)} queries")
    print(f"drop candidates (hard + compound): {len(drop)} of {n} ({len(drop) / n:.1%})")
    after = Counter(rows[i]["voice"] for i in range(n) if i not in drop)
    tot = sum(after.values())
    print("voice share of the data, before -> after: " +
          ", ".join(f"{v} {voice_n[v] / n:.1%} -> {after[v] / tot:.1%}" for v in VOICES))

    print("\npairwise overlap (rows: rule, columns: rule):")
    names = list(drop_rules) + ["deictic", "num_multi"]
    print("            " + " ".join(f"{m:>10s}" for m in names))
    for a in names:
        print(f"{a:>11s} " + " ".join(f"{len(flags[a] & flags[b]):10d}" for b in names))

    print("\nhybrid_base source rank, flagged vs unflagged, per voice (share with source rank <= 50):")
    for k in ("compound", "deictic", "num_multi"):
        for v in VOICES:
            ins = [i for i in range(n) if rows[i]["voice"] == v]
            f_ = [i for i in ins if i in flags[k]]
            u_ = [i for i in ins if i not in flags[k]]

            def s50(idx):
                return sum(ranks[i]["hybrid_base_src_rank"] is not None
                           and ranks[i]["hybrid_base_src_rank"] <= 50 for i in idx) / max(len(idx), 1)
            print(f"  {k:9s} {v:22s} flagged {s50(f_):.3f} (n={len(f_)})  unflagged {s50(u_):.3f}")

    rng = random.Random(SEED)
    SAMPLES.parent.mkdir(exist_ok=True)
    with open(SAMPLES, "w", encoding="utf-8", newline="\n") as f:
        f.write("== deictic hits, 10 per voice (read the query alone: does it need the passage?) ==\n")
        for v in VOICES:
            pool = sorted(i for i in flags["deictic"] if rows[i]["voice"] == v)
            for i in rng.sample(pool, min(10, len(pool))):
                m = DEIXIS.search(rows[i]["query"]).group(0)
                f.write(f"{rows[i]['qid']} {v} [{m}]: {rows[i]['query']}\n")
        f.write("\n== compound-clause hits, 15 random (is each really two questions?) ==\n")
        pool = sorted(flags["compound"])
        for i in rng.sample(pool, min(15, len(pool))):
            f.write(f"{rows[i]['qid']} {rows[i]['voice']}: {rows[i]['query']}\n")
        f.write("\n== shared non-single-digit number, 20 random ==\n")
        pool = sorted(flags["num_multi"])
        for i in rng.sample(pool, min(20, len(pool))):
            shared = sorted(t for t in nums(rows[i]["query"]) & nums(text[rows[i]["source_passage_id"]])
                            if not single_digit(t))
            f.write(f"{rows[i]['qid']} {rows[i]['voice']} {shared}: {rows[i]['query']}\n")
    print(f"\nsamples written to {SAMPLES}")


if __name__ == "__main__":
    main()
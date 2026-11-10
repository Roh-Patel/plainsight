"""Applying the form, duplicate-text and eval-overlap filters to queries.jsonl.
Dry run by default. --apply backs up the file, then sets kept and drop_reason."""
import argparse
import json
import os
import re
import shutil
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import torch

from retrieval.dense import DenseSearcher
from train.form_report import BANNED, COMP, DEIXIS, ends_q, nums, single_digit

sys.stdout.reconfigure(errors="replace")

PASSAGES = Path("data/processed/passages.jsonl")
QUERIES = Path("data/synthetic/queries.jsonl")
QUESTIONS = Path("data/eval/questions.jsonl")
BACKUP = Path("scratch_synth/queries_pre_t14.jsonl")
AUDIT = Path("data/synthetic/eval_overlap.jsonl")
EVAL_SIM = 0.85
VOICES = ("layperson", "confused_undergrad", "half_remembered_term")
ORDER = ("banned_phrase", "no_question_mark", "length", "compound_clause",
         "deictic", "shared_number", "duplicate_text", "eval_overlap")


def norm(s):
    return " ".join(re.sub(r"[^a-z0-9 ]", " ", s.lower()).split())


def is_year(t):
    return t.isdigit() and 1500 <= int(t) <= 2099


def form_reasons(q, src):
    out = []
    if any(b in q.lower() for b in BANNED):
        out.append("banned_phrase")
    if not ends_q(q):
        out.append("no_question_mark")
    if not 6 <= len(q.split()) <= 40:
        out.append("length")
    if COMP.search(q):
        out.append("compound_clause")
    if DEIXIS.search(q):
        out.append("deictic")
    if any(not single_digit(t) and not is_year(t) for t in nums(q) & nums(src)):
        out.append("shared_number")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--apply", action="store_true")
    args = ap.parse_args()

    with open(QUERIES, encoding="utf-8") as f:
        rows = [json.loads(l) for l in f]
    if any(r["kept"] is not True or r["drop_reason"] is not None for r in rows):
        raise SystemExit("queries.jsonl already has kept/drop_reason set; refusing to continue")
    with open(QUESTIONS, encoding="utf-8") as f:
        qs = [json.loads(l) for l in f]
    n = len(rows)

    need = {r["source_passage_id"] for r in rows}
    text = {}
    with open(PASSAGES, encoding="utf-8") as f:
        for line in f:
            p = json.loads(line)
            if p["passage_id"] in need:
                text[p["passage_id"]] = p["text"]

    reasons = [form_reasons(r["query"], text[r["source_passage_id"]]) for r in rows]

    seen = {}
    for i, r in enumerate(rows):
        if reasons[i]:
            continue
        k = norm(r["query"])
        if k in seen and rows[seen[k]]["source_passage_id"] != r["source_passage_id"]:
            reasons[i].append("duplicate_text")
        else:
            seen.setdefault(k, i)

    dense = DenseSearcher("dense_base", backend="numpy")
    emb = torch.from_numpy(dense.encode([r["query"] for r in rows], batch_size=128)).to("cuda")
    ev = torch.from_numpy(dense.encode([q["question"] for q in qs], batch_size=128)).to("cuda")
    sim = np.concatenate([(emb[i:i + 4096] @ ev.T).cpu().numpy() for i in range(0, n, 4096)])
    mx = sim.max(axis=1)
    for i in np.flatnonzero(mx >= EVAL_SIM):
        reasons[i].append("eval_overlap")

    voice = np.array([r["voice"] for r in rows])
    print(f"queries {n}")
    print("\nrule | independent count | per voice")
    for rule in ORDER:
        m = np.array([rule in rs for rs in reasons])
        print(f"{rule} | {int(m.sum())} | {[int((m & (voice == v)).sum()) for v in VOICES]}")
    first = Counter(rs[0] for rs in reasons if rs)
    print("\ndrop_reason that will be written (first rule that fires):")
    for rule in ORDER:
        print(f"  {rule}: {first.get(rule, 0)}")
    drop = np.array([bool(rs) for rs in reasons])
    print(f"\ndropped {int(drop.sum())} ({drop.mean():.1%}), kept {int((~drop).sum())}")
    before = Counter(voice)
    after = Counter(voice[~drop])
    tot = sum(after.values())
    print("voice share before -> after: " +
          ", ".join(f"{v} {before[v] / n:.1%} -> {after[v] / tot:.1%} (kept {after[v]})" for v in VOICES))

    cnt = (sim >= EVAL_SIM).sum(axis=0)
    print(f"\nsynthetic queries within {EVAL_SIM} of an eval question: {int((mx >= EVAL_SIM).sum())} "
          f"(>= 0.90: {int((mx >= 0.90).sum())}); per voice "
          f"{[int(((mx >= EVAL_SIM) & (voice == v)).sum()) for v in VOICES]}")
    print(f"eval questions with at least one such query: {int((cnt > 0).sum())} of {len(qs)}; "
          f"queries per exposed question: median {int(np.median(cnt[cnt > 0])) if (cnt > 0).any() else 0}, "
          f"max {int(cnt.max())}")
    print("eval questions by number of close queries, top 8: " +
          ", ".join(f"{qs[k]['qid']}={int(cnt[k])}" for k in np.argsort(-cnt)[:8]))

    if not args.apply:
        print("\ndry run: nothing written")
        return

    if BACKUP.exists():
        raise SystemExit(f"{BACKUP} exists; refusing to overwrite")
    BACKUP.parent.mkdir(exist_ok=True)
    shutil.copy2(QUERIES, BACKUP)
    tmp = QUERIES.with_name("queries.jsonl.tmp")
    with open(tmp, "w", encoding="utf-8", newline="\n") as f:
        for r, rs in zip(rows, reasons):
            r["kept"] = not rs
            r["drop_reason"] = rs[0] if rs else None
            f.write(json.dumps(r) + "\n")
    os.replace(tmp, QUERIES)

    nn = sim.argmax(axis=0)
    with open(AUDIT, "w", encoding="utf-8", newline="\n") as f:
        for k, q in enumerate(qs):
            f.write(json.dumps({"qid": q["qid"], "max_sim": round(float(sim[:, k].max()), 4),
                                "nearest_qid": rows[int(nn[k])]["qid"],
                                "n_ge_threshold": int(cnt[k]), "threshold": EVAL_SIM}) + "\n")
    with open(QUERIES, encoding="utf-8") as f:
        back = [json.loads(l) for l in f]
    print(f"\nwrote {QUERIES}: {len(back)} rows, kept {sum(r['kept'] for r in back)}; "
          f"backup {BACKUP}; audit {AUDIT}")


if __name__ == "__main__":
    main()
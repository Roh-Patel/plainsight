"""Validate the hand-written question draft and (with --final) freeze questions.jsonl."""
import argparse
import hashlib
import json
import re
import sys
from collections import Counter
from itertools import combinations
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TOPICS = ROOT / "data" / "eval" / "topics.txt"
OUT = ROOT / "data" / "eval" / "questions.jsonl"
STYLES = ("layperson", "confused_undergrad", "half_remembered_term")
TAGS = {"acronym", "numeric", "multi_paper"}
STOP = set("""a an the of to in on is are was were be been do does did how why what when
which who can could would should it its this that these those and or for with as at by
from about i my me we you they there than then so if not no into over under between""".split())
FLAG = re.compile(r"\b(paper|papers|study|studies|arxiv|et al|recent|recently|latest|newest|20\d\d)\b", re.I)


def words(q):
    return {w for w in re.findall(r"[a-z0-9]+", q.lower()) if w not in STOP}


def jac(a, b):
    return len(a & b) / len(a | b) if a | b else 0.0


def load_topics():
    topics = {}
    for line in TOPICS.read_text(encoding="utf-8").splitlines():
        if line.strip() and not line.startswith("#"):
            t, fam, n = (c.strip() for c in line.split("|"))
            topics[t] = (fam, int(n))
    return topics


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("draft")
    ap.add_argument("--final", action="store_true")
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()
    sys.stdout.reconfigure(errors="replace")

    topics = load_topics()
    errors, warns, rows = [], [], []
    for no, line in enumerate(Path(a.draft).read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip() or line.startswith("#"):
            continue
        parts = [p.strip() for p in line.split("|", 3)]
        if len(parts) != 4:
            errors.append(f"line {no}: expected 4 fields")
            continue
        topic, style, tags_raw, q = parts
        tags = [] if tags_raw in ("", "-") else [t.strip() for t in tags_raw.split(",")]
        if topic not in topics:
            errors.append(f"line {no}: unknown topic '{topic}'")
        if style not in STYLES:
            errors.append(f"line {no}: unknown style '{style}'")
        bad = [t for t in tags if t not in TAGS]
        if bad:
            errors.append(f"line {no}: unknown tags {bad}")
        n_words = len(q.split())
        if n_words < 4:
            errors.append(f"line {no}: question under 4 words")
        if n_words > 40:
            warns.append(f"line {no}: question over 40 words")
        m = FLAG.search(q)
        if m:
            warns.append(f"line {no}: contains '{m.group(0)}'")
        rows.append({"line": no, "topic": topic, "style": style, "tags": tags, "question": q})

    # duplicates
    ws = [words(r["question"]) for r in rows]
    for i, j in combinations(range(len(rows)), 2):
        s = jac(ws[i], ws[j])
        li, lj = rows[i]["line"], rows[j]["line"]
        if s >= 0.6:
            errors.append(f"lines {li} and {lj}: near-duplicate (jaccard {s:.2f})")
        elif s >= 0.45 and rows[i]["topic"] == rows[j]["topic"]:
            warns.append(f"lines {li} and {lj}: similar within topic (jaccard {s:.2f})")

    # counts
    have = Counter(r["topic"] for r in rows)
    short = {t: (have[t], n) for t, (_, n) in topics.items() if have[t] != n}
    for t, (h, n) in short.items():
        msg = f"topic '{t}': {h} of {n}"
        (errors if (a.final or h > n) else warns).append(msg)

    total = len(rows)
    print(f"{total} questions, need {sum(n for _, n in topics.values())}")
    sc = Counter(r["style"] for r in rows)
    print("styles:", {s: sc[s] for s in STYLES})
    tc = Counter(t for r in rows for t in r["tags"])
    print("tags:", {t: tc[t] for t in sorted(TAGS)})
    if total:
        for s in STYLES:
            if a.final and not 0.25 <= sc[s] / total <= 0.40:
                warns.append(f"style {s} is {sc[s] / total:.0%} of questions")
    if a.final:
        for t in sorted(TAGS):
            if tc[t] < 8:
                warns.append(f"tag {t} has {tc[t]} questions (target 8)")

    for w in warns:
        print("WARN ", w)
    for e in errors:
        print("ERROR", e)
    if errors:
        print(f"{len(errors)} errors, {len(warns)} warnings")
        sys.exit(1)
    print(f"0 errors, {len(warns)} warnings")

    if a.final:
        if OUT.exists() and not a.force:
            sys.exit(f"{OUT} exists; refusing to overwrite without --force")
        lines = []
        for i, r in enumerate(rows, 1):
            obj = {"qid": f"eval-{i:04d}", "question": r["question"], "topic": r["topic"],
                   "style": r["style"], "tags": r["tags"]}
            lines.append(json.dumps(obj, ensure_ascii=False))
        data = ("\n".join(lines) + "\n").encode("utf-8")
        OUT.write_bytes(data)
        print(f"wrote {OUT} ({len(lines)} rows)")
        print("sha256", hashlib.sha256(data).hexdigest())


if __name__ == "__main__":
    main()
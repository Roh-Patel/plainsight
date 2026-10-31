"""Print 5 random paper titles per named topic (titles only, no passage text)."""
import json
import random
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CAND = ROOT / "data" / "processed" / "candidates.jsonl"
SPLITS = ROOT / "data" / "processed" / "splits.json"


def main(topics_path, names):
    sys.stdout.reconfigure(errors="replace")
    splits = json.loads(SPLITS.read_text(encoding="utf-8"))
    in_split = {p for s in ("train", "val", "test") for p in splits[s]}
    papers = []
    with CAND.open(encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            if r["paper_id"] in in_split:
                papers.append((r["paper_id"], r["title"],
                               (r["title"] + " " + r["abstract"]).lower()))
    wanted = {n.lower() for n in names}
    rng = random.Random(0)
    for line in Path(topics_path).read_text(encoding="utf-8").splitlines():
        if not line.strip() or line.startswith("#"):
            continue
        topic, kws = (x.strip() for x in line.split("|", 1))
        if topic.lower() not in wanted:
            continue
        # Keep this compile line identical to eval/topic_coverage.py
        pats = [re.compile(r"\b" + k.strip(), re.IGNORECASE) for k in kws.split(",")]
        hits = [(pid, title) for pid, title, txt in papers
                if any(p.search(txt) for p in pats)]
        print(f"\n== {topic} ({len(hits)} papers)")
        for pid, title in rng.sample(hits, min(5, len(hits))):
            print(f"  {pid}  {title}")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2:])
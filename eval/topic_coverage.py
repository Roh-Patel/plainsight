"""Count papers per split for each draft topic (metadata only, no passage text)."""
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CAND = ROOT / "data" / "processed" / "candidates.jsonl"
SPLITS = ROOT / "data" / "processed" / "splits.json"


def main(topics_path):
    splits = json.loads(SPLITS.read_text(encoding="utf-8"))
    split_of = {p: s for s in ("train", "val", "test") for p in splits[s]}
    texts = {}
    with CAND.open(encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            if r["paper_id"] in split_of:
                texts[r["paper_id"]] = (r["title"] + " " + r["abstract"]).lower()
    print(f"{len(texts)} papers with a split")
    print(f"{'topic':36}{'train':>6}{'val':>5}{'test':>6}{'test%':>7}")
    for line in Path(topics_path).read_text(encoding="utf-8").splitlines():
        if not line.strip() or line.startswith("#"):
            continue
        topic, kws = (x.strip() for x in line.split("|", 1))
        pats = [re.compile(r"\b" + k.strip(), re.IGNORECASE) for k in kws.split(",")]
        c = {"train": 0, "val": 0, "test": 0}
        for pid, txt in texts.items():
            if any(p.search(txt) for p in pats):
                c[split_of[pid]] += 1
        n = sum(c.values())
        share = c["test"] / n if n else 0.0
        print(f"{topic:36}{c['train']:6}{c['val']:5}{c['test']:6}{share:7.2f}")


if __name__ == "__main__":
    main(sys.argv[1])
"""Stream the arXiv snapshot, keep astro-ph.EP papers whose v1 is 2024-01-01 or later."""
import json
from collections import Counter
from email.utils import parsedate_to_datetime
from pathlib import Path

SNAPSHOT = Path("data/raw/arxiv-metadata-oai-snapshot.json")
OUT = Path("data/processed/candidates.jsonl")
CATEGORY = "astro-ph.EP"
MIN_DATE = "2024-01-01"


def clean(s: str) -> str:
    return " ".join(s.split())


def author_names(parsed) -> list[str]:
    names = []
    for a in parsed:
        last, first, suffix = (list(a) + ["", "", ""])[:3]
        names.append(" ".join(p for p in (first, last, suffix) if p))
    return names


def main() -> None:
    OUT.parent.mkdir(parents=True, exist_ok=True)
    scanned = ep_total = bad = 0
    by_year = Counter()
    kept = []

    with SNAPSHOT.open(encoding="utf-8") as f:
        for line in f:
            scanned += 1
            rec = json.loads(line)
            cats = rec["categories"].split()
            if CATEGORY not in cats:
                continue
            ep_total += 1
            try:
                created = rec["versions"][0]["created"]
                published = parsedate_to_datetime(created).date().isoformat()
            except (KeyError, IndexError, TypeError, ValueError):
                bad += 1
                continue
            by_year[published[:4]] += 1
            if published < MIN_DATE:
                continue
            kept.append({
                "paper_id": rec["id"],
                "title": clean(rec["title"]),
                "authors": author_names(rec.get("authors_parsed", [])),
                "published": published,
                "arxiv_categories": cats,
                "abstract": clean(rec["abstract"]),
            })

    with OUT.open("w", encoding="utf-8", newline="\n") as f:
        for row in kept:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"scanned: {scanned}")
    print(f"astro-ph.EP (any date): {ep_total}")
    print(f"unparseable v1 date: {bad}")
    print(f"kept (>= {MIN_DATE}): {len(kept)}")
    for y in sorted(by_year):
        if y >= "2022":
            print(f"  {y}: {by_year[y]}")


if __name__ == "__main__":
    main()

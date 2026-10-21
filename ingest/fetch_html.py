"""T03: fetch arXiv HTML for candidates in seeded random order. Resumable, 3 s between requests."""
import argparse
import json
import os
import random
import time
from datetime import datetime
from pathlib import Path

import requests

CANDIDATES = Path("data/processed/candidates.jsonl")
HTML_DIR = Path("data/raw/html")
LOG = Path("data/raw/fetch_log.jsonl")
SEED = 0
DELAY = 3.0
FINAL = {"ok", "no_html", "bad_html"}


def load_log() -> dict:
    status = {}
    if LOG.exists():
        with LOG.open(encoding="utf-8") as f:
            for line in f:
                row = json.loads(line)
                status[row["paper_id"]] = row["status"]
    return status


def retry_after(r) -> int:
    try:
        return int(r.headers.get("Retry-After", 60))
    except ValueError:
        return 60


def is_article(content: bytes) -> bool:
    return content.count(b"ltx_para") >= 3


def fetch(session, paper_id):
    url = f"https://arxiv.org/html/{paper_id}"
    waits = [10, 30, 60]
    rate_limited = False
    detail = ""
    for attempt in range(len(waits) + 1):
        try:
            r = session.get(url, timeout=30)
        except requests.RequestException as e:
            detail = type(e).__name__
        else:
            code = r.status_code
            if code == 200:
                if is_article(r.content):
                    return "ok", r.content, code
                return "bad_html", None, code
            if code == 404:
                return "no_html", None, code
            if code == 429:
                if rate_limited:
                    raise SystemExit("Rate limited twice in a row. Stop and investigate.")
                rate_limited = True
                time.sleep(retry_after(r))
                continue
            detail = f"http {code}"
        if attempt < len(waits):
            time.sleep(waits[attempt])
    return "error", None, detail


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", type=int, default=3000, help="total ok papers to reach")
    ap.add_argument("--limit", type=int, default=None, help="max requests this run")
    args = ap.parse_args()

    contact = os.environ.get("CONTACT_EMAIL")
    if not contact:
        raise SystemExit("Run: set CONTACT_EMAIL=you@example.com")

    with CANDIDATES.open(encoding="utf-8") as f:
        ids = sorted(json.loads(line)["paper_id"] for line in f)
    random.Random(SEED).shuffle(ids)

    HTML_DIR.mkdir(parents=True, exist_ok=True)
    status = load_log()
    n_ok = sum(1 for s in status.values() if s == "ok")
    session = requests.Session()
    session.headers["User-Agent"] = f"plainsight-research (contact: {contact})"
    n_req = 0

    with LOG.open("a", encoding="utf-8", newline="\n") as log:
        for pid in ids:
            if n_ok >= args.target or (args.limit is not None and n_req >= args.limit):
                break
            if status.get(pid) in FINAL:
                continue
            st, content, code = fetch(session, pid)
            n_req += 1
            if st == "ok":
                path = HTML_DIR / f"{pid}.html"
                tmp = path.with_suffix(".tmp")
                tmp.write_bytes(content)
                tmp.replace(path)
                n_ok += 1
            status[pid] = st
            log.write(json.dumps({
                "paper_id": pid, "status": st, "http": code,
                "bytes": len(content) if content else 0,
                "ts": datetime.now().isoformat(timespec="seconds"),
            }) + "\n")
            log.flush()
            if n_req % 25 == 0:
                print(f"requests this run: {n_req}, ok total: {n_ok}", flush=True)
            time.sleep(DELAY)

    counts = {}
    for s in status.values():
        counts[s] = counts.get(s, 0) + 1
    decided = sum(counts.get(k, 0) for k in FINAL)
    print(f"status counts: {counts}")
    if decided:
        print(f"HTML success rate: {counts.get('ok', 0) / decided:.1%} of {decided} decided")


if __name__ == "__main__":
    main()

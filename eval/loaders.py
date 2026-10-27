"""Loaders for passages, qrels and runs. Every loader validates strictly."""
from __future__ import annotations

import json
from collections import defaultdict

GRADES = {"0", "1", "2"}


def load_passage_ids(path) -> set[str]:
    ids = set()
    with open(path, encoding="utf-8") as f:
        for line in f:
            ids.add(json.loads(line)["passage_id"])
    return ids


def load_qrels(path, valid_ids: set[str]) -> dict[str, dict[str, int]]:
    """TREC qrels, tab-separated, no header: qid 0 passage_id grade."""
    qrels: dict[str, dict[str, int]] = defaultdict(dict)
    with open(path, encoding="utf-8") as f:
        for n, line in enumerate(f, 1):
            line = line.rstrip("\r\n")
            if not line:
                continue
            parts = line.split("\t")
            if len(parts) != 4:
                raise ValueError(f"{path}:{n}: expected 4 columns, got {len(parts)}")
            qid, zero, pid, grade = parts
            if zero != "0":
                raise ValueError(f"{path}:{n}: second column must be 0, got {zero!r}")
            if pid not in valid_ids:
                raise ValueError(f"{path}:{n}: unknown passage_id {pid}")
            if grade not in GRADES:
                raise ValueError(f"{path}:{n}: bad grade {grade!r}")
            if pid in qrels[qid]:
                raise ValueError(f"{path}:{n}: duplicate judgment for {qid} {pid}")
            qrels[qid][pid] = int(grade)
    return dict(qrels)


def load_run(path, valid_ids: set[str], system: str | None = None) -> dict[str, list[str]]:
    """runs/{system}.jsonl -> qid -> passage_ids, best first."""
    run: dict[str, list[str]] = {}
    with open(path, encoding="utf-8") as f:
        for n, line in enumerate(f, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            qid = row["qid"]
            if qid in run:
                raise ValueError(f"{path}:{n}: duplicate qid {qid}")
            if system is not None and row["system"] != system:
                raise ValueError(f"{path}:{n}: system {row['system']!r}, expected {system!r}")
            ids = [r["passage_id"] for r in row["results"]]
            if len(set(ids)) != len(ids):
                raise ValueError(f"{path}:{n}: duplicate passage_id in results for {qid}")
            bad = [p for p in ids if p not in valid_ids]
            if bad:
                raise ValueError(f"{path}:{n}: unknown passage_id {bad[0]} ({len(bad)} total)")
            run[qid] = ids
    return run
"""Shared paths, passages loading and index-manifest verification."""
import hashlib
import json
import sys
from pathlib import Path

PASSAGES = Path("data/processed/passages.jsonl")
INDEX_DIR = Path("data/processed/index")
MANIFESTS = [
    INDEX_DIR / "dense_base.manifest.json",
    INDEX_DIR / "bm25" / "manifest.json",
    INDEX_DIR / "qdrant_passages_dense_base.manifest.json",
]


class IndexMismatch(RuntimeError):
    pass


_fp_cache = {}


def fingerprint(path=PASSAGES):
    """Return (line count, sha256 hex of the raw bytes). Cached per file state."""
    path = Path(path)
    st = path.stat()
    key = (str(path.resolve()), st.st_mtime_ns, st.st_size)
    if key not in _fp_cache:
        h = hashlib.sha256()
        n = 0
        with open(path, "rb") as f:
            for line in f:
                h.update(line)
                n += 1
        _fp_cache[key] = (n, h.hexdigest())
    return _fp_cache[key]


def check_manifest(manifest_path, passages=PASSAGES):
    """Load a manifest and refuse it if it does not match passages.jsonl."""
    manifest_path = Path(manifest_path)
    m = json.loads(manifest_path.read_text(encoding="utf-8"))
    if m.get("limit"):
        raise IndexMismatch(f"{manifest_path}: built with --limit {m['limit']}")
    n, digest = fingerprint(passages)
    if m.get("passages_lines") != n or m.get("passages_sha256") != digest:
        raise IndexMismatch(
            f"{manifest_path}: passages.jsonl changed since this index was built "
            f"(manifest {m.get('passages_lines')} lines, file {n} lines). Rebuild the index."
        )
    return m


def load_passages(fields=("passage_id", "paper_id"), path=PASSAGES):
    """List of dicts with only the requested fields. Index i is the point ID."""
    out = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            out.append({k: r[k] for k in fields})
    return out


if __name__ == "__main__":
    for p in MANIFESTS:
        try:
            check_manifest(p)
            print("ok  ", p)
        except (IndexMismatch, FileNotFoundError) as e:
            print("FAIL", p, e)
            sys.exit(1)
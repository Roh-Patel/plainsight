import json

import pytest

from retrieval.common import IndexMismatch, check_manifest, fingerprint, load_passages


def _write(path, ids):
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        for i in ids:
            f.write(json.dumps({"passage_id": f"p{i}", "paper_id": "x", "text": "t"}) + "\n")


def _manifest(path, passages, **over):
    n, digest = fingerprint(passages)
    m = {"passages_lines": n, "passages_sha256": digest, "limit": 0}
    m.update(over)
    path.write_text(json.dumps(m), encoding="utf-8")


def test_matching_manifest_passes(tmp_path):
    p = tmp_path / "passages.jsonl"
    _write(p, range(5))
    mp = tmp_path / "m.json"
    _manifest(mp, p)
    assert check_manifest(mp, p)["passages_lines"] == 5


def test_changed_passages_rejected(tmp_path):
    p = tmp_path / "passages.jsonl"
    _write(p, range(5))
    mp = tmp_path / "m.json"
    _manifest(mp, p)
    _write(p, range(6))
    with pytest.raises(IndexMismatch):
        check_manifest(mp, p)


def test_same_length_different_content_rejected(tmp_path):
    p = tmp_path / "passages.jsonl"
    _write(p, range(5))
    mp = tmp_path / "m.json"
    _manifest(mp, p)
    _write(p, [9, 1, 2, 3, 4])
    with pytest.raises(IndexMismatch):
        check_manifest(mp, p)


def test_limit_manifest_rejected(tmp_path):
    p = tmp_path / "passages.jsonl"
    _write(p, range(3))
    mp = tmp_path / "m.json"
    _manifest(mp, p, limit=50)
    with pytest.raises(IndexMismatch):
        check_manifest(mp, p)


def test_load_passages_order_and_fields(tmp_path):
    p = tmp_path / "passages.jsonl"
    _write(p, range(3))
    out = load_passages(path=p)
    assert [r["passage_id"] for r in out] == ["p0", "p1", "p2"]
    assert set(out[0]) == {"passage_id", "paper_id"}
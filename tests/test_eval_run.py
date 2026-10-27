import json

import pytest

from eval.evaluate import evaluate_run
from eval.loaders import load_passage_ids, load_qrels, load_run

IDS = {"P1::000::000", "P1::000::001", "P2::001::000"}


def write(p, text):
    p.write_text(text, encoding="utf-8", newline="\n")
    return p


def run_row(qid, ids, system="s"):
    return json.dumps({"qid": qid, "system": system,
                       "results": [{"passage_id": p, "score": 1.0} for p in ids]}) + "\n"


def test_passage_ids(tmp_path):
    p = write(tmp_path / "p.jsonl", '{"passage_id": "a"}\n{"passage_id": "b"}\n')
    assert load_passage_ids(p) == {"a", "b"}


def test_qrels_ok(tmp_path):
    p = write(tmp_path / "q.tsv", "q1\t0\tP1::000::000\t2\nq1\t0\tP2::001::000\t0\n")
    assert load_qrels(p, IDS) == {"q1": {"P1::000::000": 2, "P2::001::000": 0}}


@pytest.mark.parametrize("text,msg", [
    ("q1\t0\tZZ::000::000\t2\n", "unknown passage"),
    ("q1\t0\tP1::000::000\t3\n", "bad grade"),
    ("q1\t0\tP1::000::000\t2\nq1\t0\tP1::000::000\t1\n", "duplicate judgment"),
    ("q1 0 P1::000::000 2\n", "4 columns"),
])
def test_qrels_rejects(tmp_path, text, msg):
    with pytest.raises(ValueError, match=msg):
        load_qrels(write(tmp_path / "q.tsv", text), IDS)


def test_run_ok(tmp_path):
    p = write(tmp_path / "r.jsonl", run_row("q1", ["P2::001::000", "P1::000::000"]))
    assert load_run(p, IDS, "s") == {"q1": ["P2::001::000", "P1::000::000"]}


@pytest.mark.parametrize("text,system,msg", [
    (run_row("q1", ["ZZ::000::000"]), "s", "unknown passage"),
    (run_row("q1", ["P1::000::000", "P1::000::000"]), "s", "duplicate passage_id"),
    (run_row("q1", ["P1::000::000"]), "other", "expected 'other'"),
    (run_row("q1", ["P1::000::000"]) * 2, "s", "duplicate qid"),
])
def test_run_rejects(tmp_path, text, system, msg):
    with pytest.raises(ValueError, match=msg):
        load_run(write(tmp_path / "r.jsonl", text), IDS, system)


def test_evaluate_perfect():
    qrels = {"q1": {"P1::000::000": 2, "P1::000::001": 1}}
    run = {"q1": ["P1::000::000", "P1::000::001"]}
    m = evaluate_run(run, qrels)["q1"]
    assert m["ndcg@10"] == pytest.approx(1.0)
    assert m["recall@10"] == m["mrr@10"] == m["paper_hit@10"] == 1.0
    assert m["judged@10"] == pytest.approx(0.2)


def test_evaluate_qid_mismatch():
    qrels = {"q1": {"P1::000::000": 2}}
    with pytest.raises(ValueError, match="missing from run"):
        evaluate_run({}, qrels)
    with pytest.raises(ValueError, match="no qrels"):
        evaluate_run({"q1": [], "q2": []}, qrels)


def test_evaluate_no_relevant():
    with pytest.raises(ValueError, match="no passage with grade"):
        evaluate_run({"q1": ["P1::000::000"]}, {"q1": {"P1::000::000": 1}})
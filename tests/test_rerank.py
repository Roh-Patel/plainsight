from retrieval.rerank import Reranker


def make(texts):
    r = object.__new__(Reranker)
    r.texts = texts
    r.stats = {"pairs": 0, "passage_truncated": 0, "query_truncated": 0}
    r._cap_query = lambda q: q
    # stub scorer: score = passage length, so the expected order is known
    r.score_pairs = lambda qs, ps, bs=None, ct=True: [float(len(p)) for p in ps]
    return r


def test_reorders_each_query_separately():
    r = make(["a", "bbb", "cc", "dd"])
    c1 = [(0, 0.9), (1, 0.8), (2, 0.7)]
    c2 = [(3, 0.9), (0, 0.8)]
    out = r.rerank(["q1", "q2"], [c1, c2])
    assert out[0] == [(1, 3.0), (2, 2.0), (0, 1.0)]
    assert out[1] == [(3, 2.0), (0, 1.0)]


def test_depth_cuts_candidates():
    r = make(["a", "bb", "ccc"])
    cands = [(0, 0.3), (1, 0.2), (2, 0.1)]
    out = r.rerank(["q"], [cands], depth=2)[0]
    assert {p for p, _ in out} == {0, 1}


def test_ties_keep_fused_order():
    r = make(["xx", "yy", "zz"])
    cands = [(2, 0.3), (0, 0.2), (1, 0.1)]
    out = r.rerank(["q"], [cands])[0]
    assert [p for p, _ in out] == [2, 0, 1]


def test_empty_candidates():
    r = make(["a"])
    assert r.rerank(["q"], [[]]) == [[]]
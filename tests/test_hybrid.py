from retrieval.hybrid import HybridSearcher


class Fake:
    def __init__(self, lists):
        self.lists = lists
        self.k_seen = None

    def search(self, queries, k=100):
        self.k_seen = k
        return [[(p, 1.0) for p in lst] for lst in self.lists]


def test_fuses_both_lists():
    h = HybridSearcher(Fake([[1, 2, 3]]), Fake([[3, 4, 1]]))
    out = h.search(["q"])[0]
    assert [p for p, _ in out][:2] == [1, 3]
    assert {p for p, _ in out} == {1, 2, 3, 4}


def test_empty_bm25_falls_back_to_dense_order():
    h = HybridSearcher(Fake([[]]), Fake([[8, 5, 9]]))
    assert [p for p, _ in h.search(["q"])[0]] == [8, 5, 9]


def test_depth_and_top_n_are_applied():
    b, d = Fake([list(range(200))]), Fake([list(range(100, 300))])
    h = HybridSearcher(b, d, depth=100, top_n=100)
    out = h.search(["q"])[0]
    assert b.k_seen == 100 and d.k_seen == 100
    assert len(out) == 100


def test_one_result_list_per_query():
    h = HybridSearcher(Fake([[1], [2]]), Fake([[3], [4]]))
    assert len(h.search(["a", "b"])) == 2
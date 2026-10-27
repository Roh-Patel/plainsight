import pytest

from retrieval.fusion import rrf


def test_single_list_keeps_order():
    out = rrf([[7, 3, 9]])
    assert [pid for pid, _ in out] == [7, 3, 9]


def test_hand_computed_scores():
    out = dict(rrf([[1, 2], [2, 3]], k=60))
    assert out[2] == pytest.approx(1 / 62 + 1 / 61)
    assert out[1] == pytest.approx(1 / 61)
    assert out[3] == pytest.approx(1 / 62)
    assert [pid for pid, _ in rrf([[1, 2], [2, 3]])] == [2, 1, 3]


def test_agreement_beats_single_list_top():
    out = rrf([[10, 20], [30, 20]])
    assert out[0][0] == 20


def test_tie_breaks_by_lower_id():
    out = rrf([[5], [3]])
    assert out[0][1] == out[1][1]
    assert [pid for pid, _ in out] == [3, 5]


def test_top_n_truncates():
    out = rrf([list(range(200))], top_n=100)
    assert len(out) == 100
    assert out[0][0] == 0


def test_duplicate_in_one_ranking_raises():
    with pytest.raises(ValueError):
        rrf([[1, 1, 2]])


def test_empty_input():
    assert rrf([]) == []
    assert rrf([[], []]) == []
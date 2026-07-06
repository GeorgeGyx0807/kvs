from src.selectors import select_by_score, top_k


def test_select_by_score_respects_budget():
    chosen = select_by_score(
        scores={"a": 10.0, "b": 8.0, "c": 7.0},
        kv_bytes={"a": 5, "b": 4, "c": 3},
        budget_bytes=7,
        use_value_density=True,
    )
    assert chosen == ["c", "b"]


def test_top_k_truncates():
    assert top_k(["a", "b", "c"], 2) == ["a", "b"]

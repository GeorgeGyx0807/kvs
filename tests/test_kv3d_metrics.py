from src.kv3d.metrics import contains_answer, token_f1


def test_contains_answer_normalizes_case_and_punctuation():
    assert contains_answer("The answer is New York.", ("new york",)) == 1.0


def test_token_f1_uses_best_answer():
    assert token_f1("alpha beta", ("gamma", "alpha beta")) == 1.0

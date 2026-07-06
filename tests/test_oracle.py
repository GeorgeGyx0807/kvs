from src.oracle import score_oracle_label


def test_score_oracle_label_penalizes_costs():
    score = score_oracle_label(
        decode_attention_mass=10.0,
        kv_bytes=2.0,
        ttft_penalty=3.0,
        lambda_bytes=1.0,
        lambda_ttft=2.0,
    )
    assert score == 2.0

from src.ttft import decompose_ttft, estimate_communication_time


def test_decompose_ttft_sums_components():
    breakdown = decompose_ttft(
        kv_bytes=1000,
        bandwidth_bytes_per_sec=100,
        packaging_time=1.0,
        receiving_time=2.0,
        first_token_compute_time=3.0,
    )
    assert estimate_communication_time(1000, 100) == 10.0
    assert breakdown.total_ttft == 16.0


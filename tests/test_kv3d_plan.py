from src.kv3d.plan import filter_profiling_plan, generate_profiling_plan


def test_generate_profiling_plan_emits_removal_and_addition_specs():
    plan = generate_profiling_plan(
        sample_ids=["s1", "s2"],
        num_layers=2,
        num_heads=3,
        num_chunks=4,
        include_addition=True,
    )

    method_names = {item["method"] for item in plan}
    assert "full_kv" in method_names
    assert "b_only" in method_names
    assert "remove_layer" in method_names
    assert "remove_layer_head_chunk" in method_names
    assert "add_layer_head_chunk" in method_names
    assert len(plan) > 0


def test_generate_profiling_plan_rejects_non_positive_dimensions():
    try:
        generate_profiling_plan(sample_ids=["s1"], num_layers=0, num_heads=1, num_chunks=1)
    except ValueError as exc:
        assert "positive" in str(exc)
    else:
        raise AssertionError("expected ValueError")


def test_generate_profiling_plan_can_disable_chunk_level():
    plan = generate_profiling_plan(
        sample_ids=["s1"],
        num_layers=1,
        num_heads=1,
        num_chunks=2,
        include_chunk_level=False,
    )

    assert [item["method"] for item in plan] == ["full_kv", "b_only", "remove_layer", "remove_layer_head"]


def test_generate_profiling_plan_can_limit_chunk_targets():
    plan = generate_profiling_plan(
        sample_ids=["s1"],
        num_layers=2,
        num_heads=2,
        num_chunks=2,
        include_addition=True,
        chunk_targets={"s1": {(1, 0)}},
    )

    chunk_methods = [item for item in plan if item["method"] == "remove_layer_head_chunk"]
    assert chunk_methods
    assert {item["layer"] for item in chunk_methods} == {1}
    assert {item["head"] for item in chunk_methods} == {0}


def test_filter_profiling_plan_keeps_required_baselines_and_requested_methods():
    plan = generate_profiling_plan(
        sample_ids=["s1"],
        num_layers=1,
        num_heads=1,
        num_chunks=2,
        include_addition=True,
    )

    filtered = filter_profiling_plan(plan, methods={"remove_layer_head_chunk"})

    assert [item["method"] for item in filtered] == [
        "full_kv",
        "b_only",
        "remove_layer_head_chunk",
        "remove_layer_head_chunk",
    ]


def test_filter_profiling_plan_keeps_all_methods_when_filter_is_empty():
    plan = generate_profiling_plan(sample_ids=["s1"], num_layers=1, num_heads=1, num_chunks=1)

    assert filter_profiling_plan(plan, methods=set()) == plan

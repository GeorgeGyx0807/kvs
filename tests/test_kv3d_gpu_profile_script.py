import scripts.run_kv3d_gpu_profile as gpu_script


def test_gpu_profile_passes_include_addition_to_runner(monkeypatch, tmp_path):
    calls = {}

    class Args:
        model_name = "mock-model"
        device_map = "cpu"
        dtype = "float32"
        samples = tmp_path / "samples.json"
        dataset_name = ""
        config_name = ""
        split = "test"
        spec = None
        max_samples = 1
        sample_offset = 0
        chunk_size = 128
        max_context_tokens = 256
        max_new_tokens = 1
        max_layers = 1
        max_heads = 1
        num_chunks = 1
        include_addition = True
        include_chunk_level = True
        chunk_targets = None
        methods = ""
        output_dir = tmp_path / "out"

    Args.samples.write_text(
        '[{"sample_id":"s","dataset":"LongBench","task_name":"qa","prompt":"p","answers":["a"],"gold_answer":"a"}]'
    )

    monkeypatch.setattr(gpu_script, "parse_args", lambda: Args)
    monkeypatch.setattr(gpu_script, "load_model_bundle", lambda *args, **kwargs: object())
    monkeypatch.setattr(gpu_script, "run_model_profiling_plan", lambda **kwargs: [])

    def fake_run_offline_3d_profile(**kwargs):
        calls["include_addition"] = kwargs["include_addition"]
        return {
            "manifest": {
                "experiment_name": "x",
                "model_name": "m",
                "agent_pair": "same_checkpoint",
                "main_dataset": "LongBench",
                "auxiliary_dataset": "RULER",
                "baseline": "full-KV",
            },
            "samples": [],
            "records": [],
            "tables": {
                "layer_by_method": [{"method": "remove_layer", "layer": 0}],
                "head_by_method": [{"method": "remove_layer_head", "head": 0}],
                "chunk_by_method": [{"method": "remove_layer_head_chunk", "chunk": 0}],
                "block_utility": [{"method": "remove_layer_head_chunk", "layer": 0, "head": 0, "chunk": 0}],
                "attention_divergence": [{"method": "remove_layer_head_chunk", "layer": 0, "head": 0, "chunk": 0}],
                "attention_divergence_by_method": [{"method": "remove_layer_head_chunk", "mean_attention_js_divergence": 0.1}],
                "attention_correlation_matrix": [{"method": "remove_layer_head_chunk", "pearson": 0.5}],
                "stability": [],
                "stability_by_task": [],
                "heterogeneity": [],
                "addition_removal_alignment": [],
                "budget_curves": [{"method": "full_kv"}],
            },
            "report": "",
        }

    monkeypatch.setattr(gpu_script, "run_offline_3d_profile", fake_run_offline_3d_profile)

    gpu_script.main()

    assert calls["include_addition"] is True
    assert (Args.output_dir / "raw_results.jsonl").exists()
    assert (Args.output_dir / "layer_importance.csv").exists()
    assert (Args.output_dir / "block_utility.csv").exists()
    assert (Args.output_dir / "attention_divergence.csv").exists()
    assert (Args.output_dir / "attention_correlation_matrix.csv").exists()
    assert (Args.output_dir / "figures" / "fig_layer_importance.png").exists()


def test_gpu_profile_applies_sample_offset_and_method_filter(monkeypatch, tmp_path):
    calls = {}

    class Args:
        model_name = "mock-model"
        device_map = "cpu"
        dtype = "float32"
        samples = tmp_path / "samples.json"
        dataset_name = ""
        config_name = ""
        split = "test"
        spec = None
        max_samples = 1
        sample_offset = 1
        chunk_size = 128
        max_context_tokens = 256
        max_new_tokens = 1
        max_layers = 1
        max_heads = 1
        num_chunks = 2
        include_addition = True
        include_chunk_level = True
        chunk_targets = None
        methods = "remove_layer_head_chunk"
        output_dir = tmp_path / "out"

    Args.samples.write_text(
        "["
        '{"sample_id":"s0","dataset":"LongBench","task_name":"qa","prompt":"p0"},'
        '{"sample_id":"s1","dataset":"LongBench","task_name":"qa","prompt":"p1"}'
        "]"
    )

    monkeypatch.setattr(gpu_script, "parse_args", lambda: Args)
    monkeypatch.setattr(gpu_script, "load_model_bundle", lambda *args, **kwargs: object())

    def fake_run_model_profiling_plan(**kwargs):
        calls["sample_ids"] = [sample.sample_id for sample in kwargs["samples"]]
        calls["plan_methods"] = [item["method"] for item in kwargs["plan"]]
        return []

    monkeypatch.setattr(gpu_script, "run_model_profiling_plan", fake_run_model_profiling_plan)

    def fake_run_offline_3d_profile(**kwargs):
        return {
            "manifest": {
                "experiment_name": "x",
                "model_name": "m",
                "agent_pair": "same_checkpoint",
                "main_dataset": "LongBench",
                "auxiliary_dataset": "RULER",
                "baseline": "full-KV",
            },
            "samples": [],
            "records": [],
            "tables": {
                "layer_by_method": [{"method": "remove_layer", "layer": 0}],
                "head_by_method": [{"method": "remove_layer_head", "head": 0}],
                "chunk_by_method": [{"method": "remove_layer_head_chunk", "chunk": 0}],
                "block_utility": [{"method": "remove_layer_head_chunk", "layer": 0, "head": 0, "chunk": 0}],
                "attention_divergence": [{"method": "remove_layer_head_chunk", "layer": 0, "head": 0, "chunk": 0}],
                "attention_divergence_by_method": [{"method": "remove_layer_head_chunk", "mean_attention_js_divergence": 0.1}],
                "attention_correlation_matrix": [{"method": "remove_layer_head_chunk", "pearson": 0.5}],
                "stability": [],
                "stability_by_task": [],
                "heterogeneity": [],
                "addition_removal_alignment": [],
                "budget_curves": [{"method": "full_kv"}],
            },
            "report": "",
        }

    monkeypatch.setattr(gpu_script, "run_offline_3d_profile", fake_run_offline_3d_profile)

    gpu_script.main()

    assert calls["sample_ids"] == ["s1"]
    assert calls["plan_methods"] == [
        "full_kv",
        "b_only",
        "remove_layer_head_chunk",
        "remove_layer_head_chunk",
    ]


def test_gpu_profile_uses_span_size_for_stage2_plan_and_summary(monkeypatch, tmp_path):
    calls = {}

    class Args:
        model_name = "mock-model"
        device_map = "cpu"
        dtype = "float32"
        samples = tmp_path / "samples.json"
        dataset_name = ""
        config_name = ""
        split = "test"
        spec = None
        max_samples = 1
        sample_offset = 0
        chunk_size = 128
        span_size = 16
        max_context_tokens = 512
        max_new_tokens = 1
        max_layers = 1
        max_heads = 1
        num_chunks = 0
        num_spans = 0
        include_addition = False
        include_chunk_level = True
        chunk_targets = None
        span_targets = None
        methods = "remove_layer_head_chunk"
        output_dir = tmp_path / "out"

    Args.samples.write_text(
        '[{"sample_id":"s","dataset":"LongBench","task_name":"qa","prompt":"p","answers":["a"],"gold_answer":"a"}]'
    )

    monkeypatch.setattr(gpu_script, "parse_args", lambda: Args)
    monkeypatch.setattr(gpu_script, "load_model_bundle", lambda *args, **kwargs: object())

    def fake_run_model_profiling_plan(**kwargs):
        calls["runner_chunk_size"] = kwargs["chunk_size"]
        calls["span_specs"] = [item for item in kwargs["plan"] if item["method"] == "remove_layer_head_chunk"]
        return []

    monkeypatch.setattr(gpu_script, "run_model_profiling_plan", fake_run_model_profiling_plan)

    def fake_run_offline_3d_profile(**kwargs):
        calls["num_chunks"] = kwargs["num_chunks"]
        return {
            "manifest": {
                "experiment_name": "x",
                "model_name": "m",
                "agent_pair": "same_checkpoint",
                "main_dataset": "LongBench",
                "auxiliary_dataset": "RULER",
                "baseline": "full-KV",
            },
            "samples": [],
            "records": [],
            "tables": {
                "layer_by_method": [{"method": "remove_layer", "layer": 0}],
                "head_by_method": [{"method": "remove_layer_head", "head": 0}],
                "chunk_by_method": [{"method": "remove_layer_head_chunk", "chunk": 0, "span_id": 0}],
                "span_by_method": [{"method": "remove_layer_head_chunk", "span_id": 0}],
                "block_utility": [{"method": "remove_layer_head_chunk", "layer": 0, "head": 0, "chunk": 0}],
                "attention_divergence": [{"method": "remove_layer_head_chunk", "layer": 0, "head": 0, "chunk": 0}],
                "attention_divergence_by_method": [{"method": "remove_layer_head_chunk", "mean_attention_js_divergence": 0.1}],
                "attention_correlation_matrix": [{"method": "remove_layer_head_chunk", "pearson": 0.5}],
                "stability": [],
                "stability_by_task": [],
                "heterogeneity": [],
                "addition_removal_alignment": [],
                "budget_curves": [{"method": "full_kv"}],
            },
            "report": "",
        }

    monkeypatch.setattr(gpu_script, "run_offline_3d_profile", fake_run_offline_3d_profile)

    gpu_script.main()

    summary = __import__("json").loads((Args.output_dir / "summary.json").read_text())
    assert calls["runner_chunk_size"] == 16
    assert calls["num_chunks"] == 32
    assert len(calls["span_specs"]) == 32
    assert summary["span_size"] == 16
    assert summary["num_spans"] == 32
    assert summary["profiling_mode"] == "two_stage_layer_head_then_span"
    assert (Args.output_dir / "span_importance.csv").exists()

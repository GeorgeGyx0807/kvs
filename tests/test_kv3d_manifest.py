from src.kv3d import ProfilingManifest


def test_profiling_manifest_captures_fixed_experiment_context():
    manifest = ProfilingManifest(
        experiment_name="offline_3d_kv_utility_profiling",
        model_name="Qwen3-8B",
        agent_pair="same_checkpoint",
        main_dataset="LongBench",
        auxiliary_dataset="RULER",
        baseline="full-KV",
    )

    payload = manifest.to_dict()

    assert payload["model_name"] == "Qwen3-8B"
    assert payload["main_dataset"] == "LongBench"
    assert payload["baseline"] == "full-KV"


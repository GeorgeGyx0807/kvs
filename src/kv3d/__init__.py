"""3D KV profiling utilities."""

from .blocks import KV3DBlock, KV3DKey, chunk_bounds, chunk_index_for_position
from .analysis import aggregate_profiling_records
from .figures import render_profiling_figures
from .masks import (
    MaskSpec,
    SequentialLayerMaskDict,
    build_decode_layer_masks,
    build_layer_masks,
    build_prefill_layer_masks,
    selection_kv_bytes,
)
from .datasets import export_samples_json, load_hf_dataset_split, longbench_row_to_sample, row_to_sample, row_to_sample_for_dataset
from .io import write_csv_table, write_manifest, write_profile_artifacts, write_records, write_summary
from .plan import ProfilingSpec, filter_profiling_plan, generate_profiling_plan
from .report import build_key_findings, render_profiling_report
from .runner import ProfilingSample, annotate_record_token_spans, run_offline_3d_profile
from .records import KV3DAttentionDivergenceSnapshot, KV3DMetricSnapshot, KV3DProfilingRecord, ProfilingManifest
from .run_validation import KV3DRunValidationResult, validate_kv3d_run
from .run_spec import build_kv3d_run_spec

__all__ = [
    "KV3DBlock",
    "KV3DKey",
    "aggregate_profiling_records",
    "MaskSpec",
    "SequentialLayerMaskDict",
    "build_decode_layer_masks",
    "build_prefill_layer_masks",
    "build_kv3d_run_spec",
    "KV3DMetricSnapshot",
    "KV3DAttentionDivergenceSnapshot",
    "KV3DProfilingRecord",
    "KV3DRunValidationResult",
    "ProfilingManifest",
    "ProfilingSpec",
    "ProfilingSample",
    "annotate_record_token_spans",
    "export_samples_json",
    "load_hf_dataset_split",
    "chunk_bounds",
    "chunk_index_for_position",
    "generate_profiling_plan",
    "filter_profiling_plan",
    "longbench_row_to_sample",
    "build_layer_masks",
    "build_key_findings",
    "selection_kv_bytes",
    "row_to_sample",
    "row_to_sample_for_dataset",
    "render_profiling_report",
    "render_profiling_figures",
    "run_offline_3d_profile",
    "validate_kv3d_run",
    "write_manifest",
    "write_profile_artifacts",
    "write_csv_table",
    "write_records",
    "write_summary",
]

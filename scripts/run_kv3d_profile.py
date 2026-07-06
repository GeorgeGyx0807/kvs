#!/usr/bin/env python3
"""CLI for writing minimal 3D KV profiling artifacts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.kv3d import ProfilingManifest
from src.kv3d.io import write_manifest, write_records, write_summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest-output", type=Path, required=True)
    parser.add_argument("--records-output", type=Path, required=True)
    parser.add_argument("--summary-output", type=Path, required=True)
    parser.add_argument("--experiment-name", required=True)
    parser.add_argument("--model-name", required=True)
    parser.add_argument("--agent-pair", required=True)
    parser.add_argument("--main-dataset", required=True)
    parser.add_argument("--auxiliary-dataset", required=True)
    parser.add_argument("--baseline", required=True)
    parser.add_argument("--records-input", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest = ProfilingManifest(
        experiment_name=args.experiment_name,
        model_name=args.model_name,
        agent_pair=args.agent_pair,
        main_dataset=args.main_dataset,
        auxiliary_dataset=args.auxiliary_dataset,
        baseline=args.baseline,
    )
    records_payload = json.loads(args.records_input.read_text())
    write_manifest(manifest, args.manifest_output)
    write_records(records_payload, args.records_output)
    write_summary(
        {
            "record_count": len(records_payload),
            "model_name": args.model_name,
            "main_dataset": args.main_dataset,
            "baseline": args.baseline,
        },
        args.summary_output,
    )


if __name__ == "__main__":
    main()


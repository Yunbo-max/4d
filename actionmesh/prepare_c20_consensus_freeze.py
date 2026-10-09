"""Build C20's canonical outcome-blind decoder-consensus input freeze.

This zero-GPU builder fixes the source set and prospective scale gates.  It
does not authorize, plan, or execute the GPU producer.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from research_math import c20_consensus_target as target


MAX_SOURCE_SELF_MAP_RMS_OVER_DIAGONAL = 0.05
MAX_CONSENSUS_TARGET_RMS_OVER_DIAGONAL = 0.10
MIN_ACTION_SEED_RMS_OVER_DIAGONAL = 1.0e-4


def build_freeze(root: Path, *, uid: str, generation_seed: int,
                 source_sequence: Path, source_report: Path,
                 generation_identity: Path, capture_identity: Path,
                 window_record: Path, decoder_record: Path,
                 decoder_inputs: Path, decoder_tensors: Path,
                 weights_manifest: Path, source_review: Path,
                 output: Path) -> dict:
    root, output = Path(root).resolve(), Path(output).resolve()
    output.relative_to(root)
    if output.exists():
        raise FileExistsError("Preserve prior C20 input freeze")
    if generation_seed not in (42, 314, 2718):
        raise ValueError("C20 seed must be one of the frozen G01 inference seeds")
    paths = {
        "source_sequence_ref": source_sequence,
        "source_report_ref": source_report,
        "generation_identity_ref": generation_identity,
        "capture_identity_ref": capture_identity,
        "window_record_ref": window_record,
        "decoder_record_ref": decoder_record,
        "decoder_inputs_ref": decoder_inputs,
        "decoder_tensors_ref": decoder_tensors,
        "weights_manifest_ref": weights_manifest,
        "source_review_ref": source_review,
    }
    refs = {name: target.file_ref(root, Path(path).resolve())
            for name, path in paths.items()}
    value = {
        "kind": target.FREEZE_KIND, "version": 1,
        "candidate_id": target.CANDIDATE_ID, "uid": uid,
        "generation_seed": generation_seed,
        "benchmark_revision": target.ACTIONBENCH_REVISION,
        "source_indices": list(target.SOURCE_INDICES),
        "target_indices": list(range(16)), **refs,
        "frozen_without_candidate_or_confirmation_outcomes": True,
        "max_source_self_map_rms_over_diagonal":
            MAX_SOURCE_SELF_MAP_RMS_OVER_DIAGONAL,
        "max_consensus_target_rms_over_diagonal":
            MAX_CONSENSUS_TARGET_RMS_OVER_DIAGONAL,
        "min_action_seed_rms_over_diagonal":
            MIN_ACTION_SEED_RMS_OVER_DIAGONAL,
    }
    value["freeze_digest"] = target.canonical_digest(value)
    target.validate_freeze(root, value)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    return value


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--uid", required=True)
    parser.add_argument("--generation-seed", type=int, required=True)
    for name in ("source-sequence", "source-report", "generation-identity",
                 "capture-identity", "window-record", "decoder-record",
                 "decoder-inputs", "decoder-tensors", "weights-manifest",
                 "source-review", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args(argv)
    value = build_freeze(
        args.root, uid=args.uid, generation_seed=args.generation_seed,
        source_sequence=args.source_sequence, source_report=args.source_report,
        generation_identity=args.generation_identity,
        capture_identity=args.capture_identity,
        window_record=args.window_record, decoder_record=args.decoder_record,
        decoder_inputs=args.decoder_inputs, decoder_tensors=args.decoder_tensors,
        weights_manifest=args.weights_manifest, source_review=args.source_review,
        output=args.output)
    print(json.dumps({"freeze_digest": value["freeze_digest"],
                      "gpu_count": 0, "execution_started": False,
                      "generated_unexecuted": True}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

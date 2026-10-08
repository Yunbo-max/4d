"""C04 profile for the shared official ActionBench scorer/raw collector.

Only the bounded transport, official-adapter invocation and receipt-bound raw
collection machinery are shared with C14.  C04 construction and prospective
comparison validation remain in their candidate-specific modules.
"""
from __future__ import annotations

import importlib.util
import os
import shutil
from pathlib import Path

from research_math import c04_native_comparison as comparison_module


_path = Path(__file__).with_name("c14_native_scoring.py")
_spec = importlib.util.spec_from_file_location(
    "research_math._c04_shared_native_scoring", _path)
if _spec is None or _spec.loader is None:
    raise ImportError("Unable to load shared native scoring implementation")
_shared = importlib.util.module_from_spec(_spec)
_shared.comparison_module = comparison_module
_spec.loader.exec_module(_shared)

_shared.comparison_module = comparison_module
_shared.digest = comparison_module.digest
_shared.file_ref = comparison_module.file_ref
_shared.resolve_ref = comparison_module.resolve_ref
_shared.read_json = comparison_module.read_json
_shared.canonical_digest = comparison_module.canonical_digest
_shared.REQUEST_KIND = "c04-native-scoring-request"
_shared.GPU_OBSERVATION_KIND = "c04-gpu-identity-observation"
_shared.BUNDLE_KIND = "c04-native-scoring-raw-bundle"
_shared.RESULT_KIND = "c04-native-scoring-result"
_shared.TICKET_KIND = "c04-staged-launch-ticket"
_shared.CONSUMPTION_KIND = "c04-gpu-resume-authorization-consumption"
_shared.CLAIM_KIND = "c04-launch-claim"
_shared.CONTROLLER_ENV_PREFIX = "C04"
_shared.CANDIDATE_ROLE = "robust_conic_protection"
_shared.CONTROL_ROLES = (
    "b0", "b_star", "deterministic_protection", "strength_matched_repair")
_shared.RESULT_SCOPE = (
    "One frozen C04 physical scoring pass and receipt-bound raw collection; "
    "no confidence interval, gate, qualification or verdict")
_shared.CONTRAST_DIRECTION = "robust conic protection minus control; lower is better"


_stage_standard_cases = _shared.stage_cases
_validate_standard_archive = _shared._validate_archived_frozen_inputs


def _stage_c04_cases(root: Path, comparison: dict, raw_root: Path) -> dict:
    """Snapshot the complete common-target/source/failure closure before scoring."""
    refs = comparison.get("input_refs", [])
    if not isinstance(refs, list):
        raise ValueError("C04 comparison input_refs must be a list")
    unique = {}
    paths = []
    for ref in refs:
        path = comparison_module.resolve_ref(root, ref)
        prior = unique.get(ref["path"])
        if prior is not None and prior != ref:
            raise ValueError("Conflicting C04 retained input path: " + ref["path"])
        if prior is None:
            unique[ref["path"]] = ref
            paths.append((ref, path))
    sizes = [path.stat().st_size for _, path in paths]
    case_sizes = []
    for case in comparison.get("scoring_cases", []):
        for key in ("report_ref", "sequence_ref", "certificate_ref"):
            if case.get(key) is not None:
                case_sizes.append(
                    comparison_module.resolve_ref(root, case[key]).stat().st_size)
    total = sum(sizes) + sum(case_sizes)
    if (len(sizes) + len(case_sizes) > _shared.RAW_LIMITS["max_files"]
            or max(sizes + case_sizes, default=0)
            > _shared.RAW_LIMITS["max_member_bytes"]
            or total > _shared.RAW_LIMITS["max_unpacked_bytes"]):
        raise ValueError("C04 retained input closure exceeds frozen archive limits")
    if shutil.disk_usage(raw_root).free < 2 * total + 64 * 1024 * 1024:
        raise ValueError("Insufficient disk for C04 input snapshot and raw archive")
    manifest = _stage_standard_cases(root, comparison, raw_root)
    for ref, path in paths:
        _shared._copy_exact(
            path, Path(raw_root) / "frozen-inputs" / ref["path"], ref["sha256"])
    return manifest


def _validate_c04_archive(manifest: dict, request_path: Path,
                          request: dict, comparison: dict) -> None:
    _validate_standard_archive(manifest, request_path, request, comparison)
    inventory = {row["path"]: row for row in manifest["files"]}
    for ref in comparison.get("input_refs", []):
        row = inventory.get("frozen-inputs/" + ref["path"])
        if not isinstance(row, dict) or row.get("sha256") != ref["sha256"]:
            raise ValueError(
                "C04 archive lost retained target/source/failure evidence: "
                + ref["path"])


_shared.stage_cases = _stage_c04_cases
_shared._validate_archived_frozen_inputs = _validate_c04_archive


_shared_execution_authorization = _shared.validate_execution_authorization


def _validate_c04_execution_authorization(*args, **kwargs):
    ticket = _shared_execution_authorization(*args, **kwargs)
    # Recheck the same scientific semantic evidence at actual execution, using
    # the immutable controller plan/consumption already proven by shared code.
    import prepare_c04_native_scoring as planner
    controller_root = Path(os.environ["C04_CONTROLLER_ROOT"]).resolve()
    consumption_path = Path(os.environ["C04_CONSUMPTION_PATH"])
    consumption = comparison_module.read_json(consumption_path)
    native = comparison_module.read_json(comparison_module.resolve_ref(
        controller_root, consumption["native_plan_ref"]))
    planner.require_c04_plan_semantics(controller_root, native)
    return ticket


_shared.validate_execution_authorization = _validate_c04_execution_authorization


def __getattr__(name):
    return getattr(_shared, name)


def main() -> int:
    return _shared.main()


if __name__ == "__main__":
    raise SystemExit(main())

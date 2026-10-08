"""C11 profile for the shared official ActionBench scorer/raw collector."""
from __future__ import annotations

import importlib.util
import shutil
from pathlib import Path

from research_math import c11_native_comparison as comparison_module


_path = Path(__file__).with_name("c14_native_scoring.py")
_spec = importlib.util.spec_from_file_location(
    "research_math._c11_shared_native_scoring", _path)
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
_shared.REQUEST_KIND = "c11-native-scoring-request"
_shared.GPU_OBSERVATION_KIND = "c11-gpu-identity-observation"
_shared.BUNDLE_KIND = "c11-native-scoring-raw-bundle"
_shared.RESULT_KIND = "c11-native-scoring-result"
_shared.TICKET_KIND = "c11-staged-launch-ticket"
_shared.CONSUMPTION_KIND = "c11-gpu-resume-authorization-consumption"
_shared.CLAIM_KIND = "c11-launch-claim"
_shared.CONTROLLER_ENV_PREFIX = "C11"
_shared.CANDIDATE_ROLE = "rotation_preserving_stretch_projection"
_shared.CONTROL_ROLES = ("b0", "b_star", "arap_repair", "elastic_repair")
_shared.RESULT_SCOPE = (
    "One frozen C11 physical scoring pass and receipt-bound raw collection; "
    "no confidence interval, gate, qualification or verdict")
_shared.CONTRAST_DIRECTION = (
    "rotation-preserving stretch projection minus control; lower is better")

_standard_stage = _shared.stage_cases
_standard_archive = _shared._validate_archived_frozen_inputs


def _stage_cases(root: Path, comparison: dict, raw_root: Path) -> dict:
    refs = comparison.get("input_refs", [])
    if not isinstance(refs, list):
        raise ValueError("C11 comparison input_refs must be a list")
    resolved = [(ref, comparison_module.resolve_ref(root, ref)) for ref in refs]
    case_paths = []
    for case in comparison.get("scoring_cases", []):
        for key in ("report_ref", "sequence_ref", "certificate_ref"):
            if case.get(key) is not None:
                case_paths.append(comparison_module.resolve_ref(root, case[key]))
    sizes = [path.stat().st_size for _, path in resolved] + [
        path.stat().st_size for path in case_paths]
    if (len(sizes) > _shared.RAW_LIMITS["max_files"]
            or max(sizes, default=0) > _shared.RAW_LIMITS["max_member_bytes"]
            or sum(sizes) > _shared.RAW_LIMITS["max_unpacked_bytes"]):
        raise ValueError("C11 retained input closure exceeds frozen archive limits")
    if shutil.disk_usage(raw_root).free < 2 * sum(sizes) + 64 * 1024 * 1024:
        raise ValueError("Insufficient disk for C11 input snapshot and raw archive")
    manifest = _standard_stage(root, comparison, raw_root)
    seen = set()
    for ref, path in resolved:
        if ref["path"] in seen:
            continue
        seen.add(ref["path"])
        _shared._copy_exact(
            path, Path(raw_root) / "frozen-inputs" / ref["path"], ref["sha256"])
    return manifest


def _validate_archive(manifest: dict, request_path: Path,
                      request: dict, comparison: dict) -> None:
    _standard_archive(manifest, request_path, request, comparison)
    inventory = {row["path"]: row for row in manifest["files"]}
    for ref in comparison.get("input_refs", []):
        row = inventory.get("frozen-inputs/" + ref["path"])
        if not isinstance(row, dict) or row.get("sha256") != ref["sha256"]:
            raise ValueError("C11 archive lost retained evidence: " + ref["path"])


_shared.stage_cases = _stage_cases
_shared._validate_archived_frozen_inputs = _validate_archive


def __getattr__(name):
    return getattr(_shared, name)


def main() -> int:
    return _shared.main()


if __name__ == "__main__":
    raise SystemExit(main())

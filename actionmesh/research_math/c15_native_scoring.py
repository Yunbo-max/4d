"""C15 profile for the shared official ActionBench scorer/raw collector.

Only bounded transport, one official-adapter call per unique physical output,
and receipt-bound raw collection are shared with C14.  C15 construction and
prospective role freezing remain in :mod:`c15_native_comparison`.
"""
from __future__ import annotations

import importlib.util
import shutil
from pathlib import Path

from research_math import c15_native_comparison as comparison_module


_path = Path(__file__).with_name("c14_native_scoring.py")
_spec = importlib.util.spec_from_file_location(
    "research_math._c15_shared_native_scoring", _path)
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
_shared.REQUEST_KIND = "c15-native-scoring-request"
_shared.GPU_OBSERVATION_KIND = "c15-gpu-identity-observation"
_shared.BUNDLE_KIND = "c15-native-scoring-raw-bundle"
_shared.RESULT_KIND = "c15-native-scoring-result"
_shared.TICKET_KIND = "c15-staged-launch-ticket"
_shared.CONSUMPTION_KIND = "c15-gpu-resume-authorization-consumption"
_shared.CLAIM_KIND = "c15-launch-claim"
_shared.CONTROLLER_ENV_PREFIX = "C15"
_shared.CANDIDATE_ROLE = "protected_residual_svt"
_shared.CONTROL_ROLES = (
    "b0", "b_star", "gaussian", "unprotected_svt", "rank_matched_tsvd")
_shared.RESULT_SCOPE = (
    "One frozen C15 physical scoring pass and receipt-bound raw collection; "
    "no confidence interval, gate, qualification or verdict")
_shared.CONTRAST_DIRECTION = (
    "protected residual SVT minus control; lower is better")


_stage_standard_cases = _shared.stage_cases
_validate_standard_archive = _shared._validate_archived_frozen_inputs


def _stage_c15_cases(root: Path, comparison: dict, raw_root: Path) -> dict:
    """Snapshot the complete C15 artifact/failure closure before scoring."""
    refs = comparison.get("input_refs", [])
    if not isinstance(refs, list):
        raise ValueError("C15 comparison input_refs must be a list")
    staged_case_refs = {}
    for case in comparison.get("scoring_cases", []):
        for key, name in (("report_ref", "report.json"),
                          ("sequence_ref", "sequence.npz"),
                          ("certificate_ref", "certificate.npz")):
            ref = case.get(key)
            if ref is not None:
                staged_case_refs[ref["path"]] = (
                    ref, "cases/" + case["case_id"] + "/" + name)
    unique = {}
    paths = []
    for ref in refs:
        path = comparison_module.resolve_ref(root, ref)
        prior = unique.get(ref["path"])
        if prior is not None and prior != ref:
            raise ValueError("Conflicting C15 retained input path: " + ref["path"])
        if prior is None:
            unique[ref["path"]] = ref
            if ref["path"] not in staged_case_refs:
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
        raise ValueError("C15 retained input closure exceeds frozen archive limits")
    if shutil.disk_usage(raw_root).free < 2 * total + 64 * 1024 * 1024:
        raise ValueError("Insufficient disk for C15 input snapshot and raw archive")
    manifest = _stage_standard_cases(root, comparison, raw_root)
    for ref, path in paths:
        _shared._copy_exact(
            path, Path(raw_root) / "frozen-inputs" / ref["path"], ref["sha256"])
    return manifest


def _validate_c15_archive(manifest: dict, request_path: Path,
                          request: dict, comparison: dict) -> None:
    _validate_standard_archive(manifest, request_path, request, comparison)
    inventory = {row["path"]: row for row in manifest["files"]}
    staged_case_refs = {}
    for case in comparison.get("scoring_cases", []):
        for key, name in (("report_ref", "report.json"),
                          ("sequence_ref", "sequence.npz"),
                          ("certificate_ref", "certificate.npz")):
            ref = case.get(key)
            if ref is not None:
                staged_case_refs[ref["path"]] = (
                    ref, "cases/" + case["case_id"] + "/" + name)
    for ref in comparison.get("input_refs", []):
        row = inventory.get("frozen-inputs/" + ref["path"])
        case_location = staged_case_refs.get(ref["path"])
        if row is None and case_location is not None:
            expected_ref, archived_path = case_location
            if expected_ref != ref:
                raise ValueError("Conflicting staged C15 case evidence: " + ref["path"])
            row = inventory.get(archived_path)
        if not isinstance(row, dict) or row.get("sha256") != ref["sha256"]:
            raise ValueError(
                "C15 archive lost retained source/certificate/failure evidence: "
                + ref["path"])


_shared.stage_cases = _stage_c15_cases
_shared._validate_archived_frozen_inputs = _validate_c15_archive


def build_logical_readout(comparison: dict, official_report: dict) -> dict:
    """Expand unique physical scores to all six frozen C15 roles."""
    physical_rows = {row.get("case_id"): row
                     for row in official_report.get("cases", [])
                     if isinstance(row, dict)
                     and isinstance(row.get("case_id"), str)}
    role_lookup = {row["role"]: row for row in comparison["roles"]}
    logical = []
    for role in comparison_module.ROLES:
        frozen = role_lookup[role]
        row = {"role": role, "method_id": frozen["method_id"],
               "case_id": frozen.get("case_id")}
        alias = frozen.get("alias_of")
        if alias:
            row.update(alias_of=alias, shared_measurement_with=alias)
        if frozen.get("preparation_status") != "completed":
            row.update(status="preparation_error",
                       error=frozen.get("preparation_error"))
        else:
            physical = physical_rows.get(frozen.get("case_id"))
            metrics = _shared._metric_values(physical or {})
            if metrics is None:
                row.update(status="scoring_error",
                           error=(physical or {}).get(
                               "error", "missing physical score"))
            else:
                row.update(status="success", metrics=metrics, n_frames=16)
        logical.append(row)
    candidate = next(row for row in logical
                     if row["role"] == _shared.CANDIDATE_ROLE)
    contrasts = []
    if candidate["status"] == "success":
        for control_role in _shared.CONTROL_ROLES:
            control = next(row for row in logical
                           if row["role"] == control_role)
            if control["status"] == "success":
                contrasts.append({
                    "candidate_role": _shared.CANDIDATE_ROLE,
                    "control_role": control_role,
                    "shared_measurement": (
                        candidate.get("case_id") is not None
                        and candidate.get("case_id") == control.get("case_id")),
                    "deltas": {metric: candidate["metrics"][metric]
                               - control["metrics"][metric]
                               for metric in _shared.METRICS},
                    "direction": _shared.CONTRAST_DIRECTION,
                })
    successful = sum(row["status"] == "success" for row in logical)
    count = len(comparison_module.ROLES)
    return {
        "logical_denominator": {
            "n_roles": count, "roles": list(comparison_module.ROLES),
            "failure_policy": (
                "All frozen roles remain in the denominator; missing/error "
                "metrics are never zero-imputed."),
        },
        "roles": logical,
        "unique_physical_measurements": len({
            row["case_id"] for row in logical
            if row["status"] == "success" and row["case_id"] is not None}),
        "n_successful_roles": successful,
        "n_failed_or_missing_roles": count - successful,
        "contrasts": contrasts,
        "primary_contrast": next((row for row in contrasts
                                  if row["control_role"] == "b_star"), None),
        "confidence_intervals": None,
        "scientific_verdict": "not_computed",
        "native_qualified": False,
    }


_shared.build_logical_readout = build_logical_readout


def __getattr__(name):
    return getattr(_shared, name)


def main() -> int:
    return _shared.main()


if __name__ == "__main__":
    raise SystemExit(main())

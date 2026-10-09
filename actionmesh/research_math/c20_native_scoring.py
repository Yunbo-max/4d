"""C20 profile for the shared official ActionBench scorer/raw collector."""
from __future__ import annotations

import importlib.util
import shutil
from pathlib import Path

from research_math import c20_native_comparison as comparison_module


_path = Path(__file__).with_name("c14_native_scoring.py")
_spec = importlib.util.spec_from_file_location(
    "research_math._c20_shared_native_scoring", _path)
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
_shared.REQUEST_KIND = "c20-native-scoring-request"
_shared.GPU_OBSERVATION_KIND = "c20-gpu-identity-observation"
_shared.BUNDLE_KIND = "c20-native-scoring-raw-bundle"
_shared.RESULT_KIND = "c20-native-scoring-result"
_shared.TICKET_KIND = "c20-staged-launch-ticket"
_shared.CONSUMPTION_KIND = "c20-gpu-resume-authorization-consumption"
_shared.CLAIM_KIND = "c20-launch-claim"
_shared.CONTROLLER_ENV_PREFIX = "C20"
_shared.CANDIDATE_ROLE = "joint_monotone_phase_amplitude"
_shared.CONTROL_ROLES = ("b0", "b_star", "phase_only",
                         "amplitude_only", "simple_lag")
_shared.RESULT_SCOPE = (
    "One frozen C20 physical scoring pass and receipt-bound raw collection; "
    "no confidence interval, gate, qualification or verdict")
_shared.CONTRAST_DIRECTION = (
    "joint phase/amplitude repair minus control; lower is better")
_shared.INCLUDE_APPLICATION_STAGE = True

_standard_stage = _shared.stage_cases
_standard_archive = _shared._validate_archived_frozen_inputs


def _stage_cases(root: Path, comparison: dict, raw_root: Path) -> dict:
    refs = comparison.get("input_refs", [])
    if not isinstance(refs, list):
        raise ValueError("C20 comparison input_refs must be a list")
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
        raise ValueError("C20 retained input closure exceeds frozen archive limits")
    if shutil.disk_usage(raw_root).free < 2 * sum(sizes) + 64 * 1024 * 1024:
        raise ValueError("Insufficient disk for C20 input snapshot and raw archive")
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
            raise ValueError("C20 archive lost retained evidence: " + ref["path"])


_shared.stage_cases = _stage_cases
_shared._validate_archived_frozen_inputs = _validate_archive


def build_logical_readout(comparison: dict, official_report: dict) -> dict:
    """Expand physical scores to C20's six fixed logical roles."""
    physical = {row.get("case_id"): row for row in official_report.get("cases", [])
                if isinstance(row, dict) and isinstance(row.get("case_id"), str)}
    frozen = {row["role"]: row for row in comparison["roles"]}
    logical = []
    for role in comparison_module.ROLES:
        source = frozen[role]
        row = {"role": role, "method_id": source["method_id"],
               "case_id": source.get("case_id")}
        if source.get("alias_of"):
            row.update(alias_of=source["alias_of"],
                       shared_measurement_with=source["alias_of"])
        if source.get("preparation_status") != "completed":
            row.update(status="preparation_error",
                       error=source.get("preparation_error"))
        else:
            measured = physical.get(source.get("case_id"))
            metrics = _shared._metric_values(measured or {})
            if metrics is None:
                row.update(status="scoring_error",
                           error=(measured or {}).get("error", "missing physical score"))
            else:
                row.update(status="success", metrics=metrics, n_frames=16)
        logical.append(row)
    treatment = next(row for row in logical
                     if row["role"] == comparison_module.CANDIDATE_ROLE)
    contrasts = []
    if treatment["status"] == "success":
        for control_role in _shared.CONTROL_ROLES:
            control = next(row for row in logical if row["role"] == control_role)
            if control["status"] == "success":
                contrasts.append({
                    "candidate_role": comparison_module.CANDIDATE_ROLE,
                    "control_role": control_role,
                    "shared_measurement": treatment.get("case_id") == control.get("case_id"),
                    "deltas": {metric: treatment["metrics"][metric]
                               - control["metrics"][metric]
                               for metric in _shared.METRICS},
                    "direction": _shared.CONTRAST_DIRECTION})
    count = len(comparison_module.ROLES)
    successful = sum(row["status"] == "success" for row in logical)
    return {"logical_denominator": {"n_roles": count,
                "roles": list(comparison_module.ROLES),
                "failure_policy": "All frozen roles remain; errors are never zero-imputed."},
            "roles": logical,
            "unique_physical_measurements": len({row["case_id"] for row in logical
                if row["status"] == "success" and row["case_id"] is not None}),
            "n_successful_roles": successful,
            "n_failed_or_missing_roles": count - successful,
            "contrasts": contrasts,
            "primary_contrast": next((row for row in contrasts
                if row["control_role"] == "b_star"), None),
            "confidence_intervals": None, "scientific_verdict": "not_computed",
            "native_qualified": False}


_shared.build_logical_readout = build_logical_readout


def __getattr__(name):
    return getattr(_shared, name)


def main() -> int:
    return _shared.main()


if __name__ == "__main__":
    raise SystemExit(main())

"""C05 profile for official ActionBench scoring and raw collection."""
from __future__ import annotations

import importlib.util
from pathlib import Path

from research_math import c05_native_comparison as comparison_module


_path = Path(__file__).with_name("c14_native_scoring.py")
_spec = importlib.util.spec_from_file_location(
    "research_math._c05_shared_native_scoring", _path)
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
_shared.REQUEST_KIND = "c05-native-scoring-request"
_shared.GPU_OBSERVATION_KIND = "c05-gpu-identity-observation"
_shared.BUNDLE_KIND = "c05-native-scoring-raw-bundle"
_shared.RESULT_KIND = "c05-native-scoring-result"
_shared.TICKET_KIND = "c05-staged-launch-ticket"
_shared.CONSUMPTION_KIND = "c05-gpu-resume-authorization-consumption"
_shared.CLAIM_KIND = "c05-launch-claim"
_shared.CONTROLLER_ENV_PREFIX = "C05"
_shared.CANDIDATE_ROLE = "joint_spatial_labels"
_shared.CONTROL_ROLES = comparison_module.ROLES[:-1]
_shared.RESULT_SCOPE = (
    "One frozen C05 physical scoring pass and receipt-bound raw collection; "
    "no confidence interval, gate, qualification or verdict")
_shared.CONTRAST_DIRECTION = "joint spatial labels minus control; lower is better"


def build_logical_readout(comparison: dict, official_report: dict) -> dict:
    """Expand unique physical scores to all seven frozen C05 roles."""
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
                           error=(measured or {}).get(
                               "error", "missing physical score"))
            else:
                row.update(status="success", metrics=metrics, n_frames=16)
        logical.append(row)
    candidate = next(row for row in logical
                     if row["role"] == _shared.CANDIDATE_ROLE)
    contrasts = []
    if candidate["status"] == "success":
        for control_role in _shared.CONTROL_ROLES:
            control = next(row for row in logical if row["role"] == control_role)
            if control["status"] == "success":
                contrasts.append({
                    "candidate_role": _shared.CANDIDATE_ROLE,
                    "control_role": control_role,
                    "shared_measurement": candidate.get("case_id") is not None
                    and candidate.get("case_id") == control.get("case_id"),
                    "deltas": {metric: candidate["metrics"][metric]
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

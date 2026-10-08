"""C07 profile for the shared five-role prospective comparison freezer."""
from __future__ import annotations

import importlib.util
from pathlib import Path

from research_math import partial_transport_candidate as candidate_module


_path = Path(__file__).with_name("c11_native_comparison.py")
_spec = importlib.util.spec_from_file_location(
    "research_math._c07_shared_native_comparison", _path)
if _spec is None or _spec.loader is None:
    raise ImportError("Unable to load shared comparison implementation")
_shared = importlib.util.module_from_spec(_spec)
_shared.candidate_module = candidate_module
_spec.loader.exec_module(_shared)

_shared.candidate_module = candidate_module
_shared.CANDIDATE_ID = "4d-math-20261006-c07"
_shared.ROLES = ("b0", "b_star", "confidence_threshold_fallback", "full_mass_transport",
                 "partial_mass_native_fallback")
_shared.METHOD_ROLES = _shared.ROLES[2:]
_shared.CONTROL_ROLES = ("confidence_threshold_fallback", "full_mass_transport")
_shared.METHOD_IDS = candidate_module.METHOD_IDS
_shared.CANDIDATE_ROLE = "partial_mass_native_fallback"
_shared.FREEZE_KIND = "c07-native-comparison-freeze"
_shared.DECISION_KIND = "c07-b-star-decision"
_shared.DECISION_NO_OUTCOMES_FIELD = "selected_without_c07_native_outcomes"
_shared.REQUEST_KIND = "c07-native-comparison-request"
_shared.PROFILE_LABEL = "C07"
_shared.ALLOWED_INFERENCE_SEEDS = (42,)
_shared.PRIMARY_METRIC = "cd_motion"
_shared.GUARDRAIL_METRICS = ("cd_3d", "cd_4d")

CANDIDATE_ID = _shared.CANDIDATE_ID
ROLES = _shared.ROLES
METHOD_ROLES = _shared.METHOD_ROLES
CONTROL_ROLES = _shared.CONTROL_ROLES
METHOD_IDS = _shared.METHOD_IDS
CANDIDATE_ROLE = _shared.CANDIDATE_ROLE
FREEZE_KIND = _shared.FREEZE_KIND
DECISION_KIND = _shared.DECISION_KIND
DECISION_NO_OUTCOMES_FIELD = _shared.DECISION_NO_OUTCOMES_FIELD
REQUEST_KIND = _shared.REQUEST_KIND
ALLOWED_INFERENCE_SEEDS = _shared.ALLOWED_INFERENCE_SEEDS
PRIMARY_METRIC = _shared.PRIMARY_METRIC
GUARDRAIL_METRICS = _shared.GUARDRAIL_METRICS


def __getattr__(name):
    return getattr(_shared, name)


def main() -> int:
    return _shared.main()


if __name__ == "__main__":
    raise SystemExit(main())

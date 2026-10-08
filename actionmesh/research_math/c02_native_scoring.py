"""C02 profile for the shared official ActionBench scorer/raw collector.

Only transport, official scoring and bounded archive mechanics are shared with
C14.  C02 construction and certificate validation live in their own modules.
The shared implementation is loaded under an isolated module name so importing
this profile cannot mutate C14's in-process profile.
"""
from __future__ import annotations

import importlib.util
from pathlib import Path

from research_math import c02_native_comparison as comparison_module


_path = Path(__file__).with_name("c14_native_scoring.py")
_spec = importlib.util.spec_from_file_location(
    "research_math._c02_shared_native_scoring", _path)
if _spec is None or _spec.loader is None:
    raise ImportError("Unable to load shared native scoring implementation")
_shared = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_shared)

_shared.comparison_module = comparison_module
_shared.digest = comparison_module.digest
_shared.file_ref = comparison_module.file_ref
_shared.resolve_ref = comparison_module.resolve_ref
_shared.read_json = comparison_module.read_json
_shared.canonical_digest = comparison_module.canonical_digest
_shared.REQUEST_KIND = "c02-native-scoring-request"
_shared.GPU_OBSERVATION_KIND = "c02-gpu-identity-observation"
_shared.BUNDLE_KIND = "c02-native-scoring-raw-bundle"
_shared.RESULT_KIND = "c02-native-scoring-result"
_shared.TICKET_KIND = "c02-staged-launch-ticket"
_shared.CONSUMPTION_KIND = "c02-gpu-resume-authorization-consumption"
_shared.CLAIM_KIND = "c02-launch-claim"
_shared.CONTROLLER_ENV_PREFIX = "C02"
_shared.CANDIDATE_ROLE = "protected_step"
_shared.CONTROL_ROLES = ("b0", "b_star", "geometry_only", "strength_matched_blend")
_shared.RESULT_SCOPE = (
    "One frozen C02 physical scoring pass and receipt-bound raw collection; "
    "no confidence interval, gate, qualification or verdict")
_shared.CONTRAST_DIRECTION = "protected step minus control; lower is better"


def __getattr__(name):
    return getattr(_shared, name)


def main() -> int:
    return _shared.main()


if __name__ == "__main__":
    raise SystemExit(main())

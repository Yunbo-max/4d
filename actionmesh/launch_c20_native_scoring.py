"""Launch finalized C20 scoring through the installed harness exactly once."""
from __future__ import annotations

import importlib.util
from pathlib import Path

import prepare_c20_native_scoring as plan
from research_math import c20_native_scoring as scoring


_path = Path(__file__).with_name("launch_c14_native_scoring.py")
_spec = importlib.util.spec_from_file_location("_c20_shared_launcher", _path)
if _spec is None or _spec.loader is None:
    raise ImportError("Unable to load shared single-owner launcher")
_shared = importlib.util.module_from_spec(_spec)
_shared.plan_module = plan
_shared.scoring = scoring
_spec.loader.exec_module(_shared)

_shared.CANDIDATE_ID = "4d-math-20261006-c20"
_shared.canonical_record_digest = plan.canonical_record_digest
_shared.validate_final_consumption = plan.validate_final_consumption
_shared.scoring = scoring
_shared.PROFILE_LABEL = "C20"
_shared.CLAIM_KIND = "c20-launch-claim"
_shared.ENV_PREFIX = "C20"


def __getattr__(name):
    return getattr(_shared, name)


def main() -> int:
    return _shared.main()


if __name__ == "__main__":
    raise SystemExit(main())

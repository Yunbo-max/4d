"""C03 nine-role profile for the shared prospective comparison freezer."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import importlib.util
import json
from pathlib import Path

from research_math import correlated_calibration_candidate as candidate_module
from research_math import g01_design as g01


_path = Path(__file__).with_name("c11_native_comparison.py")
_spec = importlib.util.spec_from_file_location(
    "research_math._c03_shared_native_comparison", _path)
if _spec is None or _spec.loader is None:
    raise ImportError("Unable to load shared comparison implementation")
_shared = importlib.util.module_from_spec(_spec)
_shared.candidate_module = candidate_module
_spec.loader.exec_module(_shared)

_shared.candidate_module = candidate_module
_shared.CANDIDATE_ID = candidate_module.CANDIDATE_ID
_shared.ROLES = ("b0", "b_star", *candidate_module.ROLES)
_shared.METHOD_ROLES = candidate_module.ROLES
_shared.CONTROL_ROLES = candidate_module.ROLES[:-1]
_shared.METHOD_IDS = candidate_module.METHOD_IDS
_shared.CANDIDATE_ROLE = "full_smoothed_unsquared"
_shared.FREEZE_KIND = "c03-native-comparison-freeze"
_shared.DECISION_KIND = "c03-b-star-decision"
_shared.DECISION_NO_OUTCOMES_FIELD = "selected_without_c03_native_outcomes"
_shared.REQUEST_KIND = "c03-native-comparison-request"
_shared.PROFILE_LABEL = "C03"
_shared.ALLOWED_INFERENCE_SEEDS = candidate_module.c01.G01_GENERATION_SEEDS
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


def _time(value: str | None) -> str:
    if value is None:
        return datetime.now(timezone.utc).isoformat()
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        raise ValueError("Timezone-aware timestamp required")
    return value


def _g01_split(root: Path, design: dict, split_path: Path) -> tuple[dict, dict]:
    split_path = Path(split_path).resolve(); split_path.relative_to(root)
    split = _shared.read_json(split_path)
    split_ref = _shared.file_ref(root, split_path)
    if (split.get("kind") != g01.SPLIT_KIND
            or split.get("version") != g01.DESIGN_VERSION
            or split.get("benchmark_revision") != g01.BENCHMARK_REVISION
            or split.get("design_digest") != design["design_digest"]
            or split.get("split_digest") != g01.canonical_record_digest(
                split, "split_digest")):
        raise ValueError("Exact current G01 family split required")
    stages = [split.get(name) for name in ("d1_ids", "d2_ids", "confirmation_ids")]
    if (any(not isinstance(rows, list) or not rows for rows in stages)
            or any(set(stages[i]) & set(stages[j])
                   for i in range(3) for j in range(i + 1, 3))):
        raise ValueError("Disjoint nonempty G01 stages required")
    return split, split_ref


def _g01_selection(root: Path, design_path: Path, split_path: Path,
                   freeze_path: Path) -> tuple[dict, dict, dict, list[Path]]:
    design_path, freeze_path = Path(design_path).resolve(), Path(freeze_path).resolve()
    design_path.relative_to(root); freeze_path.relative_to(root)
    design = g01.validate_design(root, design_path)
    split, split_ref = _g01_split(root, design, split_path)
    freeze = _shared.read_json(freeze_path)
    if freeze.get("split_ref") != split_ref:
        raise ValueError("Global G01 B* freeze uses a different family split")
    closure = g01.validate_b_star_freeze(
        root, design, freeze, expected_split=split)
    selection = freeze["selected_by_candidate"].get(CANDIDATE_ID)
    if not isinstance(selection, dict):
        raise ValueError("G01 B* freeze lacks C03 selection")
    return design, split, freeze, [Path(g01.__file__).resolve(), design_path,
                                   Path(split_path).resolve(), freeze_path, *closure]


def _role_from_g01(role: str) -> str:
    return "b0" if role == "B0" else role


def make_decision(root: Path, *, uid: str, inference_seed: int,
                  application_stage: str, g01_design: Path,
                  g01_family_split: Path,
                  g01_b_star_freeze: Path | None = None,
                  decided_at: str | None = None) -> dict:
    root = Path(root).resolve()
    if (inference_seed not in ALLOWED_INFERENCE_SEEDS
            or application_stage not in ("d1", "d2", "confirmation")):
        raise ValueError("C03 comparison requires an exact G01 generation seed")
    design_path = Path(g01_design).resolve(); design_path.relative_to(root)
    design = g01.validate_design(root, design_path)
    split, split_ref = _g01_split(root, design, g01_family_split)
    stage_ids = {"d1": split["d1_ids"], "d2": split["d2_ids"],
                 "confirmation": split["confirmation_ids"]}
    if uid not in stage_ids[application_stage]:
        raise ValueError("C03 decision UID is outside the exact G01 stage")
    if application_stage == "d1":
        if g01_b_star_freeze is not None:
            raise ValueError("D1 control collection precedes the global B* freeze")
        selected = {"role": "B0", "selection_stage": "D1_provisional_placeholder",
                    "metric": PRIMARY_METRIC}
        selected_role = "b0"
        selection_status = "provisional_d1_placeholder_not_for_inference"
        global_ref = None
        closure = [Path(g01.__file__).resolve(), design_path,
                   Path(g01_family_split).resolve()]
    else:
        if g01_b_star_freeze is None:
            raise ValueError("D2/confirmation require the reviewed global G01 B* freeze")
        design, split, global_freeze, closure = _g01_selection(
            root, design_path, g01_family_split, g01_b_star_freeze)
        selected = global_freeze["selected_by_candidate"][CANDIDATE_ID]
        selected_role = _role_from_g01(selected["role"])
        selection_status = "reviewed_global_g01_b_star"
        global_ref = _shared.file_ref(root, Path(g01_b_star_freeze).resolve())
    if selected_role not in ("b0", *CONTROL_ROLES):
        raise ValueError("G01 B* selected an ineligible C03 control")
    selected_method_id = ("native-b0" if selected_role == "b0"
                          else candidate_module.METHOD_IDS[selected_role])
    refs = []
    for path in closure:
        ref = _shared.file_ref(root, path)
        if ref not in refs:
            refs.append(ref)
    value = {"kind": DECISION_KIND, "version": 1, "candidate_id": CANDIDATE_ID,
        "uid": uid, "inference_seed": inference_seed,
        "application_stage": application_stage,
        "selected_role": selected_role, "selected_method_id": selected_method_id,
        "selection_status": selection_status,
        DECISION_NO_OUTCOMES_FIELD: True,
        "g01_design_ref": _shared.file_ref(root, design_path),
        "g01_family_split_ref": split_ref,
        "g01_b_star_freeze_ref": global_ref,
        "g01_design_digest": design["design_digest"],
        "g01_selection": selected,
        "selection_basis_refs": refs,
        "decided_at": _time(decided_at)}
    value["decision_digest"] = _shared.canonical_digest(value)
    return value


def _validated_decision(root: Path, freeze: dict) -> tuple[dict, list[dict]]:
    decision = _shared.read_json(_shared.resolve_ref(root, freeze["b_star_decision_ref"]))
    core = {key: value for key, value in decision.items()
            if key != "decision_digest"}
    if (decision.get("kind") != DECISION_KIND
            or decision.get("version") != 1
            or decision.get("candidate_id") != CANDIDATE_ID
            or decision.get("uid") != freeze["uid"]
            or decision.get("inference_seed") != freeze["inference_seed"]
            or decision.get("application_stage") not in ("d1", "d2", "confirmation")
            or decision.get(DECISION_NO_OUTCOMES_FIELD) is not True
            or decision.get("decision_digest") != _shared.canonical_digest(core)):
        raise ValueError("C03 B* decision identity/digest differs")
    try:
        decided = datetime.fromisoformat(decision["decided_at"].replace("Z", "+00:00"))
        frozen = datetime.fromisoformat(freeze["frozen_at"].replace("Z", "+00:00"))
    except (KeyError, AttributeError, ValueError) as error:
        raise ValueError("Timezone-aware C03 B* decision required") from error
    if decided.tzinfo is None or decided > frozen:
        raise ValueError("C03 B* decision must precede comparison freeze")
    design_path = _shared.resolve_ref(root, decision["g01_design_ref"])
    split_path = _shared.resolve_ref(root, decision["g01_family_split_ref"])
    design = g01.validate_design(root, design_path)
    split, split_ref = _g01_split(root, design, split_path)
    stage = decision["application_stage"]
    if decision["uid"] not in {"d1": split["d1_ids"], "d2": split["d2_ids"],
                               "confirmation": split["confirmation_ids"]}[stage]:
        raise ValueError("C03 decision UID no longer belongs to its G01 stage")
    if stage == "d1":
        if (decision.get("g01_b_star_freeze_ref") is not None
                or decision.get("selection_status") !=
                   "provisional_d1_placeholder_not_for_inference"):
            raise ValueError("D1 must use a noninferential B* placeholder")
        selected = {"role": "B0", "selection_stage": "D1_provisional_placeholder",
                    "metric": PRIMARY_METRIC}
        role = "b0"
        closure = [Path(g01.__file__).resolve(), design_path, split_path]
    else:
        g01_freeze_path = _shared.resolve_ref(root, decision["g01_b_star_freeze_ref"])
        design, split, global_freeze, closure = _g01_selection(
            root, design_path, split_path, g01_freeze_path)
        selected = global_freeze["selected_by_candidate"][CANDIDATE_ID]
        role = _role_from_g01(selected["role"])
        review = _shared.read_json(_shared.resolve_ref(root, global_freeze["review_ref"]))
        evidence_time = max(datetime.fromisoformat(global_freeze["frozen_at"].replace("Z", "+00:00")),
                            datetime.fromisoformat(review["reviewed_at"].replace("Z", "+00:00")))
        if (decision.get("selection_status") != "reviewed_global_g01_b_star"
                or decided < evidence_time):
            raise ValueError("C03 decision predates the reviewed global G01 B* freeze")
    method = "native-b0" if role == "b0" else candidate_module.METHOD_IDS[role]
    refs = [_shared.file_ref(root, path) for path in closure]
    unique_refs = []
    for ref in refs:
        if ref not in unique_refs:
            unique_refs.append(ref)
    if (decision.get("g01_design_digest") != design["design_digest"]
            or decision.get("g01_family_split_ref") != split_ref
            or decision.get("g01_selection") != selected
            or decision.get("selected_role") != role
            or decision.get("selected_method_id") != method
            or decision.get("selection_basis_refs") != unique_refs):
        raise ValueError("C03 B* decision is not the reviewed global G01 freeze")
    return decision, unique_refs


_shared._decision = _validated_decision


def make_freeze(root: Path, *, source_sequence: Path, source_report: Path,
                generation_identity: Path, candidate_artifact: Path,
                b_star_decision: Path, frozen_at: str | None = None) -> dict:
    root = Path(root).resolve()
    paths = [Path(path).resolve() for path in (source_sequence, source_report,
             generation_identity, candidate_artifact, b_star_decision)]
    for path in paths:
        path.relative_to(root)
    source_sequence, source_report, generation_identity, candidate_artifact, b_star_decision = paths
    report = _shared.read_json(source_report)
    verified = candidate_module.validate_candidate_artifact(root, candidate_artifact)
    decision = _shared.read_json(b_star_decision)
    uid, seed = report.get("uid"), report.get("seed")
    if (verified.get("uid") != uid or verified.get("seed") != seed
            or decision.get("uid") != uid or decision.get("inference_seed") != seed
            or decision.get("application_stage") != verified.get("application_stage")
            or decision.get("g01_family_split_ref") !=
               verified.get("g01_family_split_ref")):
        raise ValueError("C03 source/artifact/B* decision identity differs")
    roles = [{"role": "b0", "method_id": "native-b0",
        "report_ref": _shared.file_ref(root, source_report),
        "sequence_ref": _shared.file_ref(root, source_sequence),
        "implementation_ref": _shared.file_ref(
            root, root / "actionmesh/research_math/native_context_runner.py"),
        "generation_identity_ref": _shared.file_ref(root, generation_identity)}]
    target = decision.get("selected_role")
    if target not in ("b0", *CONTROL_ROLES):
        raise ValueError("C03 B* decision selected an ineligible role")
    target_method = ("native-b0" if target == "b0" else candidate_module.METHOD_IDS[target])
    if decision.get("selected_method_id") != target_method:
        raise ValueError("C03 B* decision method identity differs")
    roles.append({"role": "b_star", "method_id": target_method, "alias_of": target})
    directory = candidate_artifact.parent
    implementation = _shared.file_ref(
        root, root / "actionmesh/research_math/correlated_calibration_candidate.py")
    for role in candidate_module.ROLES:
        arm_report_path = directory / role / "report.json"
        arm_report = _shared.read_json(arm_report_path)
        completed = arm_report.get("status") == "completed"
        roles.append({"role": role, "method_id": candidate_module.METHOD_IDS[role],
            "report_ref": _shared.file_ref(root, arm_report_path),
            "sequence_ref": (_shared.file_ref(root, directory / role / "sequence.npz")
                             if completed else None),
            "certificate_ref": (_shared.file_ref(root, directory / role / "certificate.npz")
                                if completed else None),
            "implementation_ref": implementation})
    value = {"kind": FREEZE_KIND, "version": 1, "candidate_id": CANDIDATE_ID,
        "uid": uid, "inference_seed": seed, "scoring_seed": 44,
        "primary_metric": PRIMARY_METRIC,
        "guardrail_metrics": list(GUARDRAIL_METRICS),
        "source_sequence_ref": _shared.file_ref(root, source_sequence),
        "source_report_ref": _shared.file_ref(root, source_report),
        "candidate_artifact_ref": _shared.file_ref(root, candidate_artifact),
        "b_star_decision_ref": _shared.file_ref(root, b_star_decision),
        "roles": roles, "frozen_at": _time(frozen_at)}
    value["freeze_digest"] = _shared.canonical_digest(value)
    _shared._validate_freeze(value)
    _shared._decision(root, value)
    return value


_shared_make_request = _shared.make_request


def make_request(root: Path, *, freeze_path: Path, _verify: bool = True) -> dict:
    root = Path(root).resolve()
    request = _shared_make_request(root, freeze_path=freeze_path, _verify=False)
    decision = _shared.read_json(_shared.resolve_ref(
        root, request["b_star_decision_ref"]))
    split_ref = decision["g01_family_split_ref"]
    if split_ref not in request["input_refs"]:
        request["input_refs"].append(split_ref)
    request.update(
        g01_family_split_ref=split_ref,
        g01_family_split_digest=_shared.read_json(
            _shared.resolve_ref(root, split_ref))["split_digest"],
        g01_application_stage=decision["application_stage"])
    request["request_digest"] = _shared.canonical_digest(
        {key: value for key, value in request.items() if key != "request_digest"})
    if _verify:
        _shared.verify_request(root, request)
    return request


_shared.make_request = make_request


def __getattr__(name):
    return getattr(_shared, name)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="operation", required=True)
    decision = sub.add_parser("decision")
    decision.add_argument("--root", type=Path, required=True)
    decision.add_argument("--uid", required=True)
    decision.add_argument("--inference-seed", type=int, required=True)
    decision.add_argument("--application-stage", choices=("d1", "d2", "confirmation"),
                          required=True)
    decision.add_argument("--g01-design", type=Path, required=True)
    decision.add_argument("--g01-family-split", type=Path, required=True)
    decision.add_argument("--g01-b-star-freeze", type=Path)
    decision.add_argument("--decided-at")
    decision.add_argument("--output", type=Path, required=True)
    freeze = sub.add_parser("freeze")
    for name in ("root", "source-sequence", "source-report", "generation-identity",
                 "candidate-artifact", "b-star-decision", "output"):
        freeze.add_argument("--" + name, type=Path, required=True)
    freeze.add_argument("--frozen-at")
    request = sub.add_parser("request")
    request.add_argument("--root", type=Path, required=True)
    request.add_argument("--freeze", type=Path, required=True)
    request.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.operation == "decision":
        value = make_decision(args.root, uid=args.uid, inference_seed=args.inference_seed,
            application_stage=args.application_stage, g01_design=args.g01_design,
            g01_family_split=args.g01_family_split,
            g01_b_star_freeze=args.g01_b_star_freeze,
            decided_at=args.decided_at)
    elif args.operation == "freeze":
        value = make_freeze(args.root, source_sequence=args.source_sequence,
            source_report=args.source_report, generation_identity=args.generation_identity,
            candidate_artifact=args.candidate_artifact,
            b_star_decision=args.b_star_decision, frozen_at=args.frozen_at)
    else:
        value = make_request(args.root, freeze_path=args.freeze)
    output = args.output.resolve(); output.relative_to(args.root.resolve())
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x") as stream:
        stream.write(json.dumps(value, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"output": str(output), "operation": args.operation,
                      "execution_started": False, "native_qualified": False}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

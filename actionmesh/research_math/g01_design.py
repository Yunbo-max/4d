"""Fail-closed validator for the stage-wide 15-candidate G01 design.

This module validates source/design identities and, once Local supplies reviewed
source-derived family evidence, deterministically materializes the shared
development/confirmation split.  It does not import a model, load ActionBench
arrays, run a scorer, perform statistics, authorize a GPU, or advance a gate.
"""
from __future__ import annotations

import argparse
import base64
from datetime import datetime
import hashlib
import json
import math
from pathlib import Path, PurePosixPath
import tarfile
from typing import Any


DESIGN_KIND = "stage-wide-g01-design"
DESIGN_VERSION = "1.0.0"
FAMILY_EVIDENCE_KIND = "g01-family-evidence"
FAMILY_DERIVATION_KIND = "g01-family-derivation"
FAMILY_REVIEW_KIND = "g01-family-derivation-review"
FAMILY_METADATA_KIND = "g01-upstream-family-metadata-receipt"
FAMILY_GEOMETRY_KIND = "g01-geometry-identity-receipt"
SPLIT_KIND = "g01-family-split"
B_STAR_FREEZE_KIND = "g01-b-star-freeze"
B_STAR_REVIEW_KIND = "g01-b-star-freeze-review"
B_STAR_D1_RESULT_KIND = "g01-b-star-d1-control-result"
B_STAR_D1_RECEIPT_KIND = "g01-b-star-d1-native-receipt"
GENERATION_UNIT_KIND = "g01-qualified-generation-unit"
GENERATION_IDENTITY_KIND = "g01-generation-identity"
PRODUCER_RECEIPT_KIND = "g01-generation-producer-receipt"
CONTROL_ADMISSION_KIND = "g01-control-native-admission"
CONTROL_ADMISSION_REVIEW_KIND = "g01-control-native-admission-review"
HARNESS_ATTEMPT_KIND = "g01-native-harness-attempt"
COMPARISON_REQUEST_KIND = "g01-b-star-comparison-request"
SCORING_REQUEST_KIND = "g01-b-star-scoring-request"
OFFICIAL_RESULT_KIND = "g01-b-star-official-score-result"
RAW_MANIFEST_KIND = "g01-b-star-official-raw-manifest"
BENCHMARK_ID = "facebook/actionbench"
BENCHMARK_REVISION = "2796071cbe6248422fcbeab3101fa9f9886cb7b9"
METRICS = ("cd_3d", "cd_4d", "cd_motion")
EXPECTED_MINIMUM_EFFECT = {"cd_3d": 0.002, "cd_motion": 0.003}
EXPECTED_NI_MARGIN = {"cd_3d": 0.002, "cd_4d": 0.004, "cd_motion": 0.003}
EXPECTED_SCORER_REFS = {
    "actionmesh/repo/actionbench/README.md":
        "0dcfafddb40fcfb120dcc94d7801128fe152e451dd1470d5510badd43b9c3f0a",
    "actionmesh/repo/actionbench/evaluate_dataset.py":
        "99e9f428cc82c7b8389f830dc304d3a744538ccf944e8937b61a10770683d195",
    "actionmesh/repo/actionbench/benchmark.py":
        "8dd0c56a285fad113a57048df90537a943a5d3eccc7d200f1730eaa1f57a3543",
    "actionmesh/repo/actionbench/chamfer.py":
        "467c17f00c2744ab789120dc8629eb0f6df3450db0c77b8c47f3fd13bfd63778",
    "actionmesh/repo/actionbench/icp.py":
        "b240db101366d69f45377dd4265650b88844ac745b0404d570749adb0e5956c7",
    "actionmesh/repo/actionbench/sample_mesh.py":
        "bec13522efc71cec6a99c2ea11862b5442210977fae613391212b96a21c70fef",
    "actionmesh/repo/actionbench/sample_point_cloud.py":
        "012a6d2f6fe33e8de2745b00dd57689b93605102935b07ecd6b35381da8a379f",
}
EXPECTED_DESIGN_DIGEST = "2d53b764958f89cb5edcc3949b721ac9a7ca484e7ec8c65d8b07649f7edeeb16"
EXPECTED_REVIEW_CHECKS = {
    "source_identity", "mapping_reproduction", "split_independence",
    "exposure_containment",
}
EXPECTED_ADMISSION_REVIEW_CHECKS = {
    "candidate_role_identity", "implementation_closure", "single_attempt",
    "sequence_identity", "official_scorer_closure",
}


def loads_json_strict(value: str) -> Any:
    """Parse JSON while rejecting duplicate keys and non-finite numbers."""
    def object_without_duplicates(pairs):
        result = {}
        for key, item in pairs:
            if key in result:
                raise ValueError("duplicate JSON key: " + key)
            result[key] = item
        return result

    try:
        parsed = json.loads(
            value, object_pairs_hook=object_without_duplicates,
            parse_constant=lambda item: (_ for _ in ()).throw(
                ValueError("non-finite JSON number: " + item)))
    except json.JSONDecodeError as error:
        raise ValueError("invalid JSON") from error
    return parsed


def read_json(path: Path) -> Any:
    path = Path(path)
    if not path.is_file() or path.is_symlink():
        raise ValueError("regular JSON file required: " + str(path))
    return loads_json_strict(path.read_text())


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")


def canonical_record_digest(value: dict, digest_key: str) -> str:
    payload = {key: item for key, item in value.items() if key != digest_key}
    return hashlib.sha256(_canonical_bytes(payload)).hexdigest()


def _file_digest(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        while True:
            chunk = stream.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def _exact_ref_list(root: Path, value: Any, label: str) -> list[Path]:
    if (not isinstance(value, list) or not value
            or len({_canonical_bytes(item) for item in value}) != len(value)):
        raise ValueError("nonempty distinct " + label + " references required")
    return [resolve_ref(root, item) for item in value]


def _json_pointer(document: Any, pointer: str) -> Any:
    if not isinstance(pointer, str) or not pointer.startswith("/"):
        raise ValueError("absolute JSON pointer required")
    current = document
    for raw in pointer[1:].split("/"):
        token = raw.replace("~1", "/").replace("~0", "~")
        if isinstance(current, dict) and token in current:
            current = current[token]
        elif isinstance(current, list) and token.isdigit() \
                and int(token) < len(current):
            current = current[int(token)]
        else:
            raise ValueError("family key JSON pointer does not resolve")
    return current


def _uid_derived_key(uid: str, value: str) -> bool:
    normalized = "".join(char for char in value.lower() if char.isalnum())
    uid_token = "".join(char for char in uid.lower() if char.isalnum())
    prefix_token = "".join(
        char for char in uid.split("_", 1)[0].lower() if char.isalnum())
    encoded = {
        hashlib.sha256(uid.encode()).hexdigest(),
        hashlib.md5(uid.encode()).hexdigest(),  # nosec - identity detection only
        base64.urlsafe_b64encode(uid.encode()).decode().rstrip("=").lower(),
        base64.b64encode(uid.encode()).decode().rstrip("=").lower(),
    }
    encoded = {"".join(char for char in item if char.isalnum())
               for item in encoded}
    return ((uid_token and uid_token in normalized)
            or (prefix_token and prefix_token in normalized)
            or normalized in encoded or any(item in normalized for item in encoded))


def file_ref(root: Path, path: Path) -> dict[str, str]:
    root = Path(root).resolve()
    path = Path(path)
    if not path.is_file() or path.is_symlink():
        raise ValueError("regular referenced file required")
    path = path.resolve()
    try:
        relative = path.relative_to(root)
    except ValueError as error:
        raise ValueError("referenced file escapes root") from error
    return {
        "path": relative.as_posix(),
        "sha256": _file_digest(path),
    }


def resolve_ref(root: Path, ref: dict) -> Path:
    if set(ref) != {"path", "sha256"}:
        raise ValueError("reference must contain exactly path and sha256")
    relative = ref.get("path")
    digest = ref.get("sha256")
    if (not isinstance(relative, str) or not relative
            or Path(relative).is_absolute() or ".." in Path(relative).parts
            or not isinstance(digest, str) or len(digest) != 64):
        raise ValueError("invalid file reference")
    root = Path(root).resolve()
    lexical_path = root / relative
    if not lexical_path.is_file() or lexical_path.is_symlink():
        raise ValueError("regular referenced file required: " + relative)
    path = lexical_path.resolve()
    try:
        path.relative_to(root)
    except ValueError as error:
        raise ValueError("referenced file escapes root") from error
    actual = _file_digest(path)
    if actual != digest:
        raise ValueError("reference digest mismatch: " + relative)
    return path


def _utc_timestamp(value: Any, label: str) -> datetime:
    if not isinstance(value, str) or not value:
        raise ValueError(label + " timestamp required")
    try:
        result = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as error:
        raise ValueError("invalid " + label + " timestamp") from error
    if result.tzinfo is None or result.utcoffset() is None:
        raise ValueError(label + " timestamp must include timezone")
    return result


def _finite_positive_mapping(value: Any, expected: dict[str, float]) -> bool:
    return (isinstance(value, dict) and set(value) == set(expected)
            and all(not isinstance(item, bool) and isinstance(item, (int, float))
                    and math.isfinite(item) and item > 0
                    for item in value.values()))


def _b_star_role(row: dict) -> str:
    matches = [role for role in row["control_roles"]
               if role.lower() == "b_star"]
    if len(matches) != 1:
        raise ValueError("exactly one B_star control role required")
    return matches[0]


def approved_native_code_refs(design: dict, row: dict) -> list[dict] | None:
    candidate_refs = row["approved_control_implementation_refs"]
    if candidate_refs is None:
        return None
    by_path = {}
    for ref in [*design["approved_native_support_refs"], *candidate_refs]:
        path = ref.get("path") if isinstance(ref, dict) else None
        if path in by_path and by_path[path] != ref:
            raise ValueError("conflicting approved native source identity")
        by_path[path] = ref
    return [by_path[path] for path in sorted(by_path)]


def validate_native_job_code_refs(implementation_refs: list[dict],
                                  scorer_refs: list[dict],
                                  actual_refs: Any) -> list[dict]:
    by_path = {}
    for ref in implementation_refs + scorer_refs:
        path = ref.get("path") if isinstance(ref, dict) else None
        if (not isinstance(path, str) or not path
                or path in by_path and by_path[path] != ref):
            raise ValueError("conflicting canonical native code identity")
        by_path[path] = ref
    expected = [by_path[path] for path in sorted(by_path)]
    if not isinstance(actual_refs, list):
        raise ValueError("native plan code refs must be a list")
    actual_by_path = {}
    for ref in actual_refs:
        path = ref.get("path") if isinstance(ref, dict) else None
        if (not isinstance(path, str) or not path
                or set(ref) != {"path", "sha256"}
                or path in actual_by_path):
            raise ValueError("unique shaped native plan code refs required")
        actual_by_path[path] = ref
    actual = [actual_by_path[path] for path in sorted(actual_by_path)]
    if actual != expected:
        raise ValueError("native plan code refs differ from exact frozen closure")
    return expected


def validate_design_record(root: Path, design: dict) -> dict:
    root = Path(root).resolve()
    required_keys = {
        "kind", "version", "status", "authored_at", "scope",
        "selection_ref", "benchmark", "cohort", "criteria", "statistics",
        "randomness", "failure_denominator", "fairness", "resources",
        "candidates", "approved_native_support_refs", "execution_boundaries",
        "design_digest",
    }
    if (not isinstance(design, dict) or set(design) != required_keys
            or design.get("kind") != DESIGN_KIND
            or design.get("version") != DESIGN_VERSION
            or design.get("status") !=
            "complete_design_generated_unexecuted_dynamic_admission_pending"
            or design.get("design_digest") != canonical_record_digest(
                design, "design_digest")):
        raise ValueError("invalid canonical G01 design record")
    _utc_timestamp(design["authored_at"], "design authored_at")

    selection = read_json(resolve_ref(root, design["selection_ref"]))
    selected_ids = selection.get("selected_ids")
    candidates = design.get("candidates")
    if (not isinstance(selected_ids, list) or len(selected_ids) != 15
            or not isinstance(candidates, list) or len(candidates) != 15
            or [row.get("candidate_id") for row in candidates] != selected_ids):
        raise ValueError("G01 candidates must equal the selected 15 in rank order")

    benchmark = design.get("benchmark")
    if (not isinstance(benchmark, dict)
            or benchmark.get("id") != BENCHMARK_ID
            or benchmark.get("revision") != BENCHMARK_REVISION
            or benchmark.get("split") != "complete_released_population"
            or benchmark.get("population_size") != 128
            or benchmark.get("frames_per_sample") != 16):
        raise ValueError("exact released ActionBench population required")
    population = read_json(resolve_ref(root, benchmark["population_ref"]))
    uids = population.get("uids")
    if (population.get("dataset") != BENCHMARK_ID
            or population.get("revision") != BENCHMARK_REVISION
            or not isinstance(uids, list) or len(uids) != 128
            or len(set(uids)) != 128
            or not all(isinstance(uid, str) and uid for uid in uids)):
        raise ValueError("complete released population manifest required")
    resolve_ref(root, benchmark["semantics_ref"])
    resolve_ref(root, benchmark["baseline_reproduction_ref"])
    scorer = benchmark.get("official_scorer")
    if (not isinstance(scorer, dict)
            or scorer.get("entrypoint") !=
            "actionmesh/repo/actionbench/evaluate_dataset.py"
            or scorer.get("implementation") !=
            "actionmesh/repo/actionbench/benchmark.py:compute_chamfer_3d_4d"
            or scorer.get("metrics") != list(METRICS)
            or scorer.get("direction") != "minimize"
            or scorer.get("sampling_seed") != 44
            or scorer.get("surface_samples") != 100000
            or scorer.get("icp_samples") != 10000
            or scorer.get("icp_rotations") != 24
            or scorer.get("icp_iterations") != 200
            or not isinstance(scorer.get("source_refs"), list)
            or {ref.get("path"): ref.get("sha256")
                for ref in scorer["source_refs"]
                if isinstance(ref, dict)} != EXPECTED_SCORER_REFS
            or len(scorer["source_refs"]) != len(EXPECTED_SCORER_REFS)):
        raise ValueError("exact official ActionBench scorer contract required")
    for ref in scorer["source_refs"]:
        resolve_ref(root, ref)
    support_refs = design.get("approved_native_support_refs")
    if (not isinstance(support_refs, list) or not support_refs
            or not all(isinstance(ref, dict) for ref in support_refs)
            or [ref.get("path") for ref in support_refs] !=
            sorted(ref.get("path") for ref in support_refs)
            or len({_canonical_bytes(ref) for ref in support_refs}) !=
            len(support_refs)):
        raise ValueError("canonical approved native support closure required")
    for ref in support_refs:
        resolve_ref(root, ref)

    cohort = design.get("cohort")
    if (not isinstance(cohort, dict)
            or cohort.get("independent_unit") != "asset_family"
            or cohort.get("split_salt") != "4d-math-restart-20261006-v1"
            or cohort.get("d1_unexposed_family_count") != 12
            or cohort.get("d2_unexposed_family_count") != 12
            or cohort.get("minimum_confirmation_families") != 30):
        raise ValueError("complete family-level cohort policy required")
    exposure = read_json(resolve_ref(root, cohort["exposure_ref"]))
    exposed = exposure.get("overnight_development_uids", []) + \
        exposure.get("earlier_inspected_uids", [])
    if (not exposed or not set(exposed).issubset(set(uids))):
        raise ValueError("retained exposure inventory must match population")

    criteria = design.get("criteria")
    if (not isinstance(criteria, dict)
            or criteria.get("effect_direction") !=
            "treatment_minus_control_lower_is_better"
            or not _finite_positive_mapping(
                criteria.get("minimum_effect"), EXPECTED_MINIMUM_EFFECT)
            or criteria.get("minimum_effect") != EXPECTED_MINIMUM_EFFECT
            or not _finite_positive_mapping(
                criteria.get("noninferiority_margin"), EXPECTED_NI_MARGIN)
            or criteria.get("noninferiority_margin") != EXPECTED_NI_MARGIN):
        raise ValueError("complete fixed numeric criteria required")
    resolve_ref(root, criteria["rationale_ref"])

    total_contrasts = 0
    approved_implementation_count = 0
    for row in candidates:
        expected_row_keys = {
            "candidate_id", "spec_ref", "treatment_role", "control_roles",
            "primary_metric", "guardrail_metrics", "confirmatory_contrasts",
            "approved_control_implementation_refs",
        }
        if not isinstance(row, dict) or set(row) != expected_row_keys:
            raise ValueError("invalid candidate G01 row")
        spec = read_json(resolve_ref(root, row["spec_ref"]))
        native = spec.get("native_experiment")
        treatment = row.get("treatment_role")
        controls = row.get("control_roles")
        if (spec.get("candidate_id") != row["candidate_id"]
                or not isinstance(native, dict)
                or not isinstance(controls, list) or not controls
                or len(set(controls)) != len(controls)
                or treatment in controls
                or set(native.get("arms", [])) != {treatment, *controls}
                or native.get("primary_metric") != row.get("primary_metric")
                or native.get("guardrail_metrics") != row.get("guardrail_metrics")
                or row.get("primary_metric") not in EXPECTED_MINIMUM_EFFECT
                or set(row.get("guardrail_metrics", [])) !=
                set(METRICS) - {row.get("primary_metric")}):
            raise ValueError("candidate G01 row contradicts selected specification")
        approved_refs = row["approved_control_implementation_refs"]
        if approved_refs is not None:
            if (not isinstance(approved_refs, list) or not approved_refs
                    or not all(isinstance(ref, dict) for ref in approved_refs)
                    or [ref.get("path") for ref in approved_refs] !=
                    sorted(ref.get("path") for ref in approved_refs)
                    or len({_canonical_bytes(ref) for ref in approved_refs}) !=
                    len(approved_refs)):
                raise ValueError("canonical approved candidate implementation closure required")
            for ref in approved_refs:
                resolve_ref(root, ref)
            approved_native_code_refs(design, row)
            approved_implementation_count += 1
        b_star = _b_star_role(row)
        expected_contrasts = [
            f"{treatment}-{control}:{row['primary_metric']}"
            for control in controls
        ] + [
            f"{treatment}-{b_star}:{metric}"
            for metric in row["guardrail_metrics"]
        ]
        if row.get("confirmatory_contrasts") != expected_contrasts:
            raise ValueError("candidate contrast inventory is incomplete")
        total_contrasts += len(expected_contrasts)
    if approved_implementation_count != 15:
        raise ValueError("exactly fifteen delivered candidate implementation closures required")

    statistics = design.get("statistics")
    if (not isinstance(statistics, dict)
            or statistics.get("design") != "paired_clustered"
            or statistics.get("independent_unit") != "asset_family"
            or statistics.get("uncertainty") !=
            "trusted_external_family_cluster_bootstrap"
            or statistics.get("bootstrap_method") != "basic_one_sided"
            or statistics.get("bootstrap_resamples") != 200000
            or statistics.get("bootstrap_seed") != 20261006
            or statistics.get("familywise_alpha") != 0.05
            or statistics.get("confirmatory_test_count") != total_contrasts
            or total_contrasts != 99
            or not math.isclose(statistics.get("per_test_alpha", -1),
                                0.05 / total_contrasts, rel_tol=0, abs_tol=1e-18)
            or statistics.get("complete_inventory") is not True):
        raise ValueError("complete multiplicity and clustered analysis required")

    randomness = design.get("randomness")
    if (not isinstance(randomness, dict)
            or randomness.get("generation_seeds") != [42, 314, 2718]
            or randomness.get("official_scorer_seed") != 44):
        raise ValueError("frozen generation/scoring seed coverage required")
    failure = design.get("failure_denominator")
    if (not isinstance(failure, dict)
            or failure.get("unit") !=
            "candidate_x_uid_x_generation_seed_x_logical_role"
            or "Never zero-impute" not in failure.get("policy", "")):
        raise ValueError("complete failure denominator required")
    resources = design.get("resources")
    if (not isinstance(resources, dict)
            or resources.get("environment") !=
            "native Conda; no Docker or other container runtime"
            or resources.get("window_seconds") != 28800
            or resources.get("collection_reserve_seconds") != 1800
            or resources.get("workload_budget_seconds") != 27000
            or resources.get("attempts_per_scientific_unit") != 1
            or resources.get("retries_per_scientific_unit") != 0
            or not str(resources.get("gpu_stop", "")).startswith("active")):
        raise ValueError("frozen resource and STOP policy required")
    boundaries = design.get("execution_boundaries")
    if (not isinstance(boundaries, dict)
            or boundaries.get("source_design_complete") is not True
            or boundaries.get("gpu_dispatch_allowed_now") is not False
            or any(boundaries.get(key) is not True for key in (
                "dynamic_family_evidence_required",
                "natural_gate_0_and_ipcg_required_per_candidate",
                "current_local_acceptance_required",
                "source_complete_method_required",
                "b_star_freeze_required",
                "trusted_live_official_scorer_replay_required",
                "trusted_live_cluster_analysis_required",
                "explicit_gpu_resume_authorization_required"))):
        raise ValueError("G01 execution boundaries are incomplete")
    if design["design_digest"] != EXPECTED_DESIGN_DIGEST:
        raise ValueError("G01 design differs from the independently frozen source digest")
    return design


def validate_design(root: Path, design_path: Path) -> dict:
    return validate_design_record(root, read_json(design_path))


def _validate_family_evidence(root: Path, design: dict, evidence: dict,
                              evidence_root: Path | None = None) -> dict[str, str]:
    evidence_root = Path(evidence_root or root).resolve()
    expected_keys = {
        "kind", "version", "benchmark_revision", "population_ref",
        "family_by_uid", "derivation_ref", "review_ref", "frozen_at",
        "evidence_digest",
    }
    if (not isinstance(evidence, dict) or set(evidence) != expected_keys
            or evidence.get("kind") != FAMILY_EVIDENCE_KIND
            or evidence.get("version") != DESIGN_VERSION
            or evidence.get("benchmark_revision") != BENCHMARK_REVISION
            or evidence.get("population_ref") !=
            design["benchmark"]["population_ref"]
            or evidence.get("evidence_digest") != canonical_record_digest(
                evidence, "evidence_digest")):
        raise ValueError("invalid family evidence record")
    _utc_timestamp(evidence["frozen_at"], "family evidence frozen_at")
    population = read_json(resolve_ref(root, design["benchmark"]["population_ref"]))
    uids = population["uids"]
    family_by_uid = evidence.get("family_by_uid")
    if (not isinstance(family_by_uid, dict)
            or set(family_by_uid) != set(uids)
            or not all(isinstance(family, str) and family
                       for family in family_by_uid.values())):
        raise ValueError("family evidence must cover complete released population")

    derivation = read_json(resolve_ref(evidence_root, evidence["derivation_ref"]))
    derivation_keys = {
        "kind", "version", "benchmark_revision", "units",
        "assignment_rule", "author", "derived_at", "derivation_digest",
    }
    units = derivation.get("units") if isinstance(derivation, dict) else None
    if (not isinstance(derivation, dict) or set(derivation) != derivation_keys
            or derivation.get("kind") != FAMILY_DERIVATION_KIND
            or derivation.get("version") != DESIGN_VERSION
            or derivation.get("benchmark_revision") != BENCHMARK_REVISION
            or derivation.get("assignment_rule") !=
            "family_id=sha256(canonical_family_basis); uid_and_outcomes_forbidden"
            or not isinstance(derivation.get("author"), str)
            or not derivation["author"]
            or not isinstance(units, dict) or set(units) != set(uids)
            or derivation.get("derivation_digest") != canonical_record_digest(
                derivation, "derivation_digest")):
        raise ValueError("complete reproducible family derivation required")
    _utc_timestamp(derivation["derived_at"], "family derived_at")
    derived_family_by_uid = {}
    source_authors = set()
    for uid in uids:
        unit = units[uid]
        if (not isinstance(unit, dict)
                or set(unit) != {"family_basis", "family_id",
                                 "metadata_receipt_ref",
                                 "geometry_receipt_ref"}):
            raise ValueError("complete per-UID family source evidence required")
        metadata_path = resolve_ref(evidence_root, unit["metadata_receipt_ref"])
        geometry_path = resolve_ref(evidence_root, unit["geometry_receipt_ref"])
        if metadata_path == geometry_path:
            raise ValueError("metadata and geometry receipts must be distinct")
        metadata = read_json(metadata_path)
        metadata_keys = {
            "kind", "version", "benchmark_revision", "uid",
            "metadata_source_ref", "family_key_pointers", "extractor_ref",
            "extracted_family_keys", "author", "created_at", "receipt_digest",
        }
        if (not isinstance(metadata, dict) or set(metadata) != metadata_keys
                or metadata.get("kind") != FAMILY_METADATA_KIND
                or metadata.get("version") != DESIGN_VERSION
                or metadata.get("benchmark_revision") != BENCHMARK_REVISION
                or metadata.get("uid") != uid
                or not isinstance(metadata.get("author"), str)
                or not metadata["author"]
                or metadata.get("receipt_digest") != canonical_record_digest(
                    metadata, "receipt_digest")):
            raise ValueError("exact upstream family metadata receipt required")
        _utc_timestamp(metadata["created_at"], "family metadata created_at")
        metadata_source_path = resolve_ref(
            evidence_root, metadata["metadata_source_ref"])
        extractor_path = resolve_ref(evidence_root, metadata["extractor_ref"])
        if (metadata_source_path == extractor_path
                or metadata_source_path.suffix.lower() != ".json"
                or extractor_path.suffix.lower() != ".py"):
            raise ValueError("distinct JSON metadata and Python extractor required")
        metadata_document = read_json(metadata_source_path)
        pointers = metadata.get("family_key_pointers")
        extracted = metadata.get("extracted_family_keys")
        if (not isinstance(pointers, list) or not pointers
                or pointers != sorted(set(pointers))
                or not isinstance(extracted, list) or not extracted
                or extracted != sorted(set(extracted))):
            raise ValueError("canonical family key pointers and values required")
        parsed = []
        for pointer in pointers:
            value = _json_pointer(metadata_document, pointer)
            if not isinstance(value, str) or not value:
                raise ValueError("family key pointer must resolve to a string")
            parsed.append(value)
        if sorted(set(parsed)) != extracted:
            raise ValueError("family keys differ from pinned upstream metadata")
        if any(_uid_derived_key(uid, value) for value in extracted):
            raise ValueError("UID-derived family keys are forbidden")

        geometry = read_json(geometry_path)
        geometry_keys = {
            "kind", "version", "benchmark_revision", "uid", "geometry_refs",
            "extractor_ref", "geometry_sha256", "author", "created_at",
            "receipt_digest",
        }
        if (not isinstance(geometry, dict) or set(geometry) != geometry_keys
                or geometry.get("kind") != FAMILY_GEOMETRY_KIND
                or geometry.get("version") != DESIGN_VERSION
                or geometry.get("benchmark_revision") != BENCHMARK_REVISION
                or geometry.get("uid") != uid
                or not isinstance(geometry.get("author"), str)
                or not geometry["author"]
                or geometry.get("receipt_digest") != canonical_record_digest(
                    geometry, "receipt_digest")):
            raise ValueError("exact geometry identity receipt required")
        _utc_timestamp(geometry["created_at"], "geometry identity created_at")
        geometry_paths = _exact_ref_list(
            evidence_root, geometry.get("geometry_refs"), "geometry")
        geometry_extractor = resolve_ref(evidence_root, geometry["extractor_ref"])
        allowed_geometry_suffixes = {
            ".bin", ".glb", ".gltf", ".npy", ".npz", ".obj", ".off",
            ".ply", ".stl",
        }
        if (geometry_extractor.suffix.lower() != ".py"
                or any(path.suffix.lower() not in allowed_geometry_suffixes
                       for path in geometry_paths)
                or metadata_source_path in geometry_paths
                or extractor_path in geometry_paths
                or geometry_extractor in geometry_paths):
            raise ValueError("family metadata, extractor and geometry categories overlap")
        geometry_hashes = sorted({_file_digest(path) for path in geometry_paths})
        if geometry.get("geometry_sha256") != geometry_hashes:
            raise ValueError("geometry hashes differ from pinned geometry files")
        basis = unit.get("family_basis")
        if (not isinstance(basis, dict)
                or set(basis) != {"upstream_family_keys",
                                  "metadata_content_sha256",
                                  "geometry_content_sha256",
                                  "extractor_content_sha256"}):
            raise ValueError("canonical family basis required")
        upstream = basis["upstream_family_keys"]
        metadata_hashes = basis["metadata_content_sha256"]
        geometry_hashes = basis["geometry_content_sha256"]
        extractor_hashes = basis["extractor_content_sha256"]
        expected_basis = {
            "upstream_family_keys": extracted,
            "metadata_content_sha256": [metadata["metadata_source_ref"]["sha256"]],
            "geometry_content_sha256": geometry["geometry_sha256"],
            "extractor_content_sha256": sorted({
                metadata["extractor_ref"]["sha256"],
                geometry["extractor_ref"]["sha256"],
            }),
        }
        digest_categories = [
            set(expected_basis["metadata_content_sha256"]),
            set(expected_basis["geometry_content_sha256"]),
            set(expected_basis["extractor_content_sha256"]),
        ]
        if any(digest_categories[left] & digest_categories[right]
               for left in range(len(digest_categories))
               for right in range(left + 1, len(digest_categories))):
            raise ValueError("family metadata, geometry and extractor digests overlap")
        if (basis != expected_basis
                or not all(isinstance(value, str) and len(value) == 64
                           and all(char in "0123456789abcdef" for char in value)
                           for value in metadata_hashes + geometry_hashes
                           + extractor_hashes)):
            raise ValueError("complete canonical family identity evidence required")
        family_id = hashlib.sha256(_canonical_bytes(basis)).hexdigest()
        if unit.get("family_id") != family_id:
            raise ValueError("family ID is not reproducible from source identity")
        derived_family_by_uid[uid] = family_id
        source_authors.update((metadata["author"], geometry["author"]))
    if derived_family_by_uid != family_by_uid:
        raise ValueError("family map differs from reproducible source derivation")

    review = read_json(resolve_ref(evidence_root, evidence["review_ref"]))
    review_keys = {
        "kind", "version", "derivation_ref", "outcome", "checks",
        "reviewer", "reviewed_at", "review_digest",
    }
    if (not isinstance(review, dict) or set(review) != review_keys
            or review.get("kind") != FAMILY_REVIEW_KIND
            or review.get("version") != DESIGN_VERSION
            or review.get("derivation_ref") != evidence["derivation_ref"]
            or review.get("outcome") != "verified"
            or set(review.get("checks", [])) != EXPECTED_REVIEW_CHECKS
            or not isinstance(review.get("reviewer"), str)
            or not review["reviewer"]
            or review["reviewer"] == derivation["author"]
            or review["reviewer"] in source_authors
            or review.get("review_digest") != canonical_record_digest(
                review, "review_digest")):
        raise ValueError("independent verified family review required")
    reviewed_at = _utc_timestamp(review["reviewed_at"], "family reviewed_at")
    if reviewed_at < _utc_timestamp(derivation["derived_at"], "family derived_at"):
        raise ValueError("family review precedes derivation")
    return family_by_uid


def family_evidence_closure(root: Path, design: dict, evidence: dict) -> list[Path]:
    """Validate and return every dynamic evidence file needed by a staged run."""
    root = Path(root).resolve()
    _validate_family_evidence(root, design, evidence)
    paths = [
        resolve_ref(root, evidence["derivation_ref"]),
        resolve_ref(root, evidence["review_ref"]),
    ]
    derivation = read_json(paths[0])
    for unit in derivation["units"].values():
        metadata_path = resolve_ref(root, unit["metadata_receipt_ref"])
        geometry_path = resolve_ref(root, unit["geometry_receipt_ref"])
        paths.extend((metadata_path, geometry_path))
        metadata = read_json(metadata_path)
        geometry = read_json(geometry_path)
        paths.extend((resolve_ref(root, metadata["metadata_source_ref"]),
                      resolve_ref(root, metadata["extractor_ref"]),
                      resolve_ref(root, geometry["extractor_ref"])))
        paths.extend(resolve_ref(root, ref) for ref in geometry["geometry_refs"])
    unique = {}
    for path in paths:
        unique[path.resolve()] = path.resolve()
    return [unique[path] for path in sorted(unique, key=str)]


def derive_family_split(root: Path, design: dict, evidence: dict,
                        evidence_root: Path | None = None) -> dict:
    root = Path(root).resolve()
    family_by_uid = _validate_family_evidence(
        root, design, evidence, evidence_root=evidence_root)
    population = read_json(resolve_ref(root, design["benchmark"]["population_ref"]))
    uids = population["uids"]
    exposure = read_json(resolve_ref(root, design["cohort"]["exposure_ref"]))
    exposed_uids = set(exposure["overnight_development_uids"])
    exposed_uids.update(exposure["earlier_inspected_uids"])
    exposed_families = {family_by_uid[uid] for uid in exposed_uids}
    all_families = set(family_by_uid.values())
    unexposed_families = all_families - exposed_families
    salt = design["cohort"]["split_salt"]
    ordered = sorted(
        unexposed_families,
        key=lambda family: (hashlib.sha256(
            _canonical_bytes([salt, family])).hexdigest(), family),
    )
    d1_count = design["cohort"]["d1_unexposed_family_count"]
    d2_count = design["cohort"]["d2_unexposed_family_count"]
    minimum_confirmation = design["cohort"]["minimum_confirmation_families"]
    if len(ordered) < d1_count + d2_count + minimum_confirmation:
        raise ValueError("insufficient unexposed families for frozen G01 split")
    d1_families = ordered[:d1_count]
    d2_families = ordered[d1_count:d1_count + d2_count]
    confirmation_families = ordered[d1_count + d2_count:]

    def ids_for(families):
        families = set(families)
        return [uid for uid in uids if family_by_uid[uid] in families]

    split = {
        "kind": SPLIT_KIND,
        "version": DESIGN_VERSION,
        "benchmark_revision": BENCHMARK_REVISION,
        "design_digest": design["design_digest"],
        "family_evidence_digest": evidence["evidence_digest"],
        "split_salt": salt,
        "development_exposed_families": sorted(exposed_families),
        "development_exposed_ids": ids_for(exposed_families),
        "d1_families": d1_families,
        "d1_ids": ids_for(d1_families),
        "d2_families": d2_families,
        "d2_ids": ids_for(d2_families),
        "confirmation_families": confirmation_families,
        "confirmation_ids": ids_for(confirmation_families),
        "family_by_uid": {uid: family_by_uid[uid] for uid in uids},
    }
    covered = (set(split["development_exposed_ids"]) | set(split["d1_ids"])
               | set(split["d2_ids"]) | set(split["confirmation_ids"]))
    if covered != set(uids):
        raise ValueError("derived split does not cover released population")
    split["split_digest"] = canonical_record_digest(split, "split_digest")
    return split


def validate_b_star_selections(design: dict, selections: dict) -> dict:
    if not isinstance(selections, dict):
        raise ValueError("B_star selections must be a candidate mapping")
    by_id = {row["candidate_id"]: row for row in design["candidates"]}
    if set(selections) != set(by_id):
        raise ValueError("B_star selections must cover all selected candidates")
    for candidate_id, selection in selections.items():
        row = by_id[candidate_id]
        selectable = [role for role in row["control_roles"]
                      if role.lower() != "b_star"]
        if (not isinstance(selection, dict)
                or set(selection) != {"role", "selection_stage", "metric"}
                or selection.get("role") not in selectable
                or selection.get("selection_stage") != "D1_only"
                or selection.get("metric") != row["primary_metric"]):
            raise ValueError(
                "B_star must alias a selectable declared control selected on D1 only")
    return selections


def _control_admission_payload_digest(admission: dict) -> str:
    return hashlib.sha256(_canonical_bytes({
        key: value for key, value in admission.items()
        if key not in {"review_ref", "admission_digest"}
    })).hexdigest()


def _validate_control_admission(root: Path, design: dict, row: dict,
                                role: str, uid: str, seed: int,
                                sequence_ref: dict, admission_ref: dict) \
        -> tuple[list[Path], datetime]:
    admission_path = resolve_ref(root, admission_ref)
    admission = read_json(admission_path)
    keys = {
        "kind", "version", "design_digest", "candidate_id", "control_role",
        "uid", "generation_seed", "candidate_spec_ref", "sequence_ref",
        "implementation_refs", "generation_identity_ref",
        "producer_receipt_ref", "harness_attempt_ref", "official_scorer_refs",
        "status", "author", "admitted_at", "review_ref", "admission_digest",
    }
    expected_scorer = design["benchmark"]["official_scorer"]["source_refs"]
    expected_implementation = approved_native_code_refs(design, row)
    if expected_implementation is None:
        raise ValueError("candidate implementation closure is not frozen in G01")
    if (not isinstance(admission, dict) or set(admission) != keys
            or admission.get("kind") != CONTROL_ADMISSION_KIND
            or admission.get("version") != DESIGN_VERSION
            or admission.get("design_digest") != design["design_digest"]
            or admission.get("candidate_id") != row["candidate_id"]
            or admission.get("control_role") != role
            or admission.get("uid") != uid
            or admission.get("generation_seed") != seed
            or admission.get("candidate_spec_ref") != row["spec_ref"]
            or admission.get("sequence_ref") != sequence_ref
            or admission.get("implementation_refs") != expected_implementation
            or admission.get("official_scorer_refs") != expected_scorer
            or admission.get("status") != "admitted"
            or not isinstance(admission.get("author"), str)
            or not admission["author"]
            or admission.get("admission_digest") != canonical_record_digest(
                admission, "admission_digest")):
        raise ValueError("exact candidate/control native admission required")
    admitted_at = _utc_timestamp(admission["admitted_at"], "control admitted_at")
    implementation_paths = _exact_ref_list(
        root, admission.get("implementation_refs"), "implementation")
    scorer_paths = [resolve_ref(root, ref)
                    for ref in admission["official_scorer_refs"]]
    resolve_ref(root, admission["candidate_spec_ref"])

    identity_path = resolve_ref(root, admission["generation_identity_ref"])
    identity = read_json(identity_path)
    if (not isinstance(identity, dict)
            or identity.get("producer_source_refs") !=
            admission["implementation_refs"]):
        raise ValueError("generation identity differs from admitted implementation")

    harness_path = resolve_ref(root, admission["harness_attempt_ref"])
    harness = read_json(harness_path)
    harness_keys = {
        "kind", "version", "design_digest", "candidate_id", "control_role",
        "uid", "generation_seed", "status", "max_attempts", "max_retries",
        "implementation_refs", "sequence_ref", "harness_plan_ref",
        "harness_report_ref", "native_plan_ref", "native_receipt_ref",
        "approved_plan_digest", "completed_at", "attempt_digest",
    }
    if (not isinstance(harness, dict) or set(harness) != harness_keys
            or harness.get("kind") != HARNESS_ATTEMPT_KIND
            or harness.get("version") != DESIGN_VERSION
            or harness.get("design_digest") != design["design_digest"]
            or harness.get("candidate_id") != row["candidate_id"]
            or harness.get("control_role") != role
            or harness.get("uid") != uid
            or harness.get("generation_seed") != seed
            or harness.get("status") != "completed_once"
            or harness.get("max_attempts") != 1
            or harness.get("max_retries") != 0
            or harness.get("implementation_refs") != admission["implementation_refs"]
            or harness.get("sequence_ref") != sequence_ref
            or harness.get("attempt_digest") != canonical_record_digest(
                harness, "attempt_digest")):
        raise ValueError("exact single-attempt native harness receipt required")
    if _utc_timestamp(harness["completed_at"], "harness completed_at") > admitted_at:
        raise ValueError("native admission precedes harness completion")
    raw_harness_keys = ("harness_plan_ref", "harness_report_ref",
                        "native_plan_ref", "native_receipt_ref")
    raw_refs = [harness[key] for key in raw_harness_keys]
    if len({_canonical_bytes(ref) for ref in raw_refs}) != len(raw_refs):
        raise ValueError("distinct canonical harness records required")
    harness_plan_path, harness_report_path, native_plan_path, native_receipt_path = [
        resolve_ref(root, ref) for ref in raw_refs]
    harness_plan, harness_report, native_plan, native_receipt = [
        read_json(path) for path in (harness_plan_path, harness_report_path,
                                    native_plan_path, native_receipt_path)]
    approved = harness.get("approved_plan_digest")
    tasks = harness_plan.get("tasks", []) if isinstance(harness_plan, dict) else []
    jobs = native_plan.get("jobs", []) if isinstance(native_plan, dict) else []
    attempts = native_receipt.get("attempts", []) \
        if isinstance(native_receipt, dict) else []
    limits = native_plan.get("limits", {}) if isinstance(native_plan, dict) else {}
    provenance = native_plan.get("provenance", {}) \
        if isinstance(native_plan, dict) else {}
    provenance_unit = provenance.get("g01_control_unit", {}) \
        if isinstance(provenance, dict) else {}
    job_code_refs = jobs[0].get("code_refs", []) if len(jobs) == 1 else []
    validate_native_job_code_refs(
        admission["implementation_refs"], admission["official_scorer_refs"],
        job_code_refs)
    expected_unit = {
        "candidate_id": row["candidate_id"], "control_role": role,
        "uid": uid, "generation_seed": seed,
    }
    native_digest = native_plan.get("plan_digest") \
        if isinstance(native_plan, dict) else None
    if (not isinstance(approved, str) or len(approved) != 64
            or harness_plan.get("plan_digest") != approved
            or canonical_record_digest(harness_plan, "plan_digest") != approved
            or harness_report.get("plan_digest") != approved
            or harness_report.get("status") != "completed"
            or len(tasks) != 1
            or tasks[0].get("plan_ref") != harness["native_plan_ref"]
            or native_digest != canonical_record_digest(native_plan, "plan_digest")
            or limits.get("max_attempts") != 1
            or limits.get("max_retries_per_trial") != 0
            or provenance_unit != expected_unit
            or len(jobs) != 1
            or jobs[0].get("arm_role") != role
            or native_receipt.get("plan_digest") != native_digest
            or native_receipt.get("status") != "completed"
            or len(attempts) != 1
            or attempts[0].get("status") != "completed"
            or attempts[0].get("output_refs") != [sequence_ref]):
        raise ValueError("canonical completed harness/native origin required")

    producer_path = resolve_ref(root, admission["producer_receipt_ref"])
    producer = read_json(producer_path)
    producer_keys = {
        "kind", "version", "design_digest", "candidate_id", "control_role",
        "uid", "seed", "status", "sequence_ref", "implementation_refs",
        "harness_attempt_ref", "completed_at", "receipt_digest",
    }
    if (not isinstance(producer, dict) or set(producer) != producer_keys
            or producer.get("kind") != PRODUCER_RECEIPT_KIND
            or producer.get("version") != DESIGN_VERSION
            or producer.get("design_digest") != design["design_digest"]
            or producer.get("candidate_id") != row["candidate_id"]
            or producer.get("control_role") != role
            or producer.get("uid") != uid or producer.get("seed") != seed
            or producer.get("status") != "completed_once"
            or producer.get("sequence_ref") != sequence_ref
            or producer.get("implementation_refs") !=
            admission["implementation_refs"]
            or producer.get("harness_attempt_ref") !=
            admission["harness_attempt_ref"]
            or producer.get("receipt_digest") != canonical_record_digest(
                producer, "receipt_digest")):
        raise ValueError("exact admitted producer receipt required")
    producer_completed = _utc_timestamp(
        producer["completed_at"], "producer completed_at")
    if producer_completed > admitted_at:
        raise ValueError("native admission precedes producer completion")

    review_path = resolve_ref(root, admission["review_ref"])
    review = read_json(review_path)
    review_keys = {
        "kind", "version", "admission_payload_digest", "outcome", "checks",
        "reviewer", "reviewed_at", "review_digest",
    }
    if (not isinstance(review, dict) or set(review) != review_keys
            or review.get("kind") != CONTROL_ADMISSION_REVIEW_KIND
            or review.get("version") != DESIGN_VERSION
            or review.get("admission_payload_digest") !=
            _control_admission_payload_digest(admission)
            or review.get("outcome") != "verified"
            or set(review.get("checks", [])) != EXPECTED_ADMISSION_REVIEW_CHECKS
            or not isinstance(review.get("reviewer"), str)
            or not review["reviewer"]
            or review["reviewer"] == admission["author"]
            or review.get("review_digest") != canonical_record_digest(
                review, "review_digest")):
        raise ValueError("independent native admission review required")
    reviewed_at = _utc_timestamp(review["reviewed_at"], "admission reviewed_at")
    if reviewed_at < admitted_at:
        raise ValueError("native admission review precedes admission")
    return ([admission_path, *implementation_paths, *scorer_paths, identity_path,
             harness_path, producer_path, review_path, harness_plan_path,
             harness_report_path, native_plan_path, native_receipt_path],
            reviewed_at)


def _validate_b_star_result(root: Path, design: dict, split_ref: dict,
                            split: dict, result: dict) -> tuple[tuple[str, str], float, list[Path]]:
    expected_keys = {
        "kind", "version", "design_digest", "split_ref", "candidate_id",
        "control_role", "metric", "unit_receipt_refs", "completed_at",
        "result_digest",
    }
    by_id = {row["candidate_id"]: row for row in design["candidates"]}
    candidate_id = result.get("candidate_id") if isinstance(result, dict) else None
    row = by_id.get(candidate_id)
    selectable = ([] if row is None else
                  [role for role in row["control_roles"]
                   if role.lower() != "b_star"])
    if (not isinstance(result, dict) or set(result) != expected_keys
            or result.get("kind") != B_STAR_D1_RESULT_KIND
            or result.get("version") != DESIGN_VERSION
            or result.get("design_digest") != design["design_digest"]
            or result.get("split_ref") != split_ref
            or row is None or result.get("control_role") not in selectable
            or result.get("metric") != row["primary_metric"]
            or not isinstance(result.get("unit_receipt_refs"), list)
            or result.get("result_digest") != canonical_record_digest(
                result, "result_digest")):
        raise ValueError("invalid complete D1 control result for B_star")
    result_completed_at = _utc_timestamp(
        result["completed_at"], "B_star D1 result completed_at")
    family_by_uid = split.get("family_by_uid")
    d1_ids = split.get("d1_ids")
    d1_families = split.get("d1_families")
    if (not isinstance(family_by_uid, dict) or not isinstance(d1_ids, list)
            or not isinstance(d1_families, list)
            or not d1_ids or not d1_families
            or not set(d1_ids).issubset(family_by_uid)
            or {family_by_uid[uid] for uid in d1_ids} != set(d1_families)):
        raise ValueError("admitted split lacks complete D1 family identity")
    expected_units = {
        (uid, seed) for uid in d1_ids
        for seed in design["randomness"]["generation_seeds"]
    }
    refs = result["unit_receipt_refs"]
    if len(refs) != len(expected_units):
        raise ValueError("D1 control result lacks the full UID x seed denominator")
    unit_values = {}
    source_paths = []
    receipt_keys = {
        "kind", "version", "design_digest", "split_ref", "candidate_id",
        "control_role", "uid", "generation_seed", "metric",
        "official_scorer_seed", "status", "value", "sequence_ref",
        "generation_receipt_ref", "native_admission_ref", "comparison_request_ref",
        "scoring_request_ref", "result_ref", "raw_manifest_ref",
        "raw_evidence_ref", "completed_at", "receipt_digest",
    }
    for ref in refs:
        receipt_path = resolve_ref(root, ref)
        receipt = read_json(receipt_path)
        value = receipt.get("value") if isinstance(receipt, dict) else None
        unit = (receipt.get("uid"), receipt.get("generation_seed")) \
            if isinstance(receipt, dict) else (None, None)
        if (not isinstance(receipt, dict) or set(receipt) != receipt_keys
                or receipt.get("kind") != B_STAR_D1_RECEIPT_KIND
                or receipt.get("version") != DESIGN_VERSION
                or receipt.get("design_digest") != design["design_digest"]
                or receipt.get("split_ref") != split_ref
                or receipt.get("candidate_id") != candidate_id
                or receipt.get("control_role") != result["control_role"]
                or unit not in expected_units
                or receipt.get("metric") != row["primary_metric"]
                or receipt.get("official_scorer_seed") !=
                design["randomness"]["official_scorer_seed"]
                or receipt.get("status") != "qualified_complete"
                or isinstance(value, bool) or not isinstance(value, (int, float))
                or not math.isfinite(value) or value < 0
                or receipt.get("receipt_digest") != canonical_record_digest(
                    receipt, "receipt_digest")):
            raise ValueError("invalid native D1 receipt for B_star")
        if unit in unit_values:
            raise ValueError("duplicate native D1 UID x seed receipt")
        receipt_completed_at = _utc_timestamp(
            receipt["completed_at"], "native D1 receipt completed_at")
        if receipt_completed_at > result_completed_at:
            raise ValueError("native D1 receipt was completed after its result")
        ref_keys = ("sequence_ref", "generation_receipt_ref", "native_admission_ref",
                    "comparison_request_ref", "scoring_request_ref",
                    "result_ref", "raw_manifest_ref", "raw_evidence_ref")
        identities = {
            (receipt[key].get("path"), receipt[key].get("sha256"))
            for key in ref_keys if isinstance(receipt.get(key), dict)
        }
        if len(identities) != len(ref_keys):
            raise ValueError("native D1 receipt requires eight distinct source files")
        resolved = {key: resolve_ref(root, receipt[key]) for key in ref_keys}
        generation = read_json(resolved["generation_receipt_ref"])
        generation_keys = {
            "kind", "version", "design_digest", "uid", "seed", "status",
            "sequence_ref", "producer_receipt_ref", "generation_identity_ref",
            "completed_at", "receipt_digest",
        }
        if (not isinstance(generation, dict) or set(generation) != generation_keys
                or generation.get("kind") != GENERATION_UNIT_KIND
                or generation.get("version") != DESIGN_VERSION
                or generation.get("design_digest") != design["design_digest"]
                or generation.get("status") != "qualified_complete"
                or generation.get("uid") != receipt["uid"]
                or generation.get("seed") != receipt["generation_seed"]
                or generation.get("sequence_ref") != receipt["sequence_ref"]
                or generation.get("receipt_digest") != canonical_record_digest(
                    generation, "receipt_digest")):
            raise ValueError("native D1 receipt has no matching generation receipt")
        generation_completed = _utc_timestamp(
            generation["completed_at"], "generation unit completed_at")
        if generation_completed > result_completed_at:
            raise ValueError("generation unit was completed after D1 result")
        producer_path = resolve_ref(root, generation["producer_receipt_ref"])
        producer = read_json(producer_path)
        identity_path = resolve_ref(root, generation["generation_identity_ref"])
        identity = read_json(identity_path)
        identity_keys = {
            "kind", "version", "benchmark_id", "benchmark_revision", "uid",
            "seed", "producer_source_refs", "identity_digest",
        }
        if (not isinstance(identity, dict) or set(identity) != identity_keys
                or identity.get("kind") != GENERATION_IDENTITY_KIND
                or identity.get("version") != DESIGN_VERSION
                or identity.get("benchmark_id") != BENCHMARK_ID
                or identity.get("benchmark_revision") != BENCHMARK_REVISION
                or identity.get("uid") != receipt["uid"]
                or identity.get("seed") != receipt["generation_seed"]
                or not isinstance(identity.get("producer_source_refs"), list)
                or not identity["producer_source_refs"]
                or identity.get("identity_digest") != canonical_record_digest(
                    identity, "identity_digest")):
            raise ValueError("exact qualified generation identity required")
        identity_sources = [resolve_ref(root, ref)
                            for ref in identity["producer_source_refs"]]
        admission_paths, admission_reviewed_at = _validate_control_admission(
            root, design, row, receipt["control_role"], receipt["uid"],
            receipt["generation_seed"], receipt["sequence_ref"],
            receipt["native_admission_ref"])
        admission = read_json(resolved["native_admission_ref"])
        if (generation["producer_receipt_ref"] !=
                admission["producer_receipt_ref"]
                or generation["generation_identity_ref"] !=
                admission["generation_identity_ref"]):
            raise ValueError("generation wrapper differs from native admission")
        comparison = read_json(resolved["comparison_request_ref"])
        comparison_keys = {
            "kind", "version", "design_digest", "candidate_id", "uid",
            "inference_seed", "scoring_seed", "roles", "request_digest",
        }
        comparison_core = ({key: item for key, item in comparison.items()
                            if key != "request_digest"}
                           if isinstance(comparison, dict) else {})
        comparison_roles = comparison.get("roles", []) \
            if isinstance(comparison, dict) else []
        comparison_matches = [item for item in comparison_roles
                              if isinstance(item, dict)
                              and str(item.get("role", "")).lower() ==
                              str(receipt["control_role"]).lower()]
        if (not isinstance(comparison, dict) or set(comparison) != comparison_keys
                or comparison.get("kind") != COMPARISON_REQUEST_KIND
                or comparison.get("version") != DESIGN_VERSION
                or comparison.get("design_digest") != design["design_digest"]
                or comparison.get("candidate_id") != candidate_id
                or comparison.get("uid") != receipt["uid"]
                or comparison.get("inference_seed") != receipt["generation_seed"]
                or comparison.get("scoring_seed") !=
                design["randomness"]["official_scorer_seed"]
                or comparison.get("request_digest") !=
                hashlib.sha256(_canonical_bytes(comparison_core)).hexdigest()
                or not comparison_roles
                or any(not isinstance(item, dict)
                       or set(item) != {"role", "sequence_ref",
                                       "native_admission_ref"}
                       for item in comparison_roles)
                or len(comparison_matches) != 1
                or comparison_matches[0].get("sequence_ref") !=
                receipt["sequence_ref"]
                or comparison_matches[0].get("native_admission_ref") !=
                receipt["native_admission_ref"]):
            raise ValueError("native D1 receipt disagrees with comparison request")
        scoring = read_json(resolved["scoring_request_ref"])
        scoring_keys = {
            "kind", "version", "design_digest", "candidate_id", "uid",
            "comparison_request_digest", "metrics", "official_scorer_refs",
            "request_digest",
        }
        scoring_core = ({key: item for key, item in scoring.items()
                         if key != "request_digest"}
                        if isinstance(scoring, dict) else {})
        if (not isinstance(scoring, dict) or set(scoring) != scoring_keys
                or scoring.get("kind") != SCORING_REQUEST_KIND
                or scoring.get("version") != DESIGN_VERSION
                or scoring.get("design_digest") != design["design_digest"]
                or scoring.get("candidate_id") != candidate_id
                or scoring.get("uid") != receipt["uid"]
                or scoring.get("metrics") != list(METRICS)
                or scoring.get("official_scorer_refs") !=
                design["benchmark"]["official_scorer"]["source_refs"]
                or scoring.get("comparison_request_digest") !=
                comparison["request_digest"]
                or scoring.get("request_digest") !=
                hashlib.sha256(_canonical_bytes(scoring_core)).hexdigest()):
            raise ValueError("native D1 receipt disagrees with scoring request")
        scored = read_json(resolved["result_ref"])
        readout = scored.get("readout") if isinstance(scored, dict) else None
        roles = readout.get("roles", []) if isinstance(readout, dict) else []
        matching_roles = [item for item in roles if isinstance(item, dict)
                          and str(item.get("role", "")).lower() ==
                          str(receipt["control_role"]).lower()]
        metrics = matching_roles[0].get("metrics") \
            if len(matching_roles) == 1 else None
        metric_value = metrics.get(receipt["metric"]) \
            if isinstance(metrics, dict) else None
        bundle = scored.get("raw_bundle") if isinstance(scored, dict) else None
        manifest = bundle.get("manifest") if isinstance(bundle, dict) else None
        archive = bundle.get("archive") if isinstance(bundle, dict) else None
        scored_keys = {
            "kind", "version", "design_digest", "candidate_id", "uid",
            "status", "request_digest", "comparison_request_digest",
            "readout", "raw_bundle", "completed_at", "result_digest",
        }
        scored_completed_at = (_utc_timestamp(
            scored["completed_at"], "official score completed_at")
            if isinstance(scored, dict) and "completed_at" in scored
            else None)
        if (not isinstance(scored, dict) or set(scored) != scored_keys
                or scored.get("kind") != OFFICIAL_RESULT_KIND
                or scored.get("version") != DESIGN_VERSION
                or scored.get("design_digest") != design["design_digest"]
                or scored.get("candidate_id") != candidate_id
                or scored.get("uid") != receipt["uid"]
                or scored.get("request_digest") != scoring["request_digest"]
                or scored.get("comparison_request_digest") !=
                comparison["request_digest"]
                or scored.get("status") not in (
                    "execution_completed", "execution_completed_with_failures")
                or not isinstance(readout, dict)
                or set(readout) != {"roles"}
                or any(not isinstance(item, dict)
                       or set(item) != {"role", "status", "metrics"}
                       or not isinstance(item.get("metrics"), dict)
                       or set(item["metrics"]) != set(METRICS)
                       for item in roles)
                or len(matching_roles) != 1
                or matching_roles[0].get("status") != "success"
                or isinstance(metric_value, bool)
                or not isinstance(metric_value, (int, float))
                or not math.isfinite(metric_value)
                or float(metric_value) != float(value)
                or not isinstance(manifest, dict)
                or set(bundle) != {"manifest", "archive"}
                or set(manifest) != {"sha256"}
                or manifest.get("sha256") !=
                receipt["raw_manifest_ref"]["sha256"]
                or not isinstance(archive, dict)
                or set(archive) != {"sha256"}
                or archive.get("sha256") !=
                receipt["raw_evidence_ref"]["sha256"]
                or scored.get("result_digest") != canonical_record_digest(
                    scored, "result_digest")
                or scored_completed_at is None
                or admission_reviewed_at > scored_completed_at
                or scored_completed_at > receipt_completed_at):
            raise ValueError("native D1 receipt disagrees with official score result")
        _validate_raw_bundle(
            resolved["raw_manifest_ref"], resolved["raw_evidence_ref"],
            scoring, comparison)
        source_paths.append(receipt_path)
        source_paths.extend([producer_path, identity_path, *identity_sources,
                             *admission_paths])
        source_paths.extend(resolved.values())
        unit_values[unit] = float(value)
    if set(unit_values) != expected_units:
        raise ValueError("native D1 receipts do not cover the full UID x seed denominator")
    uid_values = {
        uid: math.fsum(unit_values[(uid, seed)]
                       for seed in design["randomness"]["generation_seeds"]) /
        len(design["randomness"]["generation_seeds"])
        for uid in d1_ids
    }
    family_values = {}
    for family in d1_families:
        family_uids = [uid for uid in d1_ids if family_by_uid[uid] == family]
        if not family_uids:
            raise ValueError("D1 family has no admitted UID")
        family_values[family] = math.fsum(uid_values[uid] for uid in family_uids) / \
            len(family_uids)
    mean = math.fsum(family_values.values()) / len(family_values)
    return (candidate_id, result["control_role"]), mean, source_paths


def _b_star_payload_digest(freeze: dict) -> str:
    keys = (
        "design_digest", "split_ref", "selected_by_candidate",
        "d1_result_refs", "selection_rule", "author", "frozen_at",
    )
    return hashlib.sha256(_canonical_bytes(
        {key: freeze.get(key) for key in keys})).hexdigest()


def _validate_raw_bundle(manifest_path: Path, archive_path: Path,
                         scoring: dict, comparison: dict) -> None:
    manifest = read_json(manifest_path)
    rows = manifest.get("files") if isinstance(manifest, dict) else None
    bundle_ref = manifest.get("bundle_ref") if isinstance(manifest, dict) else None
    archive_ref = bundle_ref.get("archive") if isinstance(bundle_ref, dict) else None
    if (not isinstance(manifest, dict)
            or set(manifest) != {"kind", "version", "request_digest",
                                 "comparison_request_digest", "files",
                                 "bundle_ref"}
            or manifest.get("kind") != RAW_MANIFEST_KIND
            or manifest.get("version") != DESIGN_VERSION
            or not isinstance(rows, list) or not rows or len(rows) > 20000
            or not isinstance(bundle_ref, dict)
            or set(bundle_ref) != {"archive", "raw_file_count"}
            or manifest.get("request_digest") != scoring["request_digest"]
            or manifest.get("comparison_request_digest") !=
            comparison["request_digest"]
            or bundle_ref.get("raw_file_count") != len(rows)
            or not isinstance(archive_ref, dict)
            or set(archive_ref) != {"sha256"}
            or archive_ref.get("sha256") != _file_digest(archive_path)):
        raise ValueError("invalid official raw manifest identity")
    inventory = {}
    for row in rows:
        if (not isinstance(row, dict)
                or set(row) != {"path", "bytes", "sha256"}
                or not isinstance(row["path"], str)
                or not isinstance(row["bytes"], int) or row["bytes"] < 0
                or not isinstance(row["sha256"], str)
                or len(row["sha256"]) != 64):
            raise ValueError("invalid official raw manifest row")
        member = PurePosixPath(row["path"])
        if (member.is_absolute() or not member.parts
                or any(part in ("", ".", "..") for part in member.parts)
                or row["path"] in inventory):
            raise ValueError("unsafe or duplicate official raw member")
        inventory[row["path"]] = row
    seen = set()
    with tarfile.open(archive_path, mode="r:") as archive:
        for member in archive:
            if (not member.isfile() or member.name not in inventory
                    or member.name in seen):
                raise ValueError("official raw archive inventory mismatch")
            row = inventory[member.name]
            if member.size != row["bytes"]:
                raise ValueError("official raw archive member size mismatch")
            stream = archive.extractfile(member)
            if stream is None:
                raise ValueError("official raw archive member unavailable")
            digest = hashlib.sha256()
            with stream:
                while True:
                    chunk = stream.read(1024 * 1024)
                    if not chunk:
                        break
                    digest.update(chunk)
            if digest.hexdigest() != row["sha256"]:
                raise ValueError("official raw archive member digest mismatch")
            seen.add(member.name)
    if seen != set(inventory):
        raise ValueError("official raw archive is incomplete")


def validate_b_star_freeze(root: Path, design: dict, freeze: dict,
                           expected_split: dict | None = None) -> list[Path]:
    """Validate the prospective D1-only B* freeze and its independent review."""
    root = Path(root).resolve()
    expected_keys = {
        "kind", "version", "design_digest", "split_ref",
        "selected_by_candidate", "d1_result_refs", "selection_rule",
        "author", "review_ref", "frozen_at", "freeze_digest",
    }
    if (not isinstance(freeze, dict) or set(freeze) != expected_keys
            or freeze.get("kind") != B_STAR_FREEZE_KIND
            or freeze.get("version") != DESIGN_VERSION
            or freeze.get("design_digest") != design["design_digest"]
            or freeze.get("selection_rule") !=
            "min_primary_family_mean_over_all_qualified_controls_on_D1_lexical_tie_break"
            or not isinstance(freeze.get("author"), str) or not freeze["author"]
            or freeze.get("freeze_digest") != canonical_record_digest(
                freeze, "freeze_digest")):
        raise ValueError("invalid prospective B_star freeze")
    frozen_at = _utc_timestamp(freeze["frozen_at"], "B_star frozen_at")
    validate_b_star_selections(design, freeze.get("selected_by_candidate"))
    split_path = resolve_ref(root, freeze["split_ref"])
    split = read_json(split_path)
    if (split.get("kind") != SPLIT_KIND
            or split.get("design_digest") != design["design_digest"]
            or split.get("split_digest") != canonical_record_digest(
                split, "split_digest")
            or not split.get("d1_ids") or not split.get("confirmation_ids")):
        raise ValueError("B_star freeze requires the exact admitted family split")
    if expected_split is not None and split != expected_split:
        raise ValueError("B_star freeze split differs from current family evidence")
    result_refs = freeze.get("d1_result_refs")
    expected_result_count = sum(
        sum(role.lower() != "b_star" for role in row["control_roles"])
        for row in design["candidates"])
    if (not isinstance(result_refs, list)
            or len(result_refs) != expected_result_count):
        raise ValueError("complete D1 result references required for B_star")
    result_paths = []
    source_paths = []
    result_means = {}
    for ref in result_refs:
        result_path = resolve_ref(root, ref)
        result = read_json(result_path)
        key, mean, sources = _validate_b_star_result(
            root, design, freeze["split_ref"], split, result)
        if _utc_timestamp(
                result["completed_at"], "B_star D1 result completed_at") > frozen_at:
            raise ValueError("B_star D1 result was completed after freeze")
        if key in result_means:
            raise ValueError("duplicate D1 control result for B_star")
        result_means[key] = mean
        result_paths.append(result_path)
        source_paths.extend(sources)
    expected_pairs = {
        (row["candidate_id"], role)
        for row in design["candidates"]
        for role in row["control_roles"] if role.lower() != "b_star"
    }
    if set(result_means) != expected_pairs:
        raise ValueError("D1 results do not cover every selectable control")
    expected_selections = {}
    for row in design["candidates"]:
        candidate_id = row["candidate_id"]
        roles = [role for role in row["control_roles"]
                 if role.lower() != "b_star"]
        selected = min(roles, key=lambda role: (result_means[(candidate_id, role)], role))
        expected_selections[candidate_id] = {
            "role": selected,
            "selection_stage": "D1_only",
            "metric": row["primary_metric"],
        }
    if freeze["selected_by_candidate"] != expected_selections:
        raise ValueError("B_star selection is not the deterministic D1 minimum")
    review_path = resolve_ref(root, freeze["review_ref"])
    review = read_json(review_path)
    review_keys = {
        "kind", "version", "selection_payload_digest", "outcome", "checks",
        "reviewer", "reviewed_at", "review_digest",
    }
    if (not isinstance(review, dict) or set(review) != review_keys
            or review.get("kind") != B_STAR_REVIEW_KIND
            or review.get("version") != DESIGN_VERSION
            or review.get("selection_payload_digest") !=
            _b_star_payload_digest(freeze)
            or review.get("outcome") != "verified"
            or set(review.get("checks", [])) != {
                "complete_control_inventory", "D1_only",
                "native_metric_identity", "deterministic_selection",
                "no_confirmation_access"}
            or not isinstance(review.get("reviewer"), str)
            or not review["reviewer"]
            or review["reviewer"] == freeze["author"]
            or review.get("review_digest") != canonical_record_digest(
                review, "review_digest")):
        raise ValueError("independent verified B_star freeze review required")
    if _utc_timestamp(review["reviewed_at"], "B_star reviewed_at") < frozen_at:
        raise ValueError("B_star review precedes freeze")
    return [split_path, *result_paths, *source_paths, review_path]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Validate the stage-wide G01 design and reviewed family evidence")
    parser.add_argument("--root", required=True)
    parser.add_argument("--design", required=True)
    parser.add_argument("--family-evidence")
    parser.add_argument("--b-star-selections")
    parser.add_argument("--output-split")
    args = parser.parse_args(argv)
    if args.b_star_selections and not args.family_evidence:
        raise ValueError("--b-star-selections requires --family-evidence")
    root = Path(args.root).resolve()
    design = validate_design(root, Path(args.design))
    output = {
        "status": "design_validated_generated_unexecuted",
        "design_digest": design["design_digest"],
        "candidate_count": len(design["candidates"]),
        "confirmatory_test_count": design["statistics"]["confirmatory_test_count"],
        "gpu_dispatch_allowed": False,
    }
    if args.family_evidence:
        evidence = read_json(Path(args.family_evidence))
        split = derive_family_split(root, design, evidence)
        output["family_split_digest"] = split["split_digest"]
        output["confirmation_family_count"] = len(split["confirmation_families"])
        if args.output_split:
            target = Path(args.output_split).resolve()
            try:
                target.relative_to(root)
            except ValueError as error:
                raise ValueError("output split escapes root") from error
            if target.exists() or target.is_symlink():
                raise ValueError("refusing to overwrite family split")
            target.parent.mkdir(parents=True, exist_ok=True)
            with target.open("x") as stream:
                stream.write(json.dumps(split, indent=2, sort_keys=True) + "\n")
    elif args.output_split:
        raise ValueError("--output-split requires --family-evidence")
    if args.b_star_selections:
        validate_b_star_freeze(
            root, design, read_json(Path(args.b_star_selections)),
            expected_split=split)
        output["b_star_freeze_validated"] = True
    print(json.dumps(output, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

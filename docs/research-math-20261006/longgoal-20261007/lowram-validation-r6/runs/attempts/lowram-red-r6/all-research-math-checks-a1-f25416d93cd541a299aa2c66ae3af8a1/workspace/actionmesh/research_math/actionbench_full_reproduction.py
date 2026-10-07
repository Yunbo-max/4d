"""Fail-closed contract checks for the full ActionBench reproduction path.

This module does not run ActionMesh or the official scorer.  It separates the
published ActionMesh generation seed (42) from the official evaluator sampling
seed (44), requires the complete released 128-object population, and evaluates
only whether a returned official aggregate reproduces the current official
README values at their published precision.  It never qualifies a candidate
effect and is not a dispatch plan.
"""
from __future__ import annotations

from decimal import Decimal
import math


METRICS = ("cd_3d", "cd_4d", "cd_motion")
DECISION_RULE = {
    "complete_population_required": True,
    "required_total": 128,
    "required_successes": 128,
    "required_failures": 0,
    "metric_rule": "published_rounding_interval",
    "all_metrics_required": True,
    "uncertainty": "finite_population_reproduction_no_sampling_ci",
}


def published_interval(value: float, decimal_places: int) -> tuple[float, float]:
    """Return the half-open decimal bin represented by a published value."""
    if isinstance(decimal_places, bool) or not isinstance(decimal_places, int):
        raise ValueError("Integer published decimal precision required")
    if decimal_places < 0:
        raise ValueError("Nonnegative published decimal precision required")
    target = Decimal(str(value))
    half_unit = Decimal(5).scaleb(-(decimal_places + 1))
    return float(target - half_unit), float(target + half_unit)


def validate_contract_data(contract: dict, population: dict, audit: dict) -> None:
    """Validate the source-scoped contract without executing project code."""
    if contract.get("kind") != "actionbench-full-population-reproduction-contract":
        raise ValueError("Full-population ActionBench reproduction contract required")
    if contract.get("version") != "1.0.0":
        raise ValueError("Unsupported reproduction contract version")
    if contract.get("purpose") != "official-full-population-baseline-reproduction":
        raise ValueError("Baseline reproduction purpose required")

    benchmark = contract.get("benchmark", {})
    uids = population.get("uids")
    if (population.get("dataset") != "facebook/actionbench" or
            not isinstance(uids, list) or len(uids) != 128 or
            len(set(uids)) != 128 or any(not isinstance(uid, str) or not uid for uid in uids)):
        raise ValueError("Exactly 128 unique released ActionBench UIDs required")
    if (benchmark.get("id"), benchmark.get("revision"),
            benchmark.get("population_size"), benchmark.get("frames_per_sample")) != (
            population["dataset"], population.get("revision"), 128, 16):
        raise ValueError("Contract benchmark identity differs from released population")

    if contract.get("method", {}).get("generation_seed") != 42:
        raise ValueError("Published ActionMesh generation seed 42 required")
    scorer = contract.get("official_scorer", {})
    if scorer.get("sampling_seed") != 44:
        raise ValueError("Official evaluator sampling seed 44 required")
    if (scorer.get("n_pts_icp"), scorer.get("n_pts_chamfer"),
            scorer.get("metrics")) != (10000, 100000, list(METRICS)):
        raise ValueError("Official scorer budgets/metric order differ")

    published = audit.get("published_aggregate_contract", {})
    target = contract.get("published_target", {})
    if (published.get("population") != "all 128 animated objects" or
            published.get("actionmesh_seed") != 42):
        raise ValueError("Audit does not bind the full published ActionMesh population")
    if (target.get("source_path") != published.get("source_path") or
            target.get("metrics") != published.get("metrics") or
            target.get("decimal_places") != 3):
        raise ValueError("Contract must use the current official README values and precision")

    if contract.get("decision_rule") != DECISION_RULE:
        raise ValueError("Exact prospective full-population decision rule required")
    if (contract.get("scientific_effect_qualification") is not False or
            contract.get("candidate_methods_tested") is not False or
            contract.get("dispatch_ready") is not False):
        raise ValueError("Reproduction contract cannot claim an effect or dispatch readiness")


def _finite_metric(row: dict, metric: str) -> float:
    value = row.get(metric)
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError("Every successful row requires finite official metrics")
    return float(value)


def assess_official_reproduction(contract: dict, population: dict,
                                 rows: list[dict], summary: dict) -> dict:
    """Check a complete official CSV/summary pair against the frozen contract.

    Per-object metrics must already come from the official scorer.  Recomputing
    their arithmetic means here verifies the official output bundle's internal
    consistency; it does not replace per-object native scoring or trusted replay.
    """
    if contract.get("decision_rule") != DECISION_RULE:
        raise ValueError("Exact prospective full-population decision rule required")
    if (contract.get("method", {}).get("generation_seed") != 42 or
            contract.get("official_scorer", {}).get("sampling_seed") != 44 or
            contract.get("official_scorer", {}).get("metrics") != list(METRICS) or
            contract.get("scientific_effect_qualification") is not False or
            contract.get("candidate_methods_tested") is not False or
            contract.get("dispatch_ready") is not False):
        raise ValueError("Frozen reproduction identity/boundary required")
    uids = population.get("uids", [])
    if len(rows) != 128 or {row.get("uid") for row in rows} != set(uids):
        raise ValueError("Official output must cover the complete released population")
    if (summary.get("n_total"), summary.get("n_success"), summary.get("n_failed")) != (
            128, 128, 0):
        raise ValueError("Official output requires 128 successful samples and zero failures")
    if any(row.get("status") != "success" or row.get("n_frames") != 16 for row in rows):
        raise ValueError("Official output requires 128 successful 16-frame rows")

    decimal_places = contract["published_target"]["decimal_places"]
    targets = contract["published_target"]["metrics"]
    checks = {}
    for metric in METRICS:
        observed = math.fsum(_finite_metric(row, metric) for row in rows) / 128
        summary_key = metric + "_mean"
        recorded = summary.get(summary_key)
        if (isinstance(recorded, bool) or not isinstance(recorded, (int, float)) or
                not math.isfinite(recorded) or
                not math.isclose(observed, float(recorded), rel_tol=0.0, abs_tol=1e-12)):
            raise ValueError("Official CSV and summary aggregate differ: " + metric)
        lower, upper = published_interval(targets[metric], decimal_places)
        checks[metric] = {
            "published": targets[metric],
            "observed": observed,
            "lower_inclusive": lower,
            "upper_exclusive": upper,
            "passed": lower <= observed < upper,
        }

    passed = all(check["passed"] for check in checks.values())
    return {
        "decision": ("reproduced_within_published_precision" if passed
                     else "not_reproduced"),
        "metric_checks": checks,
        "population_size": 128,
        "generation_seed": 42,
        "scorer_sampling_seed": 44,
        "uncertainty": "finite_population_reproduction_no_sampling_ci",
        "scientific_effect_qualification": False,
        "candidate_methods_tested": False,
        "requires_live_official_replay": True,
    }

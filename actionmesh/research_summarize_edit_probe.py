#!/usr/bin/env python3
"""Summarize edit-probe seeds; synthetic diagnostics are never natural quality GT."""
from __future__ import annotations

import argparse
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path


METHODS = ["world_offset", "global_rigid_transport", "local_rigid_transport", "residual_gated_local_global"]
REGIMES = ["global_rigid", "two_region_articulation", "fast_two_region_articulation", "global_rigid_noisy_observations"]
CASES = {"kangaroo", "octopus", "mushroom", "panda"}
MATCH_CONFIG = ["points", "neighbors", "eval_neighbors", "edit_amplitude", "observation_noise", "risk_threshold"]
PRIMARY = "edit_offset_epe_over_scale"


def summary(values):
    values = [float(v) for v in values if v is not None and math.isfinite(float(v))]
    return {"mean": statistics.fmean(values) if values else None,
            "std_across_case_seed_rows_descriptive": statistics.pstdev(values) if values else None,
            "n": len(values)}


def contrast(candidate, alternatives, rows, zero_tol):
    """Compare aggregate means; the selected best control is a descriptive oracle."""
    means = {m: summary([r[m] for r in rows])["mean"] for m in [candidate, *alternatives]}
    valid = [m for m in alternatives if means[m] is not None]
    if means[candidate] is None or not valid:
        return {"available": False}
    best = min(valid, key=lambda m: means[m])
    paired = [r[best] - r[candidate] for r in rows if r.get(best) is not None and r.get(candidate) is not None]
    base, value = means[best], means[candidate]
    gain = base - value
    return {"candidate": candidate, "best_aggregate_control": best,
            "control_selection": "descriptive best of declared alternatives; not a validated deployable selector",
            "candidate_mean": value, "control_mean": base,
            "absolute_gain_positive_is_better": gain,
            "relative_gain_fraction": gain / base if base > zero_tol else None,
            "relative_gain_null_reason": None if base > zero_tol else "control is at numerical zero; percentage is unstable",
            "paired_absolute_gains": summary(paired),
            "paired_wins": sum(x > zero_tol for x in paired),
            "paired_losses": sum(x < -zero_tol for x in paired),
            "paired_ties": sum(abs(x) <= zero_tol for x in paired)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", nargs="+", type=Path, required=True, help="results.json files or run directories")
    parser.add_argument("--output", type=Path, required=True, help="fresh summary JSON path")
    parser.add_argument("--expected-seeds", default="0,1,2")
    parser.add_argument("--zero-tolerance", type=float, default=1e-7)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("output already exists; use a fresh path")
    expected = {int(x) for x in args.expected_seeds.split(",")}
    groups, natural, risk, per_seed, contrasts = defaultdict(list), defaultdict(list), defaultdict(list), defaultdict(list), defaultdict(list)
    result = {"scope": "aggregate of output-space edit diagnostics; NO natural-ground-truth quality claim",
              "primary_metric": PRIMARY, "expected_seeds": sorted(expected), "runs": [], "issues": [],
              "synthetic": {}, "natural_descriptive_only": {}, "interpretation": [
                  "Synthetic target comes from analytical transforms of generated anchor points, not natural 4D ground truth.",
                  "Global-rigid recovery and norm preservation are algebraic sanity checks, not evidence of novelty.",
                  "The fast articulation uses the same sampled poses in a different order on this 16-frame grid; frame-independent Kabsch cannot establish temporal-speed robustness from it.",
                  "Seeds resample the same four assets; case/vertex/time observations are correlated. No significance or calibration claim.",
                  "Natural heldout-edge drift and velocity changes do not establish natural edit correctness.",
                  "Best-control comparisons are descriptive selection among declared controls, with no post-hoc tuning of candidate parameters."]}
    seen, hashes, config = set(), {}, None
    for supplied in args.inputs:
        path = supplied / "results.json" if supplied.is_dir() else supplied
        data = json.loads(path.read_text())
        seed = int(data["seed"])
        if seed in seen:
            parser.error(f"duplicate seed {seed}")
        seen.add(seed)
        selected = {k: data["configuration"].get(k) for k in MATCH_CONFIG}
        if config is None:
            config = selected
        elif selected != config:
            result["issues"].append({"seed": seed, "issue": "configuration_mismatch", "configuration": selected})
        if data.get("status") != "completed_diagnostic_only" or data.get("failures"):
            result["issues"].append({"seed": seed, "issue": "run_failed_or_incomplete", "failures": data.get("failures", {})})
        result["runs"].append({"path": str(path), "seed": seed, "status": data.get("status"),
                               "elapsed_seconds": data.get("elapsed_seconds"), "device": data.get("device"),
                               "torch_peak_allocated_bytes": data.get("torch_peak_allocated_bytes"),
                               "torch_peak_reserved_bytes": data.get("torch_peak_reserved_bytes")})
        present = set(data.get("cases", {}))
        if present != CASES:
            result["issues"].append({"seed": seed, "issue": "case_set_mismatch", "present": sorted(present)})
        for case, card in data.get("cases", {}).items():
            digest = card.get("input_sha256")
            if case in hashes and hashes[case] != digest:
                result["issues"].append({"seed": seed, "case": case, "issue": "source_hash_mismatch"})
            hashes[case] = digest
            for method in METHODS:
                metrics = card.get("natural_track_diagnostics", {}).get(method, {})
                for metric, value in metrics.items():
                    if isinstance(value, (int, float)) and not isinstance(value, bool):
                        natural[(method, metric)].append(value)
            for regime in REGIMES:
                control = card.get("synthetic_controls", {}).get(regime)
                if control is None:
                    result["issues"].append({"seed": seed, "case": case, "regime": regime, "issue": "missing_regime"})
                    continue
                risk[regime].append(control.get("local_risk_auc_predicting_harm_vs_global"))
                paired = {m: control.get("methods", {}).get(m, {}).get(PRIMARY) for m in METHODS}
                if any(v is None for v in paired.values()):
                    result["issues"].append({"seed": seed, "case": case, "regime": regime, "issue": "missing_primary_method_metric"})
                contrasts[regime].append(paired)
                for method, metrics in control.get("methods", {}).items():
                    for metric, value in metrics.items():
                        if (value is None or isinstance(value, (int, float))) and not isinstance(value, bool):
                            groups[(regime, method, metric)].append(value)
                    per_seed[(regime, method, seed)].append(metrics.get(PRIMARY))
    if seen != expected:
        result["issues"].append({"issue": "seed_set_mismatch", "seen": sorted(seen)})
    result["configuration"] = config
    result["source_sha256"] = hashes
    for regime in REGIMES:
        entry = {"methods": {}, "local_fit_risk_auc_descriptive": summary(risk[regime])}
        for method in METHODS:
            entry["methods"][method] = {metric: summary(values) for (r, m, metric), values in groups.items() if r == regime and m == method}
            entry["methods"][method]["primary_mean_by_seed"] = {str(s): summary(per_seed[(regime, method, s)]) for s in sorted(seen)}
        entry["local_vs_best_world_or_global"] = contrast("local_rigid_transport", METHODS[:2], contrasts[regime], args.zero_tolerance)
        entry["gated_vs_best_ungated"] = contrast("residual_gated_local_global", METHODS[:3], contrasts[regime], args.zero_tolerance)
        result["synthetic"][regime] = entry
    for method in METHODS:
        result["natural_descriptive_only"][method] = {metric: summary(values) for (m, metric), values in natural.items() if m == method}
    result["status"] = "complete_descriptive_aggregation" if not result["issues"] else "incomplete_or_incomparable"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False))
    print("regime | method | synthetic edit-offset EPE / fixed scale")
    for regime, entry in result["synthetic"].items():
        for method, metrics in entry["methods"].items():
            value = metrics.get(PRIMARY, {}).get("mean")
            print(f"{regime} | {method} | {value}")
        for name in ("local_vs_best_world_or_global", "gated_vs_best_ungated"):
            c = entry[name]
            print(f"  {name}: best={c.get('best_aggregate_control')}, absolute_gain={c.get('absolute_gain_positive_is_better')}, relative_gain={c.get('relative_gain_fraction')}")
    print(json.dumps({"status": result["status"], "issues": result["issues"], "output": str(args.output)}))
    if result["issues"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

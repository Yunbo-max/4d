"""Constructed controls plus independent qualified real-input screening."""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
import time
import traceback

from .catalog import METHODS, load_method


def _write(path, value):
    # Reject NaN/Infinity instead of emitting nonstandard, misleading evidence.
    content = json.dumps(value, indent=2, allow_nan=False) + "\n"
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(content)
    temporary.replace(path)


def controls(methods, output):
    if len(set(methods)) != len(methods):
        raise ValueError("duplicate method IDs are not separate trials")
    output.mkdir(parents=True, exist_ok=False)
    report = {"created_utc": datetime.now(timezone.utc).isoformat(),
              "scope": "Constructed software controls; no natural-data efficacy or model training claimed",
              "python": sys.version, "methods": {}}
    failed = False
    for identifier in methods:
        started = time.perf_counter()
        row = {"name": METHODS[identifier]["name"], "status": "running"}
        report["methods"][str(identifier)] = row
        try:
            module = load_method(identifier)
            row["module_sha256"] = hashlib.sha256(Path(module.__file__).read_bytes()).hexdigest()
            results = module.demo()
            if not isinstance(results, dict):
                raise TypeError("demo must return a JSON-safe object")
            # Set this at the orchestration boundary even if a method uses its
            # own more specific fixture labels internally.
            results = dict(results, evidence_type="constructed_control", method=identifier)
            _write(output / f"method-{identifier:02d}.json", results)
            row["status"] = "completed"
        except Exception as error:
            row.update(status="failed", error=str(error), traceback=traceback.format_exc())
            failed = True
        row["elapsed_seconds"] = time.perf_counter()-started
        _write(output/"report.json", report)
        print(json.dumps({"method": identifier, "status": row["status"]}), flush=True)
    return int(failed)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("list", help="List methods and real input requirements without loading models")
    run = sub.add_parser("controls", help="Run analytic/constructed controls, not a benchmark")
    run.add_argument("--methods", type=int, nargs="+", choices=sorted(METHODS), default=sorted(METHODS))
    run.add_argument("--output", type=Path, required=True)
    geometry = sub.add_parser("geometry", help="Run ONE of H1/H5/H8 and its controls on qualified NPZ observations")
    geometry.add_argument("--method", type=int, choices=(1,5,8), required=True)
    geometry.add_argument("--input", type=Path, required=True)
    geometry.add_argument("--evidence", type=Path, required=True)
    geometry.add_argument("--output", type=Path, required=True)
    native = sub.add_parser("guidance", help="Run H9 with matched full 30-step live controls and complete mesh decoding")
    from .native_pair import add_arguments
    add_arguments(native)
    args = parser.parse_args()
    if args.command == "list":
        print(json.dumps(METHODS, indent=2))
        return 0
    try:
        if args.command == "geometry":
            from .geometry_run import run_geometry
            return run_geometry(args.method, args.input, args.evidence, args.output)
        if args.command == "guidance":
            from .native_pair import run_guidance
            return run_guidance(args)
        return controls(args.methods, args.output)
    except (ValueError, FileExistsError) as error:
        parser.error(str(error))


if __name__ == "__main__":
    raise SystemExit(main())

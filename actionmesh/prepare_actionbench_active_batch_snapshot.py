"""Freeze one coherent immutable r9 harness-state/status evidence pair.

This Local-only evidence preparation step performs no inference, scoring,
approval, or dispatch. It double-reads both changing inputs, rejects a moving or
aggregate-inconsistent pair, and writes the original bytes exactly once to the
canonical retained archive consumed by the reconciliation builder.
"""
from __future__ import annotations

import argparse
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import tempfile

from prepare_actionbench_full128_window import (
    R9_RECONCILIATION_PROFILE,
    R9_STATE_SNAPSHOT_PATH,
    R9_STATUS_SNAPSHOT_PATH,
    RETAINED_DISPOSITIONS,
    retained_observation_window,
)


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _ref(root: Path, path: Path, payload: bytes) -> dict:
    return {
        "path": path.resolve().relative_to(root.resolve()).as_posix(),
        "sha256": _sha256(payload),
    }


def _stable_pair(state_path: Path, status_path: Path) -> tuple[bytes, bytes]:
    """Read both live files twice and reject an input that changes in-window."""
    status_first = status_path.read_bytes()
    state_first = state_path.read_bytes()
    status_second = status_path.read_bytes()
    state_second = state_path.read_bytes()
    if status_first != status_second or state_first != state_second:
        raise ValueError("Live harness evidence changed during snapshot")
    return state_first, status_first


def _write_pair(state_output: Path, state_bytes: bytes,
                status_output: Path, status_bytes: bytes) -> None:
    """Publish both byte copies without replacing any retained evidence."""
    if state_output.exists() or status_output.exists():
        raise FileExistsError("Preserve existing active-batch evidence snapshot")
    state_output.parent.mkdir(parents=True, exist_ok=True)
    status_output.parent.mkdir(parents=True, exist_ok=True)
    temporary_paths: list[Path] = []
    published: list[Path] = []
    try:
        for output, payload in ((state_output, state_bytes),
                                (status_output, status_bytes)):
            with tempfile.NamedTemporaryFile(
                    dir=output.parent, prefix=output.name + ".",
                    suffix=".tmp", delete=False) as handle:
                handle.write(payload)
                handle.flush()
                os.fsync(handle.fileno())
                temporary_paths.append(Path(handle.name))
        for temporary, output in zip(temporary_paths,
                                     (state_output, status_output)):
            os.link(temporary, output)
            published.append(output)
    except Exception:
        for output in published:
            output.unlink(missing_ok=True)
        raise
    finally:
        for temporary in temporary_paths:
            temporary.unlink(missing_ok=True)


def capture_snapshot(*, root: Path, live_state_path: Path,
                     live_status_path: Path, state_output_path: Path,
                     status_output_path: Path,
                     required_profile: dict | None = None) -> dict:
    """Validate and freeze a coherent state/status pair for reconciliation."""
    root = Path(root).resolve()
    profile = (R9_RECONCILIATION_PROFILE if required_profile is None
               else required_profile)
    state_output = Path(state_output_path).resolve()
    status_output = Path(status_output_path).resolve()
    if (state_output != root / profile["state_snapshot_path"] or
            status_output != root / profile["status_snapshot_path"]):
        raise ValueError("Canonical immutable snapshot paths required")
    if state_output.exists() or status_output.exists():
        raise FileExistsError("Preserve existing active-batch evidence snapshot")

    state_bytes, status_bytes = _stable_pair(
        Path(live_state_path).resolve(), Path(live_status_path).resolve())
    state = json.loads(state_bytes)
    status = json.loads(status_bytes)
    start, stop = profile["population_range"]
    expected_indices = list(range(start, stop))
    expected_task_ids = [f"population-{index:03d}"
                         for index in expected_indices]
    tasks = state.get("tasks")
    if (state.get("format") != "research-harness-state-v1" or
            state.get("batch_id") != profile["run_id"] or
            state.get("plan_digest") != profile["plan_digest"] or
            not isinstance(tasks, dict) or
            set(tasks) != set(expected_task_ids)):
        raise ValueError("Exact complete retained harness state required")

    counts = {name: 0 for name in RETAINED_DISPOSITIONS}
    for task_id in expected_task_ids:
        task = tasks[task_id]
        task_status = task.get("status") if isinstance(task, dict) else None
        if task_status not in counts:
            raise ValueError("Unsupported retained harness task status")
        counts[task_status] += 1
    window = retained_observation_window(status, state, profile)
    if (not isinstance(window, dict) or
            window.get("run_id") != profile["run_id"] or
            window.get("plan_digest") != profile["plan_digest"] or
            window.get("indices") != expected_indices or
            any(window.get(name) != count for name, count in counts.items())):
        raise ValueError("Harness state/status counts or identity differ")
    observed_at = window.get("observed_at")
    try:
        if not isinstance(observed_at, str) or not observed_at.endswith("Z"):
            raise ValueError
        datetime.fromisoformat(observed_at.removesuffix("Z") + "+00:00")
    except ValueError:
        raise ValueError("UTC status observation time required") from None

    _write_pair(state_output, state_bytes, status_output, status_bytes)
    return {
        "kind": "actionbench-full128-active-batch-evidence-snapshot",
        "version": "1.0.0",
        "run_id": profile["run_id"],
        "plan_digest": profile["plan_digest"],
        "observed_at": observed_at,
        "counts": counts,
        "state_snapshot_ref": _ref(root, state_output, state_bytes),
        "status_snapshot_ref": _ref(root, status_output, status_bytes),
        "execution_started": False,
        "queue_approved": False,
        "dispatch_ready": False,
        "scientific_effect_qualification": False,
        "candidate_methods_tested": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--live-state", type=Path, required=True)
    parser.add_argument("--live-status", type=Path, required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    result = capture_snapshot(
        root=root,
        live_state_path=args.live_state,
        live_status_path=args.live_status,
        state_output_path=root / R9_STATE_SNAPSHOT_PATH,
        status_output_path=root / R9_STATUS_SNAPSHOT_PATH,
    )
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

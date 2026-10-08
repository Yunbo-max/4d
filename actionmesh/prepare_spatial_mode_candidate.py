"""Build C05's CPU-only five-role artifact plan; never run or score it.

The plan consumes one completed same-anchor mode bank and the *separate*,
byte-bound B0 parity receipt.  Parameters are prospective command-line inputs;
no value is selected from ActionBench outcomes.  The emitted task has one
attempt, zero retries, and zero GPUs.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
from pathlib import Path
import platform
import sys


CANDIDATE_ID = "4d-math-20261006-c05"
MODE_BANK_KIND = "c05-same-anchor-stage1-mode-bank-manifest"
ROLE_ORDER = ("localized_mean", "temperature_matched_mean",
              "surface_projected_mean", "independent_top1",
              "joint_spatial_labels")


def _digest(path: Path) -> str:
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def _physical(root: Path, path: Path) -> Path:
    root, path = Path(root).resolve(), Path(path)
    path = path if path.is_absolute() else root / path
    if path.is_symlink() or any(parent.is_symlink() for parent in path.parents):
        raise ValueError("C05 inputs must be regular physical files")
    path = path.resolve()
    path.relative_to(root)
    if not path.is_file():
        raise ValueError("Missing C05 input: " + str(path))
    return path


def _ref(root: Path, path: Path) -> dict:
    path = _physical(root, path)
    return {"path": path.relative_to(Path(root).resolve()).as_posix(),
            "sha256": _digest(path)}


def _load(path: Path) -> dict:
    def unique(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError("Duplicate JSON key: " + key)
            result[key] = value
        return result
    result = json.loads(Path(path).read_text(), object_pairs_hook=unique)
    if not isinstance(result, dict):
        raise ValueError("JSON object required: " + str(path))
    return result


def _positive(name: str, value) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) \
            or not math.isfinite(value) or value <= 0:
        raise ValueError(name + " must be finite and positive")
    return float(value)


def _canonical_relative(value, *, name: str) -> Path:
    if not isinstance(value, str):
        raise ValueError("Canonical C05 request path required: " + name)
    path = Path(value)
    if path.is_absolute() or ".." in path.parts or path.as_posix() != value:
        raise ValueError("Canonical C05 request path required: " + name)
    return path


def _checked_local_ref(directory: Path, value: dict) -> Path:
    if (not isinstance(value, dict)
            or not {"path", "sha256", "bytes"}.issubset(value)):
        raise ValueError("Complete mode-bank content reference required")
    relative = Path(value["path"])
    if relative.is_absolute() or ".." in relative.parts or relative.as_posix() != value["path"]:
        raise ValueError("Canonical mode-bank relative path required")
    path = directory / relative
    if (not path.is_file() or path.is_symlink()
            or path.stat().st_size != value["bytes"]
            or _digest(path) != value["sha256"]):
        raise ValueError("Mode-bank retained file changed: " + value["path"])
    return path


def _producer_execution(root: Path, plan_path: Path, receipt_path: Path, native):
    """Verify the exact successful one-attempt producer and return its workspace."""
    plan_path = _physical(root, plan_path)
    receipt_path = _physical(root, receipt_path)
    plan, receipt = _load(plan_path), _load(receipt_path)
    native.validate_plan(root, plan)
    run_root = (root / plan.get("output_root", "") / plan.get("run_id", "")).resolve()
    run_root.relative_to(root)
    if receipt_path != run_root / "receipt.json":
        raise ValueError("Canonical C05 producer receipt path required")
    jobs, attempts = plan.get("jobs"), receipt.get("attempts")
    if (receipt.get("status") != "completed"
            or receipt.get("run_id") != plan.get("run_id")
            or receipt.get("plan_digest") != plan.get("plan_digest")
            or receipt.get("purpose") != plan.get("purpose")
            or receipt.get("evidence_mode") != plan.get("evidence_mode")
            or receipt.get("provenance") != plan.get("provenance")
            or not isinstance(jobs, list) or len(jobs) != 1
            or not isinstance(attempts, list) or len(attempts) != 1
            or plan.get("limits", {}).get("max_attempts") != 1
            or plan.get("limits", {}).get("max_retries_per_trial") != 0):
        raise ValueError("Exact completed single-attempt C05 producer receipt required")
    job, attempt = jobs[0], attempts[0]
    if (job.get("trial_id") != "c05-same-anchor-mode-bank"
            or attempt.get("trial_id") != job["trial_id"]
            or attempt.get("status") != "completed"
            or attempt.get("exit_code") != 0
            or attempt.get("retry_index") != 0
            or any(attempt.get(name) != job.get(name)
                   for name in ("input_refs", "code_refs", "seed", "group", "arm_role"))):
        raise ValueError("C05 producer job/attempt identity mismatch")
    attempt_path = Path(attempt.get("attempt_path", ""))
    if (attempt_path.is_absolute() or ".." in attempt_path.parts):
        raise ValueError("Canonical C05 producer attempt path required")
    workspace = (root / attempt_path / "workspace").resolve()
    workspace.relative_to(root)
    if (attempt_path.parent != run_root.relative_to(root)
            or attempt.get("attempt_id") != attempt_path.name
            or not workspace.is_dir()):
        raise ValueError("Retained C05 producer workspace is missing")
    replacements = {}
    for ref in job["input_refs"] + job["code_refs"]:
        source = _physical(root, ref["path"])
        staged = _physical(root, workspace / ref["path"])
        if _digest(source) != ref["sha256"] or _digest(staged) != ref["sha256"]:
            raise ValueError("C05 producer source/staged identity mismatch")
        replacements[str(source)] = str(staged)
    expected_command = []
    for argument in job["command"]:
        for source, staged in sorted(replacements.items(), key=lambda row: -len(row[0])):
            argument = argument.replace(source, staged)
        expected_command.append(argument)
    expected_cwd = workspace if job["cwd"] == "." else workspace / job["cwd"]
    if attempt.get("command") != expected_command or Path(attempt.get("cwd", "")) != expected_cwd:
        raise ValueError("C05 producer staged command/cwd differs from frozen plan")
    expected_outputs = []
    for relative in job["output_paths"]:
        path = _physical(root, workspace / relative)
        expected_outputs.append(_ref(root, path))
    if attempt.get("output_refs") != expected_outputs:
        raise ValueError("C05 producer output receipt inventory mismatch")
    try:
        index = job["command"].index("--output-relative")
        output_relative = _canonical_relative(
            job["command"][index + 1], name="producer output")
    except (ValueError, IndexError) as error:
        raise ValueError("Frozen C05 producer output argv required") from error
    output = (workspace / output_relative).resolve()
    output.relative_to(workspace)
    parity = output.parent / (output.name + "-b0-parity.json")
    if (str(output_relative / "result.json") not in job["output_paths"]
            or parity.relative_to(workspace).as_posix() not in job["output_paths"]):
        raise ValueError("C05 producer plan omits result/parity outputs")
    return plan_path, receipt_path, workspace, output, parity


def mode_bank_refs(root: Path, mode_bank_root: Path, *, producer_workspace: Path
                   ) -> tuple[list[dict], dict, dict, dict]:
    """Pin the exact complete producer directory without importing NumPy."""
    root = Path(root).resolve()
    producer_workspace = Path(producer_workspace).resolve()
    producer_workspace.relative_to(root)
    unresolved = Path(mode_bank_root)
    unresolved = unresolved if unresolved.is_absolute() else root / unresolved
    if unresolved.is_symlink() or any(
            parent.is_symlink() for parent in unresolved.parents
            if parent != root.parent):
        raise ValueError("Physical mode-bank directory required")
    directory = unresolved.resolve()
    directory.relative_to(root)
    result_path = _physical(root, directory / "result.json")
    manifest_path = _physical(root, directory / "raw-manifest.json")
    result, manifest = _load(result_path), _load(manifest_path)
    if (result.get("kind") != "c05-same-anchor-stage1-mode-bank-result"
            or result.get("version") != 1
            or result.get("status") != "completed_unqualified"
            or result.get("candidate_id") != "C05"
            or manifest.get("kind") != MODE_BANK_KIND
            or manifest.get("version") != 1
            or manifest.get("status") != "completed_unqualified"
            or manifest.get("candidate_id") != "C05"):
        raise ValueError("Completed unqualified C05 mode bank required")
    if result.get("manifest_ref", {}).get("sha256") != _digest(manifest_path):
        raise ValueError("Mode-bank manifest identity mismatch")
    retained = [_checked_local_ref(directory, manifest.get("mode_bank_ref"))]
    branches = manifest.get("branches")
    inner_seeds = manifest.get("inner_stage1_seeds")
    if (not isinstance(branches, list) or len(branches) < 3
            or not isinstance(inner_seeds, list) or len(inner_seeds) != len(branches)
            or manifest.get("branch_count") != len(branches)
            or result.get("full_sequences_retained") != len(branches)):
        raise ValueError("At least three complete retained C05 branches required")
    for index, row in enumerate(branches):
        if (not isinstance(row, dict) or row.get("branch_index") != index
                or row.get("stage1_seed") != inner_seeds[index]):
            raise ValueError("Canonical complete C05 branch inventory required")
        retained.append(_checked_local_ref(directory, row.get("sequence_ref")))
    actual = {path.relative_to(directory).as_posix() for path in directory.rglob("*")
              if path.is_file()}
    expected = {"result.json", "raw-manifest.json"}
    expected.update(path.relative_to(directory).as_posix() for path in retained)
    if actual != expected:
        raise ValueError("Undeclared or missing C05 mode-bank file")
    files = [result_path, manifest_path, *retained]
    closures = {}
    for field in ("producer_input_refs", "producer_code_refs"):
        closure = manifest.get(field)
        if (not isinstance(closure, list) or not closure or len(closure) > 128
                or result.get(field) != closure):
            raise ValueError("Complete matching C05 " + field + " required")
        seen = set()
        for ref in closure:
            if (not isinstance(ref, dict) or set(ref) != {"path", "sha256"}
                    or ref["path"] in seen):
                raise ValueError("Canonical unique C05 " + field + " required")
            seen.add(ref["path"])
            path = _physical(producer_workspace, ref["path"])
            if _ref(producer_workspace, path) != ref:
                raise ValueError("C05 producer closure changed: " + ref["path"])
            files.append(path)
        closures[field] = closure
    request_refs = [ref for ref in closures["producer_input_refs"]
                    if Path(ref["path"]).name == "request.json"]
    if len(request_refs) != 1:
        raise ValueError("One exact retained C05 producer request required")
    request = _load(_physical(producer_workspace, request_refs[0]["path"]))
    request_output = _canonical_relative(
        request.get("output_relative"), name="output_relative")
    request_frames = _canonical_relative(
        request.get("frames_relative"), name="frames_relative")
    request_source = _canonical_relative(
        request.get("source_root_relative"), name="source_root_relative")
    if (request.get("kind") != "c05-mode-bank-request"
            or request.get("version") != 1
            or request.get("candidate_id") != CANDIDATE_ID
            or request.get("uid") != manifest.get("provenance", {}).get("generation_uid")
            or request.get("outer_seed") != manifest.get("outer_generation_seed")
            or request.get("branch_count") != manifest.get("branch_count")
            or request.get("generation_parameters") != manifest.get("generation_parameters")
            or request.get("provenance") != manifest.get("provenance")
            or (producer_workspace / request_output).resolve() != directory
            or request.get("producer_input_refs")
               != [ref for ref in closures["producer_input_refs"]
                   if ref != request_refs[0]]
            or request.get("producer_code_refs") != closures["producer_code_refs"]):
        raise ValueError("C05 producer request/closure identity mismatch")
    frame_root = (producer_workspace / request_frames).resolve()
    expected_frames = [_ref(producer_workspace, frame_root / f"{frame:02d}.png")
                       for frame in range(16)]
    source_root = (producer_workspace / request_source).resolve()
    expected_native_sources = [_ref(producer_workspace, source_root / path) for path in (
        "actionmesh/pipeline.py", "actionmesh/model/temporal_autoencoder.py",
        "actionmesh/model/utils/storage.py", "actionmesh/scheduler/scheduler.py",
        "actionmesh/io/video_input.py")]
    request_provenance = request.get("provenance", {})
    semantic_inputs = [request.get("authorization_ref"),
                       *(request_provenance.get(name) for name in
                         ("model_ref", "input_ref", "config_ref", "environment_ref")),
                       *expected_frames]
    semantic_codes = [request_provenance.get("source_ref"),
                      *expected_native_sources]
    if (any(ref not in closures["producer_input_refs"] for ref in semantic_inputs)
            or any(ref not in closures["producer_code_refs"]
                   for ref in semantic_codes)
            or request.get("producer_code_refs") != closures["producer_code_refs"]):
        raise ValueError("C05 producer closure lacks exact semantic inputs/code")
    return [_ref(root, path) for path in files], result, manifest, request


def _exact_receipt_refs(root: Path, parity: dict, expected: dict[str, Path]) -> list[dict]:
    refs = []
    for field, path in expected.items():
        current = _ref(root, path)
        if parity.get(field) != current:
            raise ValueError("B0 parity receipt does not bind exact " + field)
        refs.append(current)
    return refs


def build_plans(root: Path, *, skill_dir: Path, mode_bank_root: Path,
                mode_bank_plan: Path, mode_bank_receipt: Path,
                parity_receipt: Path, b0_sequence: Path, b0_report: Path,
                output: Path,
                plan_dir: Path, run_id: str, landmark_count: int,
                cluster_radius: float, natural_gate_min_fraction: float,
                localized_radius: float, temperature: float, unary_weight: float,
                spatial_weight: float, max_sweeps: int,
                displacement_clip_multiplier: float, face_chunk_size: int,
                max_artifact_bytes: int, wall_seconds: int, ram_mib: int):
    root, skill_dir, output, plan_dir = (Path(item).resolve() for item in
                                         (root, skill_dir, output, plan_dir))
    output.relative_to(root); plan_dir.relative_to(root)
    if output.exists() or plan_dir.exists():
        raise FileExistsError("Preserve previous C05 output/plan; use new paths")
    if type(landmark_count) is not int or not 3 <= landmark_count <= 256:
        raise ValueError("landmark_count must be an integer in [3,256]")
    if (type(max_sweeps) is not int or max_sweeps < 1
            or type(face_chunk_size) is not int or face_chunk_size < 1):
        raise ValueError("Positive integer solver limits required")
    if (type(max_artifact_bytes) is not int or max_artifact_bytes < 10240
            or type(wall_seconds) is not int or not 1 <= wall_seconds <= 26940
            or type(ram_mib) is not int or ram_mib < 1):
        raise ValueError("Explicit bounded artifact/wall/RAM budgets required")
    parameters = {
        "cluster_radius": _positive("cluster_radius", cluster_radius),
        "natural_gate_min_fraction": _positive(
            "natural_gate_min_fraction", natural_gate_min_fraction),
        "localized_radius": _positive("localized_radius", localized_radius),
        "temperature": _positive("temperature", temperature),
        "unary_weight": _positive("unary_weight", unary_weight),
        "spatial_weight": _positive("spatial_weight", spatial_weight),
        "displacement_clip_multiplier": _positive(
            "displacement_clip_multiplier", displacement_clip_multiplier),
    }
    if parameters["natural_gate_min_fraction"] > 1:
        raise ValueError("natural_gate_min_fraction must be <= 1")

    scripts = skill_dir / "scripts"
    if not (scripts / "run_harness.py").is_file():
        raise ValueError("Complete installed research-autopilot required")
    sys.path.insert(0, str(scripts))
    import run_experiments as native
    import run_harness as harness
    producer_plan_path, producer_receipt_path, producer_workspace, derived_mode_root, \
        derived_parity = _producer_execution(
            root, mode_bank_plan, mode_bank_receipt, native)
    raw_mode_bank_root = Path(mode_bank_root)
    mode_bank_root = (raw_mode_bank_root if raw_mode_bank_root.is_absolute()
                      else root / raw_mode_bank_root).resolve()
    if mode_bank_root != derived_mode_root:
        raise ValueError("Mode-bank path differs from exact producer receipt")
    inputs, result, manifest, request = mode_bank_refs(
        root, mode_bank_root, producer_workspace=producer_workspace)
    inputs.extend((_ref(root, producer_plan_path), _ref(root, producer_receipt_path)))
    provenance = manifest.get("provenance")
    required_provenance = ("source_ref", "model_ref", "input_ref", "config_ref",
                           "environment_ref")
    if not isinstance(provenance, dict):
        raise ValueError("Complete C05 producer provenance required")
    for name in required_provenance:
        ref = provenance.get(name)
        if not isinstance(ref, dict) or set(ref) != {"path", "sha256"}:
            raise ValueError("Exact C05 producer provenance ref required: " + name)
        path = _physical(producer_workspace, ref["path"])
        if _ref(producer_workspace, path) != ref:
            raise ValueError("C05 producer provenance changed: " + name)
        inputs.append(_ref(root, path))
    parity_path = _physical(root, parity_receipt)
    if parity_path != derived_parity:
        raise ValueError("Parity path differs from exact producer receipt")
    b0_path = _physical(root, b0_sequence)
    if b0_path.name != "sequence.npz":
        raise ValueError("Retained B0 sequence.npz required")
    b0_report_path = _physical(root, b0_report)
    if b0_report_path != b0_path.with_name("report.json"):
        raise ValueError("B0 report.json must be the sequence sibling")
    b0_report = _load(b0_report_path)
    request_refs = [ref for ref in manifest["producer_input_refs"]
                    if Path(ref["path"]).name == "request.json"]
    request_path = _physical(producer_workspace, request_refs[0]["path"])
    expected_b0_path = _physical(
        producer_workspace, request["b0_sequence_ref"]["path"])
    expected_b0_report = _physical(
        producer_workspace, request["b0_report_ref"]["path"])
    if (b0_path != expected_b0_path or b0_report_path != expected_b0_report
            or request.get("b0_sequence_ref") != _ref(producer_workspace, b0_path)
            or request.get("b0_report_ref") != _ref(
                producer_workspace, b0_report_path)):
        raise ValueError("Retained C05 producer request differs from supplied B0 pair")
    if (b0_report.get("status") != "completed"
            or b0_report.get("seed") != result.get("outer_generation_seed")
            or b0_report.get("sha256", {}).get("sequence.npz") != _digest(b0_path)):
        raise ValueError("Completed retained B0 sequence/report pair required")
    parity = _load(parity_path)
    branch0 = mode_bank_root / manifest["branches"][0]["sequence_ref"]["path"]
    if (parity.get("kind") != "c05-b0-parity-receipt"
            or parity.get("version") != 1
            or parity.get("candidate_id") != CANDIDATE_ID
            or parity.get("matches") is not True
            or parity.get("outer_seed") != result.get("outer_generation_seed")
            or parity.get("uid") != b0_report.get("uid")
            or parity.get("candidate_methods_tested") is not False
            or parity.get("scientific_effect_qualification") is not False
            or parity.get("native_qualified") is not False):
        raise ValueError("Exact unqualified successful C05 B0 parity receipt required")
    _exact_receipt_refs(producer_workspace, parity, {
        "mode_bank_result_ref": mode_bank_root / "result.json",
        "mode_bank_manifest_ref": mode_bank_root / "raw-manifest.json",
        "mode_bank_array_ref": mode_bank_root / manifest["mode_bank_ref"]["path"],
        "branch_zero_ref": branch0,
        "retained_b0_sequence_ref": b0_path,
        "retained_b0_report_ref": b0_report_path,
    })
    inputs.extend(_ref(root, path) for path in (
        mode_bank_root / "result.json", mode_bank_root / "raw-manifest.json",
        mode_bank_root / manifest["mode_bank_ref"]["path"], branch0,
        b0_path, b0_report_path))
    inputs.append(_ref(root, parity_path))
    unique = {}
    for ref in inputs:
        if ref["path"] in unique and unique[ref["path"]] != ref:
            raise ValueError("Conflicting C05 candidate input identity")
        unique[ref["path"]] = ref

    code_paths = (
        "actionmesh/research_math/__init__.py",
        "actionmesh/research_math/c05_mode_bank.py",
        "actionmesh/research_math/spatial_mode_candidate.py",
        "actionmesh/research_math/c05_candidate_artifacts.py",
    )
    code_refs = [_ref(root, root / path) for path in code_paths]
    if manifest.get("producer_code_sha256") != code_refs[1]["sha256"]:
        raise ValueError("Retained mode bank differs from pinned C05 producer source")
    environment = {
        "python_executable": sys.executable, "python": platform.python_version(),
        "numpy": importlib.metadata.version("numpy"),
        "scope": "CPU C05 five-role artifact only; no model, GPU, scorer or admission",
        "parameters_prospective": True,
    }
    environment_digest = hashlib.sha256(json.dumps(
        environment, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    program = r'''import argparse,sys
from pathlib import Path
parser=argparse.ArgumentParser()
for name in ("project-sentinel","producer-request","producer-request-relative",
 "mode-bank-result","parity-receipt","b0-sequence","b0-report","output-relative"):
 parser.add_argument("--"+name,required=True)
for name in ("landmark-count","max-sweeps","face-chunk-size","max-artifact-bytes"):
 parser.add_argument("--"+name,type=int,required=True)
for name in ("cluster-radius","natural-gate-min-fraction","localized-radius",
 "temperature","unary-weight","spatial-weight","displacement-clip-multiplier"):
 parser.add_argument("--"+name,type=float,required=True)
parser.add_argument("--closure-file",action="append",default=[])
parser.add_argument("--expected-closure-count",type=int,required=True)
ns=parser.parse_args()
project=Path(ns.project_sentinel).resolve().parents[2]
producer_request=Path(ns.producer_request).resolve()
producer_request_relative=Path(ns.producer_request_relative)
if (producer_request_relative.is_absolute() or ".." in producer_request_relative.parts
 or producer_request_relative.as_posix()!=ns.producer_request_relative):
 raise ValueError("canonical C05 producer request relative path required")
producer=producer_request
for _ in producer_request_relative.parts:producer=producer.parent
if ((producer/producer_request_relative).resolve()!=producer_request
 or not producer.is_relative_to(project)):
 raise ValueError("C05 producer workspace cannot be derived from staged request")
for path in (Path(ns.project_sentinel).resolve(),producer_request,Path(ns.mode_bank_result).resolve(),
 Path(ns.parity_receipt).resolve(),Path(ns.b0_sequence).resolve(),Path(ns.b0_report).resolve()):
 path.relative_to(project)
closure={Path(path).resolve() for path in ns.closure_file}
if (len(closure)!=ns.expected_closure_count or any(not path.is_file() for path in closure)
 or any(not path.is_relative_to(project) for path in closure)):
 raise ValueError("every C05 artifact input/code file must be an explicit staged argv token")
required={Path(ns.project_sentinel).resolve(),producer_request,Path(ns.mode_bank_result).resolve(),
 Path(ns.parity_receipt).resolve(),Path(ns.b0_sequence).resolve(),Path(ns.b0_report).resolve()}
if not required.issubset(closure):raise ValueError("C05 staged runtime input missing from closure")
relative=Path(ns.output_relative)
if (relative.is_absolute() or ".." in relative.parts
 or relative.as_posix()!=ns.output_relative):raise ValueError("canonical workspace output required")
output=(project/relative).resolve();output.relative_to(project)
sys.path.insert(0,str(project/"actionmesh"))
from research_math import c05_candidate_artifacts as candidate
if Path(candidate.__file__).resolve()!=Path(ns.project_sentinel).resolve():
 raise ValueError("C05 candidate implementation did not load from staged workspace")
result=candidate.materialize_candidate(project,Path(ns.mode_bank_result).resolve().parent,
 Path(ns.parity_receipt).resolve(),Path(ns.b0_sequence).resolve(),Path(ns.b0_report).resolve(),
 output,landmark_count=ns.landmark_count,cluster_radius=ns.cluster_radius,
 natural_gate_min_fraction=ns.natural_gate_min_fraction,localized_radius=ns.localized_radius,
 temperature=ns.temperature,unary_weight=ns.unary_weight,spatial_weight=ns.spatial_weight,
 max_sweeps=ns.max_sweeps,displacement_clip_multiplier=ns.displacement_clip_multiplier,
 max_artifact_bytes=ns.max_artifact_bytes,face_chunk_size=ns.face_chunk_size,
 producer_root=producer)
if result["status"] not in ("completed_unqualified","incomplete_natural_gate"):
 raise SystemExit(2)
candidate.validate_candidate_artifact(project,output)
'''
    closure_refs = {}
    for ref in [*unique.values(), *code_refs]:
        if ref["path"] in closure_refs and closure_refs[ref["path"]] != ref:
            raise ValueError("Conflicting C05 staged closure identity")
        closure_refs[ref["path"]] = ref
    relative_output = output.relative_to(root).as_posix()
    command = [
        sys.executable, "-c", program,
        "--project-sentinel", str(root / "actionmesh/research_math/c05_candidate_artifacts.py"),
        "--producer-request", str(request_path),
        "--producer-request-relative", request_refs[0]["path"],
        "--mode-bank-result", str(mode_bank_root / "result.json"),
        "--parity-receipt", str(parity_path), "--b0-sequence", str(b0_path),
        "--b0-report", str(b0_report_path), "--output-relative", relative_output,
        "--landmark-count", str(landmark_count),
        "--cluster-radius", str(parameters["cluster_radius"]),
        "--natural-gate-min-fraction", str(parameters["natural_gate_min_fraction"]),
        "--localized-radius", str(parameters["localized_radius"]),
        "--temperature", str(parameters["temperature"]),
        "--unary-weight", str(parameters["unary_weight"]),
        "--spatial-weight", str(parameters["spatial_weight"]),
        "--max-sweeps", str(max_sweeps),
        "--displacement-clip-multiplier", str(parameters["displacement_clip_multiplier"]),
        "--face-chunk-size", str(face_chunk_size),
        "--max-artifact-bytes", str(max_artifact_bytes),
        "--expected-closure-count", str(len(closure_refs)),
    ]
    command.extend(item for ref in closure_refs.values()
                   for item in ("--closure-file", str((root / ref["path"]).resolve())))
    outputs = [f"{relative_output}/{name}" for name in (
        "candidate.json", "result.json", "raw-manifest.json", "certificate.json",
        "solver-certificate.json", "raw-evidence.tar")]
    outputs.extend(
        f"{relative_output}/roles/{role}/report.json"
        for role in ROLE_ORDER
    )
    plan = native.make_plan(root, run_id=run_id, jobs=[{
        "trial_id": "prepare-c05-spatial-mode-candidate", "command": command,
        "cwd": ".", "input_refs": list(unique.values()),
        "code_refs": code_refs, "output_paths": outputs,
        "seed": result["outer_generation_seed"], "group": "candidate-artifacts-only",
        "arm_role": "five-common-bank-artifacts-no-scorer-no-admission",
    }], provenance={
        "git_revision": "exact source files pinned in code_refs",
        "model_revision": manifest.get("provenance", {}).get("model_ref", {}).get(
            "sha256", "mode-bank-provenance-bound"),
        "data_revision": _digest(mode_bank_root / "result.json"),
        "environment_digest": environment_digest,
    }, limits={
        "max_attempts": 1, "max_development_trials": 1,
        "max_confirmation_trials": 0, "max_retries_per_trial": 0,
        "wall_time_seconds": wall_seconds, "attempt_timeout_seconds": wall_seconds,
    })
    plan_dir.mkdir(parents=True, exist_ok=False)
    native_path = plan_dir / "native.json"
    native_path.write_text(json.dumps(plan, indent=2) + "\n")
    (plan_dir / "environment.json").write_text(json.dumps(environment, indent=2) + "\n")
    outer = harness.make_plan(root, batch_id=run_id, tasks=[{
        "task_id": "prepare-c05-spatial-mode-candidate", "idea_id": CANDIDATE_ID,
        "depends_on": [], "priority": 1, "plan_ref": _ref(root, native_path),
        "resources": {"cpu_cores": 1, "ram_mib": ram_mib, "gpu_count": 0,
            "gpu_peak_mib": None, "allow_gpu_share": False,
            "memory_profile_ref": None, "exclusive_keys": []},
    }], limits={"total_wall_seconds": wall_seconds + 60,
        "window_seconds": wall_seconds + 60, "max_parallel_tasks": 1,
        "cpu_cores": 1, "ram_mib": ram_mib, "max_gpu_task_seconds": 0})
    (plan_dir / "harness.json").write_text(json.dumps(outer, indent=2) + "\n")
    return plan, outer


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("root", "skill-dir", "mode-bank-root", "mode-bank-plan",
                 "mode-bank-receipt", "parity-receipt",
                 "b0-sequence", "b0-report", "output", "plan-dir"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--landmark-count", type=int, required=True)
    for name in ("cluster-radius", "natural-gate-min-fraction", "localized-radius",
                 "temperature", "unary-weight", "spatial-weight",
                 "displacement-clip-multiplier"):
        parser.add_argument("--" + name, type=float, required=True)
    parser.add_argument("--max-sweeps", type=int, required=True)
    parser.add_argument("--face-chunk-size", type=int, default=4096)
    parser.add_argument("--max-artifact-bytes", type=int, required=True)
    parser.add_argument("--wall-seconds", type=int, required=True)
    parser.add_argument("--ram-mib", type=int, required=True)
    args = parser.parse_args(argv)
    _, outer = build_plans(args.root, skill_dir=args.skill_dir,
        mode_bank_root=args.mode_bank_root, mode_bank_plan=args.mode_bank_plan,
        mode_bank_receipt=args.mode_bank_receipt,
        parity_receipt=args.parity_receipt,
        b0_sequence=args.b0_sequence, b0_report=args.b0_report,
        output=args.output, plan_dir=args.plan_dir,
        run_id=args.run_id, landmark_count=args.landmark_count,
        cluster_radius=args.cluster_radius,
        natural_gate_min_fraction=args.natural_gate_min_fraction,
        localized_radius=args.localized_radius, temperature=args.temperature,
        unary_weight=args.unary_weight, spatial_weight=args.spatial_weight,
        max_sweeps=args.max_sweeps,
        displacement_clip_multiplier=args.displacement_clip_multiplier,
        face_chunk_size=args.face_chunk_size,
        max_artifact_bytes=args.max_artifact_bytes,
        wall_seconds=args.wall_seconds, ram_mib=args.ram_mib)
    print(json.dumps({"plan": str(args.plan_dir.resolve() / "harness.json"),
        "approved_plan_digest": outer["plan_digest"], "execution_started": False,
        "source_delivery_status": "generated_unexecuted",
        "candidate_id": CANDIDATE_ID, "gpu_count": 0,
        "native_qualified": False, "scientific_admission": False,
        "local_method_verified": False}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Build one STOP-gated, single-attempt C05 same-anchor mode-bank plan.

The builder does not execute inference.  A future explicit GPU-resume
authorization is required even to materialize a dispatchable plan.  The inner
task compares branch zero with the retained native B0 before emitting its
separate parity receipt; candidate materialization must require that receipt.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import sys


CANDIDATE_ID = "4d-math-20261006-c05"
AUTHORIZATION_KIND = "c05-gpu-resume-authorization"
GENERATION_PARAMETERS = {
    "stage_0_steps": 100,
    "stage_1_steps": 30,
    "face_decimation": 40000,
    "floaters_threshold": 0.02,
    "guidance_scales": [7.5],
    "anchor_idx": 0,
}


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def read_json(path: Path) -> dict:
    value = json.loads(Path(path).read_text())
    if not isinstance(value, dict):
        raise ValueError("Expected JSON object: " + str(path))
    return value


def file_ref(root: Path, path: Path) -> dict:
    root, path = Path(root).resolve(), Path(path).resolve()
    if not path.is_file() or path.is_symlink():
        raise ValueError("Physical input file required: " + str(path))
    return {"path": path.relative_to(root).as_posix(), "sha256": digest(path)}


def _authorization(path: Path, *, uid: str, seed: int,
                   branches: int, gpu_uuid: str, run_id: str,
                   environment_sha256: str) -> dict:
    value = read_json(path)
    if (value.get("kind") != AUTHORIZATION_KIND or value.get("version") != 1
            or value.get("candidate_id") != CANDIDATE_ID
            or value.get("scope") != "single_c05_mode_bank_attempt"
            or value.get("status") != "approved"
            or value.get("gpu_stop_lifted") is not True
            or value.get("uid") != uid or value.get("outer_seed") != seed
            or value.get("branch_count") != branches
            or not isinstance(gpu_uuid, str) or not gpu_uuid
            or value.get("gpu_uuid") != gpu_uuid
            or value.get("run_id") != run_id
            or value.get("environment_sha256") != environment_sha256
            or value.get("max_attempts") != 1
            or value.get("max_retries_per_trial") != 0):
        raise ValueError("Exact single-attempt C05 GPU-resume authorization required")
    try:
        expiry = datetime.fromisoformat(value["expires_at"].replace("Z", "+00:00"))
    except (KeyError, AttributeError, ValueError) as error:
        raise ValueError("Timezone-aware C05 authorization expiry required") from error
    if expiry.tzinfo is None or expiry <= datetime.now(timezone.utc):
        raise ValueError("C05 authorization is expired")
    return value


def _frames(root: Path, directory: Path) -> list[dict]:
    directory = Path(directory).resolve(); directory.relative_to(root)
    expected = [directory / f"{frame:02d}.png" for frame in range(16)]
    if ({path.name for path in directory.iterdir() if path.is_file()}
            != {path.name for path in expected}):
        raise ValueError("Exact 16-frame ActionBench image directory required")
    return [file_ref(root, path) for path in expected]


def _source_code_paths(source_root: Path) -> list[Path]:
    """Return the bounded Python closure imported by the staged native job."""
    source_root = Path(source_root).resolve()
    paths = []
    for relative in ("actionmesh", "third_party/TripoSG"):
        directory = source_root / relative
        if directory.is_dir():
            paths.extend(path for path in directory.rglob("*.py")
                         if path.is_file()
                         and "pretrained_weights" not in
                             path.relative_to(source_root).parts
                         and ".git" not in path.relative_to(source_root).parts)
    paths = sorted(set(path.resolve() for path in paths))
    required = {source_root / path for path in (
        "actionmesh/pipeline.py", "actionmesh/model/temporal_autoencoder.py",
        "actionmesh/model/utils/storage.py", "actionmesh/scheduler/scheduler.py",
        "actionmesh/io/video_input.py")}
    if not required.issubset(set(paths)) or not paths or len(paths) > 96:
        raise ValueError("Bounded complete staged ActionMesh Python closure required")
    return paths


def _validate_weights_manifest_schema(path: Path) -> list[dict]:
    rows = json.loads(Path(path).read_text())
    if not isinstance(rows, list) or not rows or len(rows) > 256:
        raise ValueError("Nonempty bounded model-cache manifest required")
    names = set()
    for row in rows:
        if (not isinstance(row, dict)
                or not {"path", "size", "sha256"}.issubset(row)
                or not isinstance(row["path"], str)
                or Path(row["path"]).is_absolute()
                or ".." in Path(row["path"]).parts
                or Path(row["path"]).as_posix() != row["path"]
                or len(Path(row["path"]).parts) < 2
                or Path(row["path"]).parts[0] != "weights"
                or row["path"] in names
                or type(row["size"]) is not int or row["size"] < 0
                or not isinstance(row["sha256"], str)
                or len(row["sha256"]) != 64):
            raise ValueError("Canonical path/size/SHA256 model-cache row required")
        names.add(row["path"])
    if {Path(name).parts[1] for name in names} != {
            "ActionMesh", "TripoSG", "dinov2", "RMBG"}:
        raise ValueError("Exact four-root ActionMesh model-cache manifest required")
    return rows


def validate_exact_cache(cache_root: Path, rows: list[dict]) -> list[dict]:
    """Rehash one physical four-root cache and reject every unlisted byte."""
    cache_root = Path(cache_root).resolve()
    manifest_paths = {row.get("path") for row in rows if isinstance(row, dict)}
    if (len(manifest_paths) != len(rows) or None in manifest_paths
            or any(not isinstance(path, str) or len(Path(path).parts) < 2
                   or Path(path).is_absolute() or ".." in Path(path).parts
                   or Path(path).parts[0] != "weights"
                   for path in manifest_paths)
            or {Path(path).parts[1] for path in manifest_paths} !=
               {"ActionMesh", "TripoSG", "dinov2", "RMBG"}):
        raise ValueError("Exact four-root model cache manifest required")
    weights_root = cache_root / "weights"
    if (not weights_root.is_dir() or weights_root.is_symlink()
            or any(path.is_symlink() for path in weights_root.rglob("*"))):
        raise ValueError("Physical non-symbolic external model cache required")
    actual = {path.relative_to(cache_root).as_posix()
              for path in weights_root.rglob("*") if path.is_file()}
    if actual != manifest_paths:
        raise ValueError("External model cache inventory differs from manifest")
    inventory = []
    by_name = {row["path"]: row for row in rows}
    for name in sorted(actual):
        row = by_name[name]
        path = cache_root / name
        if (type(row.get("size")) is not int or row["size"] < 0
                or not isinstance(row.get("sha256"), str)
                or len(row["sha256"]) != 64
                or path.stat().st_size != row["size"]
                or path.with_name(path.name + ".aria2").exists()
                or digest(path) != row["sha256"]):
            raise ValueError("External model cache file differs: " + name)
        inventory.append({"path": name, "size": row["size"],
                          "sha256": row["sha256"]})
    return inventory


def build_plans(root: Path, *, skill_dir: Path, source_root: Path,
                frames: Path, b0_sequence: Path, b0_report: Path,
                authorization: Path,
                environment: Path, weights_manifest: Path, config: Path,
                plan_dir: Path, output: Path, uid: str, outer_seed: int,
                branch_count: int, run_id: str, wall_seconds: int,
                gpu_peak_mib: int) -> tuple[dict, dict]:
    root, skill_dir, source_root, plan_dir, output = (
        Path(item).resolve() for item in
        (root, skill_dir, source_root, plan_dir, output))
    source_root.relative_to(root); plan_dir.relative_to(root); output.relative_to(root)
    if output.exists() or plan_dir.exists():
        raise FileExistsError("Preserve prior C05 plan/output; use new paths")
    if (type(branch_count) is not int or branch_count < 3
            or type(outer_seed) is not int or outer_seed not in (42, 314, 2718)):
        raise ValueError("At least three branches and an exact G01 outer seed required")
    if type(wall_seconds) is not int or not 1 <= wall_seconds <= 26940:
        raise ValueError("C05 must fit the unchanged single-unit wall budget")
    if type(gpu_peak_mib) is not int or gpu_peak_mib < 1:
        raise ValueError("Positive measured/admitted GPU-memory ceiling required")
    scripts = skill_dir / "scripts"
    if not (scripts / "run_harness.py").is_file():
        raise ValueError("Complete installed research-autopilot required")
    authorization = Path(authorization).resolve()
    b0_sequence = Path(b0_sequence).resolve()
    b0_report = Path(b0_report).resolve()
    environment = Path(environment).resolve()
    gpu_uuid = read_json(environment).get("gpu_uuid")
    _authorization(authorization, uid=uid, seed=outer_seed,
                   branches=branch_count, gpu_uuid=gpu_uuid, run_id=run_id,
                   environment_sha256=digest(environment))
    weights_manifest = Path(weights_manifest).resolve()
    weight_rows = _validate_weights_manifest_schema(weights_manifest)
    validate_exact_cache(weights_manifest.parent, weight_rows)
    config = Path(config).resolve()
    frame_refs = _frames(root, frames)
    refs = [file_ref(root, path) for path in
        (authorization, b0_sequence, b0_report, environment, weights_manifest,
         config)]
    refs.extend(frame_refs)
    source_paths = _source_code_paths(source_root)
    project_sources = [root / "actionmesh/research_math" / name for name in
        ("__init__.py", "c05_mode_bank.py")]
    project_sources.append(root / "actionmesh/prepare_c05_mode_bank.py")
    code_refs = [file_ref(root, path) for path in [*project_sources, *source_paths]]
    provenance = {
        "generation_uid": uid,
        "gpu_uuid": gpu_uuid,
        "source_ref": file_ref(root, source_root / "actionmesh/pipeline.py"),
        "model_ref": file_ref(root, weights_manifest),
        "input_ref": file_ref(root, Path(frames).resolve() / "00.png"),
        "config_ref": file_ref(root, config),
        "environment_ref": file_ref(root, environment),
    }
    request = {
        "kind": "c05-mode-bank-request", "version": 1,
        "candidate_id": CANDIDATE_ID, "uid": uid,
        "run_id": run_id,
        "outer_seed": outer_seed, "branch_count": branch_count,
        "source_root_relative": source_root.relative_to(root).as_posix(),
        "frames_relative": Path(frames).resolve().relative_to(root).as_posix(),
        "b0_sequence_ref": file_ref(root, b0_sequence),
        "b0_report_ref": file_ref(root, b0_report),
        "output_relative": output.relative_to(root).as_posix(),
        "config_name": config.name, "generation_parameters": GENERATION_PARAMETERS,
        "provenance": provenance,
        "authorization_ref": file_ref(root, authorization),
        "producer_input_refs": refs,
        "producer_code_refs": code_refs,
    }
    plan_dir.mkdir(parents=True)
    request_path = plan_dir / "request.json"
    request_path.write_text(json.dumps(request, indent=2, allow_nan=False) + "\n")
    refs.append(file_ref(root, request_path))
    program = r'''import argparse,hashlib,json,os,sys
from pathlib import Path
import numpy as np
parser=argparse.ArgumentParser()
for name in ("request","project-sentinel","source-sentinel","authorization",
 "b0-sequence","b0-report","environment","weights-manifest","config"):
 parser.add_argument("--"+name,required=True)
parser.add_argument("--expected-request-sha256",required=True)
parser.add_argument("--output-relative",required=True)
parser.add_argument("--external-cache-root",required=True)
parser.add_argument("--frame",action="append",default=[])
parser.add_argument("--closure-file",action="append",default=[])
ns=parser.parse_args()
request_path=Path(ns.request).resolve(); expected=ns.expected_request_sha256
actual=hashlib.sha256(request_path.read_bytes()).hexdigest()
if actual != expected: raise ValueError("C05 request changed")
r=json.loads(request_path.read_text())
project=Path(ns.project_sentinel).resolve().parents[1]
source=Path(ns.source_sentinel).resolve().parents[1]
request_path.relative_to(project); source.relative_to(project)
sys.path.insert(0,str(project/"actionmesh"))
from prepare_c05_mode_bank import validate_exact_cache
closure={Path(path).resolve() for path in ns.closure_file}
expected_closure={request_path}
for ref in r["producer_input_refs"]+r["producer_code_refs"]:
 path=(project/ref["path"]).resolve(); path.relative_to(project)
 if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest()!=ref["sha256"]:
  raise ValueError("staged C05 closure changed: "+ref["path"])
 expected_closure.add(path)
if closure!=expected_closure or any(not path.is_file() for path in closure):
 raise ValueError("every staged C05 input/code file must be an explicit argv token")
output_relative=Path(ns.output_relative)
if (output_relative.is_absolute() or ".." in output_relative.parts
 or output_relative.as_posix()!=r["output_relative"]):
 raise ValueError("canonical request-bound workspace output required")
output=(project/output_relative).resolve(); output.relative_to(project)
frames=[Path(path).resolve() for path in ns.frame]
frame_root_relative=Path(r["frames_relative"])
if (frame_root_relative.is_absolute() or ".." in frame_root_relative.parts):
 raise ValueError("canonical staged C05 frame root required")
expected_frames=[(project/frame_root_relative/f"{i:02d}.png").resolve() for i in range(16)]
if (len(frames)!=16 or frames!=expected_frames
 or len({path.parent for path in frames})!=1 or any(path not in closure for path in frames)):
 raise ValueError("exact staged 16-frame directory required")
auth=Path(ns.authorization).resolve(); b0_sequence=Path(ns.b0_sequence).resolve()
b0_report=Path(ns.b0_report).resolve(); environment=Path(ns.environment).resolve()
weights_manifest=Path(ns.weights_manifest).resolve(); config=Path(ns.config).resolve()
for path in (auth,b0_sequence,b0_report,environment,weights_manifest,config):
 if path not in closure: raise ValueError("C05 runtime input was not explicitly staged")
bindings=((auth,r["authorization_ref"]),(b0_sequence,r["b0_sequence_ref"]),
 (b0_report,r["b0_report_ref"]),(environment,r["provenance"]["environment_ref"]),
 (weights_manifest,r["provenance"]["model_ref"]),(config,r["provenance"]["config_ref"]))
for path,ref in bindings:
 if path!=(project/ref["path"]).resolve():
  raise ValueError("C05 staged runtime binding differs from request: "+ref["path"])
source_relative=Path(r["source_root_relative"])
if (source_relative.is_absolute() or ".." in source_relative.parts
 or source!=(project/source_relative).resolve()):
 raise ValueError("staged C05 source root differs from request")
cache_root=Path(ns.external_cache_root).resolve()
rows=json.loads(weights_manifest.read_text())
if not isinstance(rows,list) or not rows: raise ValueError("nonempty staged weight manifest required")
manifest_paths={row["path"] for row in rows}
if (len(manifest_paths)!=len(rows)
 or {Path(path).parts[1] for path in manifest_paths}!={"ActionMesh","TripoSG","dinov2","RMBG"}):
 raise ValueError("exact four-root model cache manifest required")
initial_cache=validate_exact_cache(cache_root,rows)
pretrained=source/"pretrained_weights"
if pretrained.exists() or pretrained.is_symlink(): raise ValueError("staged source already owns pretrained_weights")
pretrained.mkdir()
for row in rows:
 relative=Path(row["path"]); link=pretrained/Path(*relative.parts[1:])
 link.parent.mkdir(parents=True,exist_ok=True)
 link.symlink_to((cache_root/relative).resolve())
linked={path.relative_to(pretrained).as_posix() for path in pretrained.rglob("*") if path.is_symlink()}
expected_links={Path(*Path(path).parts[1:]).as_posix() for path in manifest_paths}
if (linked!=expected_links
 or any(path.is_file() and not path.is_symlink() for path in pretrained.rglob("*"))):
 raise ValueError("staged model view differs from exact manifest")
os.chdir(source); sys.path.insert(0,str(source/"third_party/TripoSG"));sys.path.insert(0,str(source));sys.path.insert(0,str(project/"actionmesh"))
os.environ.update(HF_HUB_OFFLINE="1",TRANSFORMERS_OFFLINE="1",
                  HF_DATASETS_OFFLINE="1",HF_HUB_DISABLE_IMPLICIT_TOKEN="1")
import torch
if not torch.cuda.is_available() or torch.cuda.device_count()!=1:
    raise ValueError("Exactly one allocated CUDA device required; no CPU fallback")
from actionmesh.io.video_input import load_frames
from actionmesh.pipeline import ActionMeshPipeline
from prepare_c05_mode_bank import _authorization,digest
from research_math.c05_mode_bank import produce_same_anchor_mode_bank,validate_retained_mode_bank
if digest(auth)!=r["authorization_ref"]["sha256"]:
    raise ValueError("C05 authorization bytes changed")
_authorization(auth,uid=r["uid"],seed=r["outer_seed"],branches=r["branch_count"],
               gpu_uuid=r["provenance"]["gpu_uuid"],run_id=r["run_id"],
               environment_sha256=r["provenance"]["environment_ref"]["sha256"])
for ref in r["provenance"].values():
    if isinstance(ref,dict) and set(ref)=={"path","sha256"}:
        path=project/ref["path"]
        if digest(path)!=ref["sha256"]: raise ValueError("C05 provenance changed: "+ref["path"])
pipeline=ActionMeshPipeline(config_name=r["config_name"],
    config_dir=str(config.parent),dtype=torch.float16,lazy_loading=True).to("cuda")
result=produce_same_anchor_mode_bank(pipeline,load_frames(frames[0].parent,max_frames=16),
    output,outer_seed=r["outer_seed"],branch_count=r["branch_count"],
    generation_parameters=r["generation_parameters"],provenance=r["provenance"],
    producer_input_refs=r["producer_input_refs"]+[{"path":request_path.relative_to(project).as_posix(),"sha256":actual}],
    producer_code_refs=r["producer_code_refs"])
post_links={path.relative_to(pretrained).as_posix() for path in pretrained.rglob("*") if path.is_symlink()}
if (validate_exact_cache(cache_root,rows)!=initial_cache or post_links!=expected_links
 or any(path.is_file() and not path.is_symlink() for path in pretrained.rglob("*"))):
 raise ValueError("model cache/view inventory changed during C05 production")
validate_retained_mode_bank(output,project)
def arrays(path):
    with np.load(path,allow_pickle=False) as z:return {k:z[k].copy() for k in z.files}
left=arrays(output/"sequences/branch-0000.npz"); right=arrays(b0_sequence)
fields=("vertices","faces","frame_indices","timesteps","query_vertex_ids")
matches=set(left)==set(right)==set(fields) and all(np.array_equal(left[k],right[k]) for k in fields)
def exact_ref(path):
    p=Path(path).resolve(); p.relative_to(project)
    return {"path":p.relative_to(project).as_posix(),"sha256":digest(p)}
out=output
receipt={"kind":"c05-b0-parity-receipt","version":1,"candidate_id":r["candidate_id"],
 "uid":r["uid"],"outer_seed":r["outer_seed"],"matches":bool(matches),
 "comparison":"exact array equality for all native sequence fields",
 "mode_bank_result_ref":exact_ref(out/"result.json"),
 "mode_bank_manifest_ref":exact_ref(out/"raw-manifest.json"),
 "mode_bank_array_ref":exact_ref(out/"mode-bank.npz"),
 "branch_zero_ref":exact_ref(out/"sequences/branch-0000.npz"),
 "retained_b0_sequence_ref":exact_ref(b0_sequence),
 "retained_b0_report_ref":exact_ref(b0_report),
 "candidate_methods_tested":False,"scientific_effect_qualification":False,
 "native_qualified":False}
path=output.parent/(output.name+"-b0-parity.json")
path.write_text(json.dumps(receipt,indent=2)+"\n")
if not matches: raise ValueError("C05 branch zero differs from retained B0")
'''
    sys.path.insert(0, str(scripts))
    import run_experiments as native
    import run_harness as harness
    outputs = [
        str(output.relative_to(root) / "result.json"),
        str(output.relative_to(root) / "raw-manifest.json"),
        str(output.relative_to(root) / "mode-bank.npz"),
        str(output.relative_to(root).parent /
            (output.name + "-b0-parity.json")),
    ]
    outputs.extend(str(output.relative_to(root) / "sequences" /
                       f"branch-{index:04d}.npz") for index in range(branch_count))
    native_plan = native.make_plan(root, run_id=run_id, jobs=[{
        "trial_id": "c05-same-anchor-mode-bank", "command": (
            [sys.executable, "-c", program,
             "--request", str(request_path),
             "--expected-request-sha256", digest(request_path),
             "--project-sentinel", str(project_sources[-1]),
             "--source-sentinel", str(source_root / "actionmesh/pipeline.py"),
             "--authorization", str(authorization),
             "--b0-sequence", str(b0_sequence), "--b0-report", str(b0_report),
             "--environment", str(environment),
             "--weights-manifest", str(weights_manifest), "--config", str(config),
             "--output-relative", output.relative_to(root).as_posix(),
             "--external-cache-root", str(weights_manifest.parent)]
            + [item for path in [Path(frames).resolve() / f"{index:02d}.png"
                                 for index in range(16)] for item in ("--frame", str(path))]
            + [item for ref in [*refs, *code_refs] for item in
               ("--closure-file", str((root / ref["path"]).resolve()))]),
        "cwd": ".", "input_refs": refs, "code_refs": code_refs,
        "output_paths": outputs, "seed": outer_seed, "group": "candidate-artifacts-only",
        "arm_role": "same-anchor-empirical-mode-bank-no-native-admission"}],
        provenance={"git_revision": "exact code_refs/source refs",
            "model_revision": digest(weights_manifest),
            "data_revision": hashlib.sha256(json.dumps(frame_refs,
                sort_keys=True).encode()).hexdigest(),
            "environment_digest": digest(environment)},
        limits={"max_attempts": 1, "max_development_trials": 1,
            "max_confirmation_trials": 0, "max_retries_per_trial": 0,
            "wall_time_seconds": wall_seconds,
            "attempt_timeout_seconds": wall_seconds})
    native_path = plan_dir / "native.json"
    native_path.write_text(json.dumps(native_plan, indent=2) + "\n")
    outer = harness.make_plan(root, batch_id=run_id, tasks=[{
        "task_id": "c05-same-anchor-mode-bank", "idea_id": CANDIDATE_ID,
        "depends_on": [], "priority": 1, "plan_ref": file_ref(root, native_path),
        "resources": {"cpu_cores": 1, "ram_mib": 32768, "gpu_count": 1,
            "gpu_peak_mib": gpu_peak_mib, "allow_gpu_share": False,
            "memory_profile_ref": None,
            "exclusive_keys": ["gpu", "actionmesh-model-cache"]}}],
        limits={"total_wall_seconds": wall_seconds, "window_seconds": wall_seconds,
            "max_parallel_tasks": 1, "cpu_cores": 1, "ram_mib": 32768,
            "max_gpu_task_seconds": wall_seconds},
        gpus={"uuids": [read_json(environment)["gpu_uuid"]],
              "safety_margin_mib": 1024, "max_tasks_per_gpu": 1})
    (plan_dir / "harness.json").write_text(json.dumps(outer, indent=2) + "\n")
    return native_plan, outer


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("root", "skill-dir", "source-root", "frames", "b0-sequence",
                 "b0-report",
                 "authorization", "environment", "weights-manifest", "config",
                 "plan-dir", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--uid", required=True)
    parser.add_argument("--outer-seed", type=int, choices=(42, 314, 2718), required=True)
    parser.add_argument("--branch-count", type=int, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--wall-seconds", type=int, required=True)
    parser.add_argument("--gpu-peak-mib", type=int, required=True)
    args = parser.parse_args(argv)
    _, outer = build_plans(args.root, skill_dir=args.skill_dir,
        source_root=args.source_root, frames=args.frames,
        b0_sequence=args.b0_sequence, b0_report=args.b0_report,
        authorization=args.authorization,
        environment=args.environment, weights_manifest=args.weights_manifest,
        config=args.config, plan_dir=args.plan_dir, output=args.output,
        uid=args.uid, outer_seed=args.outer_seed, branch_count=args.branch_count,
        run_id=args.run_id, wall_seconds=args.wall_seconds,
        gpu_peak_mib=args.gpu_peak_mib)
    print(json.dumps({"plan": str(args.plan_dir.resolve() / "harness.json"),
        "approved_plan_digest": outer["plan_digest"], "execution_started": False,
        "source_delivery_status": "generated_unexecuted",
        "gpu_stop_must_remain_lifted_at_launch": True,
        "scientific_admission": False}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

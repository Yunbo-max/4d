"""H9 full native 30-step paired screening using existing census caches.

Four fresh three-branch rollouts share ONE initial noise, image context, anchor,
schedule and checkpoint. The published native two-branch result is retained as
a separately costed cached reference. All frames are decoded for the existing
census evaluator. No GT is accepted, no video/Stage0 is regenerated, no training
or automatic scientific-success claim is made.
"""
from __future__ import annotations
import argparse
import gc
import json
import os
from pathlib import Path
import shutil
import sys
import time
import traceback

import numpy as np

from .case_io import BASE_COMMIT, digest, load_census_case, save_sequence, write_json

ARMS = ("scalar", "projection", "norm_matched", "random")


def scheduler_parameters(config):
    """Never silently inherit SchedulerFlow's subtractive default."""
    values = config.get("model", {}).get("scheduler", {})
    keys = {"num_inference_steps", "num_train_timesteps", "shift", "is_additive", "split_cfg_batch"}
    if (not isinstance(values, dict) or keys - values.keys()
            or set(values) - keys - {"_target_", "_partial_"}
            or values.get("_target_") != "actionmesh.scheduler.scheduler.SchedulerFlow"
            or config.get("stage_1_steps") != 30 or values.get("num_inference_steps") != 30
            or values.get("is_additive") is not True or values.get("split_cfg_batch") is not True
            or not isinstance(values.get("num_train_timesteps"), int) or values["num_train_timesteps"] < 1
            or not np.isfinite(values.get("shift", np.nan)) or values["shift"] <= 0):
        raise ValueError("explicit official lowram 30-step additive, split-CFG configuration required")
    return {key: values[key] for key in sorted(keys)}


def add_arguments(parser):
    parser.add_argument("--root", type=Path, required=True, help="Existing model root containing repo/")
    parser.add_argument("--case-dir", type=Path, required=True, help="Completed native census case, any cohort UID")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--max-seconds", type=float, default=1800.)
    parser.add_argument("--memory-cap-mib", type=int, default=20480)


def run_guidance(args):
    if (not np.isfinite(args.max_seconds) or not 0 < args.max_seconds <= 7200
            or not 8192 <= args.memory_cap_mib <= 22528):
        raise ValueError("time budget must be <=7200s and allocator cap within 8192..22528MiB")
    root, source_dir, output = (Path(p).resolve() for p in (args.root, args.case_dir, args.output))
    if output.exists():
        raise FileExistsError("output already exists: " + str(output))
    source, hashes, cached_latents, original, faces, prepared = load_census_case(source_dir)
    if source.get("stage1_steps") != 30 or source.get("config") != "actionmesh_lowram":
        raise ValueError("source must be an official 30-step lowram inference case")
    repo = root/"repo"
    provenance_path = source_dir/"code-provenance.json"
    provenance = json.loads(provenance_path.read_text())
    names = ("actionmesh/pipeline.py", "actionmesh/model/temporal_autoencoder.py",
        "actionmesh/model/temporal_denoiser.py", "actionmesh/scheduler/scheduler.py",
        "actionmesh/configs/actionmesh.yaml", "actionmesh/configs/actionmesh_lowram.yaml")
    code_hashes = {name: digest(repo/name) for name in names}
    for name, current in code_hashes.items():
        if current != provenance.get("sha256", {}).get(name):
            raise ValueError("native backbone/config changed: " + name)
    hashes["code-provenance.json"] = digest(provenance_path)
    output.mkdir(parents=True, exist_ok=False)
    arm_names = ("official_cached", *ARMS)
    manifest = {"schema_version": 1, "method": 9, "cases": [dict(
        case_id="h09-"+mode, uid=source["uid"], case_dir="variants/"+mode) for mode in arm_names]}
    write_json(output/"manifest.json", manifest)
    rows = {}
    for mode in arm_names:
        folder = output/"variants"/mode
        folder.mkdir(parents=True, exist_ok=False)
        rows[mode] = dict(status="pending", uid=source["uid"], seed=source["seed"],
            frames=16, method=9, arm=mode, training=False, gt_used=False,
            new_stage1_branch_calls=0, branch_forward_returns=0)
        write_json(folder/"report.json", rows[mode])
    report = dict(status="running", method=9, uid=source["uid"], seed=source["seed"],
        source_case_dir=str(source_dir), source_hashes=hashes, code_sha256=code_hashes,
        runner_sha256=digest(__file__), guidance_sha256=digest(Path(__file__).with_name("m09_guidance.py")),
        source_base_commit=BASE_COMMIT, previous_results_reinterpreted=False,
        input_evidence="existing_natural_census_cache", training=False, gt_used=False,
        cached_denoiser_checkpoint_identity_verified=False,
        natural_quality_claim=False, max_seconds=args.max_seconds, memory_cap_mib=args.memory_cap_mib,
        decode_costs={},
        arms=rows, protocol=dict(inference_steps=30, fresh_branch_calls_per_live_arm=90,
            branch_order=[[0, 0], [0, 1], [1, 1]], guidance_scales=[1., 7.5],
            interventions=list(ARMS), reused_stage0=True, reused_image_context=True,
            same_initial_noise=True, unchanged_stageII=True, cached_reference_new_stageI_calls=0),
        limitations=["This is screening, not novelty validation or a pass of research-autopilot gates.",
            "APG/CFG++ strong-baseline qualification and an independent held-out cohort remain open.",
            "Full quality metrics must be computed separately; velocity/latent disagreement is not accuracy.",
            "Time budget is checked at native branch/decoder callbacks; a hung GPU kernel is not preempted."])
    started = time.monotonic()
    monitor = None
    torch = None
    def persist():
        for mode, row in rows.items():
            write_json(output/"variants"/mode/"report.json", row)
        write_json(output/"report.json", report)
    def check(stage):
        elapsed = time.monotonic()-started
        write_json(output/"progress.json", dict(stage=stage, elapsed_seconds=elapsed))
        print(json.dumps({"stage": stage, "elapsed_seconds": round(elapsed, 2)}), flush=True)
        if elapsed > args.max_seconds:
            raise TimeoutError("full-pair time budget exceeded; no reduced-step fallback")
    persist()
    try:
        os.environ.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", HF_DATASETS_OFFLINE="1")
        sys.path.insert(0, str(repo))
        import torch
        from omegaconf import OmegaConf
        from actionmesh.utils import load_config
        from actionmesh.scheduler.scheduler import SchedulerFlow
        from actionmesh.scheduler.guidance import ClassifierFreeGuidance
        from actionmesh.model.temporal_autoencoder import ActionMeshAutoencoder
        from actionmesh.model.temporal_denoiser import ActionMeshDenoiser
        from research_three_ideas import Resources
        from .actionmesh_adapter import ActionMeshDecoderAdapter
        from .m09_guidance import make_torch_guidance
        device = torch.device(args.device)
        if device.type != "cuda" or device.index is None or not torch.cuda.is_available():
            raise RuntimeError("explicit available CUDA device required, e.g. cuda:0")
        torch.cuda.set_device(device)
        torch.set_num_threads(2)
        free, total = torch.cuda.mem_get_info(device)
        if free < 14*1024**3:
            raise RuntimeError("require >=14GiB free; no competing process is stopped")
        torch.cuda.set_per_process_memory_fraction(min(1., args.memory_cap_mib*1024**2/total), device)
        monitor = Resources(output, torch, str(device.index))
        monitor.start()
        cfg = OmegaConf.to_container(load_config("actionmesh_lowram", str(repo/"actionmesh/configs")), resolve=True)
        parameters = scheduler_parameters(cfg)
        native_cfg = cfg["model"]["cf_guidance"]
        if (native_cfg["guidance_at_inference"] != [[0, 1], [1, 1]]
                or native_cfg["guidance_scales"] != [7.5] or native_cfg["inference_enabled"] is not True):
            raise ValueError("unexpected official two-branch reference guidance")
        report["scheduler_parameters"] = parameters
        report["versions"] = dict(torch=torch.__version__, numpy=np.__version__, cuda=torch.version.cuda)
        write_json(output/"resolved-config.json", cfg)
        checkpoint_hashes = {}
        for kind in ("autoencoder", "denoiser"):
            check("hash_local_"+kind+"_checkpoint")
            directory = repo/"pretrained_weights/ActionMesh"/kind
            files = [p for p in sorted(directory.rglob("*")) if p.is_file() and ".cache" not in p.parts]
            if not files:
                raise FileNotFoundError("local checkpoint absent: " + str(directory))
            for path in files:
                check("hash_"+kind+"_"+path.name)
                checkpoint_hashes[str(path.relative_to(repo))] = digest(path)
        report["checkpoint_sha256"] = checkpoint_hashes
        def load_frozen(model_type, kind):
            check("load_frozen_"+kind)
            model = model_type.from_pretrained(str(repo/"pretrained_weights/ActionMesh"/kind),
                local_files_only=True).eval().to(device)
            for parameter in model.parameters():
                parameter.requires_grad_(False)
            if any(p.is_floating_point() and p.dtype != torch.float32 for p in model.parameters()):
                raise ValueError("native checkpoint parameter storage must stay FP32")
            return model
        def decode(decoder, latents, mode):
            cost = dict(decoder_forward_attempts=1, target_attempts=0, target_iterations_returned=0,
                point_targets_attempted=0, validated_completed_targets=0, validated_completed_point_targets=0,
                completion_validated=False)
            report["decode_costs"][mode] = cost
            begin = time.monotonic()
            def callback(step, count):
                # Callback precedes this target; prior Python iteration returns
                # do not prove completion of asynchronous CUDA kernels.
                cost["target_iterations_returned"] = step-1
                check(mode+"_decode_"+str(step)+"_of_"+str(count))
                cost["target_attempts"] = step
                cost["point_targets_attempted"] = step*original.shape[1]
                persist()
            try:
                adapter = ActionMeshDecoderAdapter(decoder, latents, prepared["timesteps"],
                    device=str(device), autocast=True, step_callback=callback)
                values = adapter(0., prepared["timesteps"][1:], prepared["anchor_query_features"][:, :3],
                    prepared["anchor_query_features"][:, 3:], prepared["query_vertex_ids"])
                torch.cuda.synchronize(device)
                if values.shape != original[1:].shape:
                    raise ValueError("decoder did not return every material vertex/frame")
                cost.update(target_iterations_returned=15, validated_completed_targets=15,
                    validated_completed_point_targets=15*original.shape[1], completion_validated=True)
                return np.concatenate((original[:1], values.astype(original.dtype)), axis=0)
            finally:
                cost["elapsed_seconds"] = time.monotonic()-begin
                if mode in rows:
                    rows[mode]["stage2_seconds"] = cost["elapsed_seconds"]
                    rows[mode]["decode_cost"] = cost
                persist()
        # Qualify the exact frozen decoder and cached queries before spending
        # any fresh Stage-I budget. This replay is separately recorded and paid.
        decoder = load_frozen(ActionMeshAutoencoder, "autoencoder")
        replay = decode(decoder, cached_latents, "cached_reference_replay")
        diagonal = float(np.linalg.norm(np.ptp(original[0].astype(float), axis=0)))
        if not np.isfinite(diagonal) or diagonal <= 0:
            raise ValueError("zero-size reference anchor")
        parity = float(np.sqrt(np.mean((replay.astype(float)-original)**2))/diagonal)
        report["decoder_replay_rms_over_diagonal"] = parity
        report["replay_extra_point_targets"] = 15*original.shape[1]
        if parity > 1e-4:
            raise ValueError("frozen decoder does not reproduce source; repair provenance before screening")
        folder = output/"variants/official_cached"
        shutil.copyfile(source_dir/"sequence.npz", folder/"sequence.npz")
        rows["official_cached"].update(status="completed", sha256={"sequence.npz": digest(folder/"sequence.npz")},
            reused=True, new_stage1_branch_calls=0, source_generation_report_sha256=hashes["report.json"],
            cached_native_stage1_steps=30, cached_native_cfg_branches=2,
            source_stage1_branch_calls=60, source_stage_seconds=source.get("stage_seconds"), anchor_exact=True)
        del decoder, replay
        gc.collect(); torch.cuda.empty_cache()
        scheduler = SchedulerFlow(**parameters)
        denoiser = load_frozen(ActionMeshDenoiser, "denoiser")
        context = torch.from_numpy(prepared["context"]).to(device)[None]
        clock = torch.from_numpy(prepared["timesteps"]).to(device)[None]
        anchor = torch.from_numpy(prepared["anchor_latent"][0]).float().to(device)
        mask = torch.zeros((1, 16), device=device)
        mask[:, 0] = 1.
        initial = scheduler.get_noise([2048, 64], 1, 16, str(device),
            generator=torch.Generator(device=device).manual_seed(source["seed"]))
        initial[:, 0] = anchor
        np.savez_compressed(output/"shared-initial-noise.npz", latents=initial.cpu().numpy())
        report["shared_initial_noise_sha256"] = digest(output/"shared-initial-noise.npz")
        original_forward = denoiser.forward
        for mode in ARMS:
            row = rows[mode]
            row["status"] = "running"
            calls = []
            def forward(*a, **kw):
                check(mode+"_stage1_branch_"+str(len(calls)+1))
                calls.append(float(kw["diffusion_time"][0]))
                row["new_stage1_branch_calls"] = len(calls)
                write_json(output/"variants"/mode/"report.json", row)
                result = original_forward(*a, **kw)
                row["branch_forward_returns"] += 1
                return result
            denoiser.forward = forward
            base = ClassifierFreeGuidance(guidance_at_inference=[[0, 0], [0, 1], [1, 1]], guidance_scales=[1., 7.5])
            hook = make_torch_guidance(base, mode=mode, seed=source["seed"])
            begin = time.monotonic()
            try:
                with torch.inference_mode(), torch.autocast("cuda", dtype=torch.float16):
                    result = scheduler.denoise(denoiser, hook, initial.clone(), context,
                        device=device, disable_prog=True, mask=mask, framestep=clock)
                torch.cuda.synchronize(device)
            finally:
                denoiser.forward = original_forward
                row.update(stage1_seconds=time.monotonic()-begin, guidance_records=hook.records,
                    diffusion_times=calls, completed_guidance_steps=len(hook.records))
                persist()
            if (len(calls) != 90 or len(hook.records) != 30 or not torch.isfinite(result).all()
                    or not torch.equal(result[:, 0], initial[:, 0])):
                raise ValueError("30 fresh three-branch steps, finite output and exact anchor required")
            values = result[0].float().cpu().numpy()
            folder = output/"variants"/mode
            np.savez_compressed(folder/"denoised.npz", latents=values, timesteps=prepared["timesteps"], seed=np.asarray(source["seed"]))
            row.update(status="stage1_completed", stage1_seconds=time.monotonic()-begin, inference_steps=30,
                new_stage1_branch_calls=len(calls), stage1_anchor_exact=True, guidance_records=hook.records,
                diffusion_times=calls, denoised_sha256=digest(folder/"denoised.npz"))
            if mode == "scalar":
                row["matched_scalar_vs_cached_latent_rms"] = float(np.sqrt(np.mean((values.astype(float)-cached_latents)**2)))
                row["comparison_note"] = ("Matched scalar is the causal baseline. Historical cached reference may differ "
                    "in aggregation precision or checkpoint/runtime identity; source did not hash denoiser checkpoints.")
            persist()
            del result, values, hook
        if any(p.requires_grad or p.grad is not None for p in denoiser.parameters()):
            raise ValueError("denoiser freezing invariant failed")
        report["denoiser_frozen_no_gradients"] = True
        # Bound methods keep a reference to the model; release them as well.
        del denoiser, original_forward, forward, context, initial, anchor
        gc.collect(); torch.cuda.empty_cache()
        decoder = load_frozen(ActionMeshAutoencoder, "autoencoder")
        for mode in ARMS:
            folder = output/"variants"/mode
            begin = time.monotonic()
            with np.load(folder/"denoised.npz", allow_pickle=False) as saved:
                latents = saved["latents"].copy()
            sequence = decode(decoder, latents, mode)
            row = rows[mode]
            row.update(status="completed", stage2_seconds=time.monotonic()-begin,
                sha256={"sequence.npz": save_sequence(folder, sequence, faces, prepared["timesteps"], prepared["query_vertex_ids"])},
                stage2_point_targets=15*original.shape[1], anchor_exact=np.array_equal(sequence[0], original[0]),
                faces_unchanged=True, all_frames_once=True)
            persist()
            del latents, sequence
        if any(p.requires_grad or p.grad is not None for p in decoder.parameters()):
            raise ValueError("decoder freezing invariant failed")
        report["decoder_frozen_no_gradients"] = True
        try:
            unchanged = all(digest(source_dir/name) == expected for name, expected in hashes.items())
            unchanged = unchanged and all(digest(repo/name) == expected for name, expected in code_hashes.items())
            unchanged = unchanged and all(digest(repo/name) == expected for name, expected in checkpoint_hashes.items())
        except Exception:
            for row in rows.values():
                row.update(status="failed", error="end-of-run source/code/checkpoint verification failed; all arms invalidated")
            raise
        report["source_hashes_unchanged"] = unchanged
        if not unchanged:
            for row in rows.values():
                row.update(status="failed", error="source/code/checkpoint changed; all arms invalidated")
            raise ValueError("source/code/checkpoint changed during the paired run")
        report.update(status="completed", source_hashes_unchanged=True,
            new_stage1_branch_calls_total=sum(rows[mode]["new_stage1_branch_calls"] for mode in ARMS),
            stage2_point_targets_total=75*original.shape[1])
    except Exception as error:
        report.update(status="failed", error=str(error), traceback=traceback.format_exc())
        for row in rows.values():
            if row["status"] != "completed":
                row.update(status="failed", error="pair incomplete: "+str(error))
    finally:
        if monitor is not None:
            try:
                report["resources"] = monitor.finish()
            except Exception as error:
                report["resource_collection_error"] = str(error)
        report["elapsed_seconds"] = time.monotonic()-started
        report["new_stage1_branch_calls_total"] = sum(rows[mode]["new_stage1_branch_calls"] for mode in ARMS)
        report["stage2_point_targets_attempted_total"] = sum(c["point_targets_attempted"] for c in report["decode_costs"].values())
        report["stage2_validated_completed_point_targets"] = sum(c["validated_completed_point_targets"] for c in report["decode_costs"].values())
        persist()
    return int(report["status"] != "completed")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    add_arguments(parser)
    args = parser.parse_args(argv)
    try:
        return run_guidance(args)
    except (ValueError, FileExistsError) as error:
        parser.error(str(error))


if __name__ == "__main__":
    raise SystemExit(main())

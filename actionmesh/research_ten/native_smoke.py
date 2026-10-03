"""Bounded native GPU integration check for H2 and H9, not an efficacy benchmark.

Uses existing verified ActionMesh census caches. H2 executes real source/target
decoder compositions. Optional H9 executes two independent TWO-STEP Stage-I
rollouts from identical noise with scalar and projected 3-branch guidance.
Two steps are explicitly an interface check, not the official 30-step protocol.
No GT, fresh downloads, model training, or existing-output overwrite.
"""
from __future__ import annotations
import argparse
import gc
import hashlib
import json
import os
from pathlib import Path
import sys
import time
import traceback
import numpy as np


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8*1024*1024), b""):
            h.update(block)
    return h.hexdigest()


def serial(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(k): serial(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [serial(v) for v in value]
    return value


def save(path, value):
    temp = path.with_suffix(path.suffix+".tmp")
    temp.write_text(json.dumps(serial(value), indent=2, allow_nan=False)+"\n")
    temp.replace(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--case-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--points", type=int, default=32)
    parser.add_argument("--max-seconds", type=float, default=600.)
    parser.add_argument("--native-guidance", action="store_true")
    args = parser.parse_args()
    if not 4 <= args.points <= 128 or not 0 < args.max_seconds <= 1200:
        parser.error("use 4..128 points and <=1200 seconds for this integration check")
    args.output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    report = {"status": "running", "training": False, "gt_used": False,
              "scope": "Native integration check, not full benchmark/efficacy validation",
              "script_sha256": digest(__file__), "argv": sys.argv}
    def check(stage):
        save(args.output/"progress.json", {"stage": stage, "seconds": time.monotonic()-started})
        if time.monotonic()-started > args.max_seconds:
            raise TimeoutError("native smoke time cap reached")
        print(stage, flush=True)
    monitor = None
    try:
        os.environ.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", HF_DATASETS_OFFLINE="1")
        sys.path[:0] = [str(args.root/"repo"), str(args.root/"research/census-20261002")]
        import torch
        import trimesh
        from research_census_time_direction import load_source
        from research_three_ideas import Resources
        from actionmesh.model.temporal_autoencoder import ActionMeshAutoencoder
        from actionmesh.preprocessing.mesh_processor import get_mesh_features
        from research_ten.actionmesh_adapter import ActionMeshDecoderAdapter
        from research_ten.m02_cycles import probe_cycles
        from research_ten.m09_guidance import make_torch_guidance
        if not torch.cuda.is_available():
            raise RuntimeError("native integration requires CUDA")
        torch.cuda.set_device(0)
        torch.set_num_threads(2)
        free, total = torch.cuda.mem_get_info()
        if free < 14*1024**3:
            raise RuntimeError("require >=14GiB free; no competing task is stopped")
        torch.cuda.set_per_process_memory_fraction(min(1.,20*1024**3/total))
        torch.manual_seed(42)
        monitor = Resources(args.output, torch, "0")
        monitor.start()
        check("verify_cached_source")
        source, hashes, latent, vertices, faces, prepared = load_source(args.case_dir)
        report.update(uid=source["uid"], source_hashes=hashes,
                      input_frames=16, probe_times=[0.,5.,10.,15.], query_points=args.points)
        provenance = json.loads((args.case_dir/"code-provenance.json").read_text())
        for name in ("actionmesh/model/temporal_autoencoder.py", "actionmesh/pipeline.py"):
            if digest(args.root/"repo"/name) != provenance["sha256"][name]:
                raise ValueError("backbone source changed: "+name)
        check("load_frozen_temporal_decoder")
        decoder = ActionMeshAutoencoder.from_pretrained(
            str(args.root/"repo/pretrained_weights/ActionMesh/autoencoder"),
            local_files_only=True).eval().cuda()
        decoder.verbose = False
        for parameter in decoder.parameters():
            parameter.requires_grad_(False)
        adapter = ActionMeshDecoderAdapter(decoder, latent, np.arange(16,dtype=float),
                                            device="cuda:0", autocast=True)
        ids = np.linspace(0,len(vertices[0])-1,args.points,dtype=np.int64)
        source_normals = prepared["anchor_query_features"][:,3:]
        check("native_decoder_parity")
        native = adapter(0., [5.,10.,15.], vertices[0,ids],source_normals[ids],ids)
        diagonal = float(np.linalg.norm(np.ptp(vertices[0],axis=0)))
        parity = float(np.sqrt(np.mean((native-vertices[[5,10,15]][:,ids])**2))/diagonal)
        report["decoder_parity_rms_over_diagonal"] = parity
        if not np.isfinite(parity) or parity > 1e-4:
            raise ValueError("native decoder adapter does not reproduce cached query output")
        def official_normals(v, f):
            mesh = trimesh.Trimesh(vertices=v, faces=f, process=False)
            return get_mesh_features(mesh, with_normals=True).numpy()[:,3:]
        check("native_cycle_compositions")
        cycles = probe_cycles(adapter, vertices[0], faces, [0.,5.,10.,15.], ids,
                              max_query_points=32768, normal_fn=official_normals,
                              source_normals=source_normals)
        arrays = {key: value for key,value in cycles.items() if isinstance(value,np.ndarray)}
        np.savez_compressed(args.output/"native-cycles.npz", **arrays)
        report["cycles"] = {key:value for key,value in cycles.items() if not isinstance(value,np.ndarray)}
        report["cycle_array_shapes"] = {key:list(value.shape) for key,value in arrays.items()}
        report["parity_extra_point_targets"] = 3*len(ids)
        report["decoder_frozen"] = all(not p.requires_grad for p in decoder.parameters())
        report["decoder_gradients_absent"] = all(p.grad is None for p in decoder.parameters())
        del adapter, decoder
        gc.collect(); torch.cuda.empty_cache()
        if args.native_guidance:
            check("load_frozen_temporal_denoiser")
            from actionmesh.model.temporal_denoiser import ActionMeshDenoiser
            from actionmesh.scheduler.scheduler import SchedulerFlow
            from actionmesh.scheduler.guidance import ClassifierFreeGuidance
            denoiser = ActionMeshDenoiser.from_pretrained(
                str(args.root/"repo/pretrained_weights/ActionMesh/denoiser"),
                local_files_only=True).eval().cuda()
            for parameter in denoiser.parameters():
                parameter.requires_grad_(False)
            with np.load(args.case_dir/"prepared.npz", allow_pickle=False) as data:
                context = torch.from_numpy(data["context"].copy()).cuda()[None]
            anchor = torch.from_numpy(prepared["anchor_latent"][0].copy()).float().cuda()
            mask = torch.zeros((1,16),device="cuda")
            mask[:,0] = 1.
            times = torch.arange(16,dtype=torch.float32,device="cuda")[None]
            scheduler = SchedulerFlow(num_inference_steps=2, split_cfg_batch=True)
            noise = scheduler.get_noise([2048,64],1,16,device="cuda:0",
                                         generator=torch.Generator(device="cuda").manual_seed(42))
            noise[:,0] = anchor
            outputs = {}
            report["guidance"] = {}
            original_forward = denoiser.forward
            for mode in ("scalar", "projection"):
                check("native_guidance_two_steps_"+mode)
                calls = []
                def tracked_forward(*a, **kw):
                    check("denoiser_"+mode+"_branch_"+str(len(calls)+1))
                    calls.append({"diffusion_time":float(kw["diffusion_time"][0]),
                                  "input_mean":float(kw["hidden_states"].float().mean())})
                    return original_forward(*a,**kw)
                denoiser.forward = tracked_forward
                base = ClassifierFreeGuidance(guidance_at_inference=[[0,0],[0,1],[1,1]],
                                               guidance_scales=[1.,7.5])
                guidance = make_torch_guidance(base, mode=mode)
                with torch.inference_mode(), torch.autocast("cuda",dtype=torch.float16):
                    output = scheduler.denoise(denoiser,guidance,noise.clone(),context,
                        device="cuda:0",disable_prog=True,mask=mask,framestep=times)
                torch.cuda.synchronize()
                if len(calls)!=6 or not torch.isfinite(output).all():
                    raise ValueError("expected six finite branch evaluations per two-step run")
                if not torch.equal(output[:,0],noise[:,0]):
                    raise ValueError("known anchor changed")
                outputs[mode] = output.detach().cpu().numpy()
                report["guidance"][mode] = {"inference_steps":2,"branch_calls":len(calls),
                    "anchor_bitwise_unchanged":True,"calls":calls,
                    "records":getattr(guidance,"records",None)}
            denoiser.forward = original_forward
            report["guidance"]["scalar_projection_rms_difference"] = float(
                np.sqrt(np.mean((outputs["scalar"].astype(float)-outputs["projection"])**2)))
            report["guidance"]["scope"] = "Two independent native TWO-STEP rollouts; no final mesh or 30-step quality claim"
            report["denoiser_frozen"] = all(not p.requires_grad for p in denoiser.parameters())
            report["denoiser_gradients_absent"] = all(p.grad is None for p in denoiser.parameters())
            np.savez_compressed(args.output/"guidance-two-step.npz",**outputs)
        for name, expected in hashes.items():
            if digest(args.case_dir/name) != expected:
                raise ValueError("source changed during check: "+name)
        report.update(status="completed",source_hashes_unchanged=True)
    except Exception as error:
        report.update(status="failed",error=str(error),traceback=traceback.format_exc())
        raise
    finally:
        if monitor is not None:
            report["resources"] = monitor.finish()
        report["elapsed_seconds"] = time.monotonic()-started
        save(args.output/"report.json",report)


if __name__ == "__main__":
    main()

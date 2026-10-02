#!/usr/bin/env python3
"""Bounded, offline Wan 2.2 TI2V-5B frontend for ActionMesh application tests.

Example (run with the isolated GPU environment's Python):
  python generate_application_video.py --root /path/to/actionmesh-repro \
    --mode image --image mushroom.png --prompt 'A mushroom singing opera.' \
    --output /path/to/new-result-directory

CPU FP32 UMT5 encoding and GPU video generation run in separate processes.
This tests a chosen frontend, not the precise video model used by ActionMesh's
authors. A passing technical check is not a semantic or benchmark evaluation.
"""

import argparse
import csv
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import signal
import subprocess
import sys
import time
import traceback


MODEL_NAME = "Wan2.2-TI2V-5B-Diffusers"
NEGATIVE_PROMPT = (
    "overexposed, static, blurred details, subtitles, watermark, worst quality, "
    "low quality, JPEG compression artifacts, deformed, disfigured, extra limbs, "
    "fused limbs, still picture, messy background, many subjects, camera movement"
)


def utc_now():
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def write_json(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")
    temporary.replace(path)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def finite(tensor, label):
    import torch

    if not bool(torch.isfinite(tensor).all().item()):
        raise FloatingPointError(f"Non-finite values in {label}")


def cuda_stats(torch):
    # The CPU-only text stage must not initialize CUDA just for telemetry.
    if not torch.cuda.is_initialized():
        return {"cuda_initialized": False, "peak_allocated_bytes": 0,
                "peak_reserved_bytes": 0}
    torch.cuda.synchronize()
    return {"cuda_initialized": True,
            "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
            "peak_reserved_bytes": torch.cuda.max_memory_reserved()}


def versions():
    result = {"python": sys.version, "platform": platform.platform()}
    for package in ("torch", "diffusers", "transformers", "accelerate", "safetensors"):
        try:
            result[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            result[package] = None
    return result


def encode_text(config, out):
    import torch
    from diffusers import WanPipeline
    from safetensors.torch import save_file
    from transformers import AutoTokenizer, UMT5EncoderModel

    torch.set_num_threads(config["cpu_threads"])
    model_dir = config["model_directory"]
    tokenizer = AutoTokenizer.from_pretrained(
        model_dir, subfolder="tokenizer", local_files_only=True
    )
    encoder = UMT5EncoderModel.from_pretrained(
        model_dir, subfolder="text_encoder", torch_dtype=torch.float32,
        low_cpu_mem_usage=True, local_files_only=True,
    ).eval()
    # Calling the upstream method preserves prompt_clean(), attention-mask
    # lengths, and zero padding exactly. No hand-written tokenizer substitute.
    encoder_pipe = WanPipeline(
        tokenizer=tokenizer, text_encoder=encoder, vae=None,
        scheduler=None, transformer=None, expand_timesteps=True,
    )
    with torch.inference_mode():
        positive, negative = encoder_pipe.encode_prompt(
            prompt=config["prompt"], negative_prompt=config["negative_prompt"],
            do_classifier_free_guidance=config["guidance_scale"] > 1,
            max_sequence_length=512, device=torch.device("cpu"),
            dtype=torch.float32,
        )
    embeddings = {"prompt_embeds": positive.contiguous()}
    if negative is not None:
        embeddings["negative_prompt_embeds"] = negative.contiguous()
    for key, tensor in embeddings.items():
        finite(tensor, f"CPU FP32 {key}")
        finite(tensor.to(torch.float16), f"FP16 conversion of {key}")
    save_file(embeddings, str(out / "prompt-embeddings.safetensors"), metadata={
        "model": "Wan-AI/" + MODEL_NAME,
        "dtype": "float32", "max_sequence_length": "512",
        "prompt_sha256": hashlib.sha256(config["prompt"].encode()).hexdigest(),
    })
    return {"encoder_device": "cpu", "encoder_dtype": "float32",
            "max_sequence_length": 512, "finite_before_and_after_fp16_cast": True,
            "embeddings": {k: list(v.shape) for k, v in embeddings.items()}}


def prepare_image(config, out):
    from PIL import Image, ImageOps

    with Image.open(config["image"]) as source:
        source = ImageOps.exif_transpose(source).convert("RGBA")
        background = Image.new("RGBA", source.size, (255, 255, 255, 255))
        background.alpha_composite(source)
        image = background.convert("RGB")
    image = ImageOps.pad(
        image, (config["width"], config["height"]),
        method=Image.Resampling.LANCZOS, color="white",
    )
    image.save(out / "input.png")
    return image


def validate_and_export(frames, config, out):
    import imageio.v2 as imageio
    import numpy as np
    from PIL import Image
    from diffusers.utils import export_to_video

    frames = np.asarray(frames)
    expected = (config["frames"], config["height"], config["width"], 3)
    if frames.shape != expected or frames.shape[0] < 16:
        raise RuntimeError(f"Unexpected video shape {frames.shape}; expected {expected}")
    if not np.isfinite(frames).all():
        raise FloatingPointError("Non-finite decoded video frames")
    # Pipeline output_type='np' is RGB float data in [0,1]. Check before conversion.
    if float(frames.min()) < -1e-5 or float(frames.max()) > 1.00001:
        raise RuntimeError("Decoded video is outside the expected [0, 1] range")
    pixels = np.rint(frames * 255).clip(0, 255).astype(np.uint8)
    means = pixels.mean(axis=(1, 2, 3))
    max_pixels = pixels.max(axis=(1, 2, 3))
    delta = np.abs(pixels[1:].astype(np.int16) - pixels[:-1].astype(np.int16))
    checks = {
        "frame_count_matches_request": True, "at_least_16_frames": True,
        "decoded_frames_finite": True,
        "no_black_frames": bool(np.all(means > 0.5) and np.all(max_pixels > 2)),
        "not_pixel_identical_static": bool(np.any(delta)),
    }
    validation = {
        "checks": checks, "frame_shape": list(frames.shape),
        "mean_pixel_value_per_frame": means.tolist(),
        "mean_absolute_adjacent_frame_difference_8bit": float(delta.mean()),
        "max_adjacent_frame_difference_8bit": int(delta.max()),
        "semantic_evaluation": "not_performed_requires_visual_review",
        "scope": "runtime_smoke_test_only" if config["smoke"] else "short_clip_application_test",
        "full_resolution_benchmark": False,
    }
    write_json(out / "validation.json", validation)
    # Preserve generated evidence even when a quality guard rejects the result.
    frame_dir = out / "frames"
    frame_dir.mkdir(exist_ok=False)
    images = []
    for index, frame in enumerate(pixels):
        image = Image.fromarray(frame)
        image.save(frame_dir / f"{index:03d}.png")
        image.thumbnail((384, 384), Image.Resampling.LANCZOS)
        images.append(image)
    images[0].save(
        out / "preview.gif", save_all=True, append_images=images[1:],
        duration=round(1000 / config["fps"]), loop=0,
    )
    # Diffusers multiplies numpy input by 255 internally; feed original [0,1]
    # floats, not already quantized uint8 (which would overflow on a second multiply).
    export_to_video(list(frames), str(out / "generated.mp4"), fps=config["fps"])
    # Re-open the actual encoded artifact rather than only checking the source array.
    reader = imageio.get_reader(str(out / "generated.mp4"))
    try:
        decoded_count = sum(1 for _ in reader)
    finally:
        reader.close()
    if decoded_count != config["frames"]:
        raise RuntimeError(f"MP4 readback yielded {decoded_count} frames")
    validation["mp4_readback_frames"] = decoded_count
    write_json(out / "validation.json", validation)
    failed = [name for name, passed in checks.items() if not passed]
    if failed:
        raise RuntimeError("Video validation failed: " + ", ".join(failed))
    return validation


def generate_video(config, out):
    import torch
    from diffusers import AutoencoderKLWan, WanImageToVideoPipeline, WanPipeline
    from diffusers import WanTransformer3DModel
    from safetensors.torch import load_file

    torch.set_num_threads(config["cpu_threads"])
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU is required for the video stage")
    torch.cuda.set_device(0)
    torch.cuda.reset_peak_memory_stats()
    gpu = torch.cuda.get_device_properties(0)
    write_json(out / "gpu.json", {
        "name": gpu.name, "total_memory_bytes": gpu.total_memory,
        "capability": [gpu.major, gpu.minor], "torch_cuda": torch.version.cuda,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
    })
    model_dir = config["model_directory"]
    transformer = WanTransformer3DModel.from_pretrained(
        model_dir, subfolder="transformer", torch_dtype=torch.float16,
        local_files_only=True, low_cpu_mem_usage=True,
    ).eval()
    if transformer.dtype != torch.float16:
        raise RuntimeError(f"Transformer compute dtype is {transformer.dtype}, expected float16")
    # Diffusers declares these modules as FP32 for numerical stability. Ensure
    # the loader honored that declaration, including with derived FP16 shards.
    keep_fp32 = tuple(transformer._keep_in_fp32_modules)
    preserved = []
    for name, parameter in transformer.named_parameters():
        if any(part in name.split(".") for part in keep_fp32):
            if parameter.dtype != torch.float32:
                raise RuntimeError(f"Required FP32 parameter loaded as {parameter.dtype}: {name}")
            preserved.append(name)
    if not preserved:
        raise RuntimeError("Wan Transformer FP32 preservation rules were not applied")
    vae = AutoencoderKLWan.from_pretrained(
        model_dir, subfolder="vae", torch_dtype=torch.float32,
        local_files_only=True, low_cpu_mem_usage=True,
    ).eval()
    vae.enable_tiling()
    pipeline_class = WanPipeline if config["mode"] == "text" else WanImageToVideoPipeline
    pipe = pipeline_class.from_pretrained(
        model_dir, transformer=transformer, vae=vae, text_encoder=None,
        torch_dtype=torch.float16, local_files_only=True,
    )
    if not pipe.config.expand_timesteps or pipe.transformer.config.in_channels != 48:
        raise RuntimeError("Expected the Wan 2.2 TI2V-5B configuration")
    if pipe.vae.dtype != torch.float32:
        raise RuntimeError("VAE must remain FP32 on this Turing setup")
    pipe.enable_model_cpu_offload(gpu_id=0)
    embeddings = load_file(str(out / "prompt-embeddings.safetensors"), device="cpu")
    for name in embeddings:
        finite(embeddings[name], name)
        embeddings[name] = embeddings[name].to(device="cuda", dtype=torch.float16)
        finite(embeddings[name], f"GPU FP16 {name}")
    step_records = []
    started = time.monotonic()

    def check_latents(_pipeline, step, timestep, callback_kwargs):
        latents = callback_kwargs["latents"]
        finite(latents, f"denoising step {step + 1} latents")
        record = {"step": step + 1, "timestep": float(timestep.item()),
                  "elapsed_seconds": time.monotonic() - started,
                  "latent_abs_max": float(latents.abs().max().item()),
                  "allocated_bytes": torch.cuda.memory_allocated(),
                  "reserved_bytes": torch.cuda.memory_reserved()}
        step_records.append(record)
        with (out / "denoising-steps.jsonl").open("a") as stream:
            stream.write(json.dumps(record) + "\n")
        print(json.dumps(record), flush=True)
        return callback_kwargs

    inputs = dict(
        **embeddings, height=config["height"], width=config["width"],
        num_frames=config["frames"], num_inference_steps=config["steps"],
        guidance_scale=config["guidance_scale"], max_sequence_length=512,
        generator=torch.Generator(device="cpu").manual_seed(config["seed"]),
        output_type="np", callback_on_step_end=check_latents,
        callback_on_step_end_tensor_inputs=["latents"],
    )
    if config["mode"] == "image":
        inputs["image"] = prepare_image(config, out)
    with torch.inference_mode():
        output = pipe(**inputs).frames[0]
    torch.cuda.synchronize()
    generation_seconds = time.monotonic() - started
    if len(step_records) != config["steps"]:
        raise RuntimeError(f"Only {len(step_records)} of {config['steps']} denoising steps completed")
    validation = validate_and_export(output, config, out)
    return {
        "generation_seconds": generation_seconds,
        "all_denoising_latents_finite": True,
        "denoising_steps_checked": len(step_records),
        "transformer_dtype": "float16_with_upstream_fp32_modules",
        "fp32_preserved_parameter_count": len(preserved),
        "vae_dtype": "float32", "model_cpu_offload": True, "vae_tiling": True,
        "attention_backend": "PyTorch_SDPA_automatic_dispatch",
        "validation": validation,
    }


def stage_worker(stage, out):
    started = time.monotonic()
    record = {"stage": stage, "started_utc": utc_now(), "status": "running"}
    config = json.loads((out / "config.json").read_text())
    # Internal stages cannot silently rerun into a previously used directory.
    with (out / f"{stage}.started").open("x") as stream:
        stream.write(f"{os.getpid()}\n")
    try:
        record["result"] = (encode_text if stage == "encode-text" else generate_video)(config, out)
        record["status"] = "completed"
    except Exception as exc:
        record.update(status="failed", error_type=type(exc).__name__, error=str(exc))
        raise
    finally:
        record.update(elapsed_seconds=time.monotonic() - started, finished_utc=utc_now())
        if "torch" in sys.modules:
            try:
                record["cuda"] = cuda_stats(sys.modules["torch"])
            except Exception as exc:
                record["cuda_telemetry_error"] = str(exc)
        write_json(out / f"{stage}-report.json", record)


def stop_group(process):
    if process.poll() is not None:
        return
    os.killpg(process.pid, signal.SIGTERM)
    try:
        process.wait(timeout=10)
    except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGKILL)
        process.wait(timeout=10)


def sampled_peak(path):
    peaks = {}
    if not path.exists():
        return peaks
    with path.open() as stream:
        for raw_row in csv.DictReader(stream):
            row = {k.strip(): v.strip() for k, v in raw_row.items() if k and v}
            key = next((k for k in row if k.startswith("memory.used")), None)
            try:
                value = float(row[key])
            except (KeyError, TypeError, ValueError):
                continue
            gpu = row.get("uuid", row.get("index", "unknown"))
            peaks[gpu] = max(peaks.get(gpu, 0), value)
    return peaks


def run_stage(stage, out, timeout):
    sampler = None
    process = None
    started = time.monotonic()
    sampling_error = None
    command = [sys.executable, "-u", str(Path(__file__).resolve()),
               "--stage", stage, "--output", str(out)]
    print(f"Starting {stage}; timeout={timeout}s; log={out / (stage + '.log')}", flush=True)
    try:
        with (out / f"{stage}-gpu.csv").open("w") as gpu_log:
            try:
                sampler = subprocess.Popen([
                    "nvidia-smi", "--query-gpu=index,uuid,timestamp,memory.used,utilization.gpu",
                    "--format=csv,nounits", "-l", "1",
                ], stdout=gpu_log, stderr=subprocess.DEVNULL)
            except OSError as exc:
                sampling_error = str(exc)
            with (out / f"{stage}.log").open("w") as log:
                process = subprocess.Popen(
                    command, stdout=log, stderr=subprocess.STDOUT, start_new_session=True,
                )
                try:
                    return_code = process.wait(timeout=timeout)
                except subprocess.TimeoutExpired:
                    stop_group(process)
                    raise TimeoutError(f"{stage} exceeded {timeout}s; see {stage}.log")
                if return_code:
                    raise RuntimeError(f"{stage} failed with exit {return_code}; see {stage}.log")
    finally:
        if process is not None:
            stop_group(process)
        if sampler is not None:
            sampler.terminate()
            try:
                sampler.wait(timeout=5)
            except subprocess.TimeoutExpired:
                sampler.kill()
                sampler.wait(timeout=5)
        write_json(out / f"{stage}-supervisor.json", {
            "stage": stage, "command": command, "timeout_seconds": timeout,
            "elapsed_seconds": time.monotonic() - started,
            "return_code": process.returncode if process else None,
            "nvidia_smi_sampled_peak_mib_by_gpu": sampled_peak(out / f"{stage}-gpu.csv"),
            "nvidia_smi_scope": "whole_GPU_including_other_processes_not_process_exclusive",
            "sampling_error": sampling_error,
        })


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("text", "image"))
    parser.add_argument("--prompt")
    parser.add_argument("--negative-prompt", default=NEGATIVE_PROMPT)
    parser.add_argument("--image", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--frames", type=int, default=33)
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--guidance-scale", type=float, default=5.0)
    parser.add_argument("--fps", type=int, default=24)
    parser.add_argument("--smoke", action="store_true", help="256x256, 17 frames, 2 steps; runtime only")
    parser.add_argument("--timeout", type=int, default=3600, help="Seconds per stage, maximum 3600")
    parser.add_argument("--cpu-threads", type=int, default=min(8, os.cpu_count() or 1))
    parser.add_argument("--stage", choices=("encode-text", "video"), help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.stage:
        return args
    if args.mode is None or not args.prompt or not args.prompt.strip():
        parser.error("--mode and a nonempty --prompt are required")
    if args.mode == "image" and (args.image is None or not args.image.is_file()):
        parser.error("--mode image requires an existing --image")
    if args.mode == "text" and args.image is not None:
        parser.error("--mode text does not consume --image; use --mode image")
    if args.smoke:
        args.height = args.width = 256
        args.frames, args.steps = 17, 2
    elif args.steps <= 2:
        # Manually selecting the smoke budget must not receive a stronger label.
        args.smoke = True
    if min(args.height, args.width) < 32 or args.height % 32 or args.width % 32:
        parser.error("height and width must be positive multiples of 32")
    if args.frames < 17 or args.frames % 4 != 1:
        parser.error("frames must be 4n+1 and at least 17")
    if args.steps < 1 or args.fps < 1 or args.cpu_threads < 1 or args.guidance_scale < 1:
        parser.error("steps, fps, cpu-threads and guidance-scale must be >= 1")
    if not 1 <= args.timeout <= 3600:
        parser.error("timeout must be between 1 and 3600 seconds per stage")
    return args


def main():
    args = parse_args()
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    out = args.output.resolve()
    if args.stage:
        stage_worker(args.stage, out)
        return
    model_dir = args.root.resolve() / "weights" / MODEL_NAME
    if not (model_dir / "model_index.json").is_file():
        raise FileNotFoundError(f"Local model is missing: {model_dir}")
    if list(model_dir.rglob("*.aria2")):
        raise RuntimeError("The model directory contains incomplete aria2 downloads")
    # Never overwrite or mix results from previous runs, even in an empty folder.
    out.mkdir(parents=True, exist_ok=False)
    config = {k: str(v.resolve()) if isinstance(v, Path) else v for k, v in vars(args).items()}
    config.pop("stage", None)
    config.update(
        created_utc=utc_now(), model_directory=str(model_dir),
        frontend_reference="Wan-AI/" + MODEL_NAME,
        image_sha256=sha256(args.image) if args.image else None,
        image_preprocessing="EXIF orientation, white alpha composite, aspect-preserving white pad",
        precision="T5_CPU_FP32_then_transformer_FP16_with_FP32_modules_and_VAE_FP32",
        scope="runtime_smoke_test_only" if args.smoke else "short_clip_application_test",
        author_exact_video_model_reproduction=False,
        official_default_resolution=False,
        # Controller separately verifies source and derived weight manifests.
        config_sha256={str(p.relative_to(model_dir)): sha256(p) for p in (
            model_dir / "model_index.json", model_dir / "transformer/config.json",
            model_dir / "vae/config.json", model_dir / "scheduler/scheduler_config.json",
        )},
    )
    write_json(out / "config.json", config)
    write_json(out / "environment.json", versions())
    started = time.monotonic()
    report = {"status": "running", "started_utc": utc_now(),
              "semantic_evaluation": "not_performed_requires_visual_review"}
    try:
        for stage in ("encode-text", "video"):
            report["current_stage"] = stage
            write_json(out / "report.json", report)
            run_stage(stage, out, args.timeout)
        report["status"] = "runtime_smoke_completed" if args.smoke else "technical_checks_passed"
        report["validation"] = json.loads((out / "validation.json").read_text())
        report["artifacts"] = {
            name: {"bytes": (out / name).stat().st_size, "sha256": sha256(out / name)}
            for name in ("generated.mp4", "preview.gif", "prompt-embeddings.safetensors")
        }
    except BaseException as exc:
        report.update(status="failed", error_type=type(exc).__name__, error=str(exc))
        raise
    finally:
        report.update(finished_utc=utc_now(), elapsed_seconds=time.monotonic() - started)
        write_json(out / "report.json", report)
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc()
        sys.exit(1)

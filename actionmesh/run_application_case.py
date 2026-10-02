#!/usr/bin/env python3
"""Run one explicitly selected stage of a predefined ActionMesh application case."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys


def verify_video_result(video_out):
    """Allow only a verified full frontend artifact into the predefined backend."""
    report = json.loads((video_out / 'report.json').read_text())
    config = json.loads((video_out / 'config.json').read_text())
    validation = report.get('validation', {})
    if report.get('status') != 'technical_checks_passed':
        raise ValueError('Frontend technical checks did not pass: ' + str(video_out))
    if config.get('smoke') is not False or config.get('scope') != 'short_clip_application_test':
        raise ValueError('Backend requires a full application video, not a smoke test')
    if validation.get('scope') != 'short_clip_application_test':
        raise ValueError('Frontend validation scope is missing or is only a smoke test')
    required_checks = ('frame_count_matches_request', 'at_least_16_frames',
                       'decoded_frames_finite', 'no_black_frames', 'not_pixel_identical_static')
    if any(validation.get('checks', {}).get(name) is not True for name in required_checks):
        raise ValueError('Frontend video validation checks are missing or failed')
    video = video_out / 'generated.mp4'
    artifact = report.get('artifacts', {}).get('generated.mp4', {})
    if not video.is_file() or video.stat().st_size != artifact.get('bytes'):
        raise ValueError('Generated video is missing or its size differs from the frontend report')
    digest = hashlib.sha256()
    with video.open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(block)
    if digest.hexdigest() != artifact.get('sha256'):
        raise ValueError('Generated video SHA256 differs from the frontend report')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--case", required=True,
                        choices=("text-octopus", "image-mushroom", "mesh-panda"))
    parser.add_argument("--stage", required=True, choices=("video", "backend", "export"))
    parser.add_argument("--smoke", action="store_true", help="Video runtime test only")
    parser.add_argument("--dry-run", action="store_true", help="Print argv without executing")
    args = parser.parse_args()
    if args.smoke and args.stage != "video":
        parser.error("--smoke is video-only; smoke output cannot be sent to backend/export")
    scripts = Path(__file__).resolve().parent
    config = json.loads((scripts / "application-cases.json").read_text())
    case = config["cases"][args.case]
    root = args.root.expanduser().resolve()
    runs = root / "outputs/applications-20261002"
    video_out = runs / (args.case + "-video" + ("-smoke" if args.smoke else ""))
    backend_out = runs / (args.case + "-4d")
    common = ["--root", str(root)]
    if args.stage == "video":
        command = [sys.executable, str(scripts / "generate_application_video.py"),
                   *common, "--mode", case["mode"], "--prompt", case["prompt"],
                   "--output", str(video_out)]
        for key, value in config.items():
            if key != "cases":
                command.extend(["--" + key.replace("_", "-"), str(value)])
        if "image" in case:
            command.extend(["--image", str(root / case["image"])])
        if args.smoke:
            command.append("--smoke")
    elif args.stage == "backend":
        # The backend selects 16 actual frames and always runs official low_ram.
        command = [sys.executable, str(scripts / "run_application_backend.py"),
                   *common, "--video", str(video_out / "generated.mp4"),
                   "--output", str(backend_out), "--seed", str(config["seed"])]
        if "mesh" in case:
            command.extend(["--mesh", str(root / case["mesh"])])
    else:
        command = [sys.executable, str(scripts / "export_application_result.py"),
                   *common, "--output", str(backend_out), "--publish",
                   str(root / "publication/applications-20261002" / args.case)]
    print(json.dumps({"case": args.case, "stage": args.stage, "argv": command},
                     ensure_ascii=False, indent=2), flush=True)
    if not args.dry_run:
        if args.stage == 'backend':
            verify_video_result(video_out)
        subprocess.run(command, check=True)


if __name__ == "__main__":
    main()

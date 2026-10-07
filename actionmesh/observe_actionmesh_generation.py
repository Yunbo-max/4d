"""Harness child entry for an explicitly requested native-context capture unit.

The parent complete-unit runner admits source, weights, dataset and device first.
This entry calls the pinned official run_actionmesh function; it implements no
generation algorithm or scorer. Never substitute its cost for historical pricing.
"""
import argparse
import json
from pathlib import Path
import runpy
import sys
import tarfile

from research_math.pipeline_decoder_capture import PipelineDecoderCapture


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('source-root', 'capture-root', 'identity', 'input', 'output_dir'):
        parser.add_argument('--'+name, type=Path, required=True)
    parser.add_argument('--seed', type=int, required=True)
    parser.add_argument('--dtype', choices=('float16', 'bfloat16'), required=True)
    parser.add_argument('--low_ram', action='store_true')
    parser.add_argument('--stage_0_steps', type=int, required=True)
    parser.add_argument('--stage_1_steps', type=int, required=True)
    parser.add_argument('--face_decimation', type=int, required=True)
    parser.add_argument('--floaters_threshold', type=float, required=True)
    parser.add_argument('--guidance_scales', nargs='+', type=float, required=True)
    parser.add_argument('--anchor_idx', type=int, required=True)
    args = parser.parse_args()
    source = args.source_root.resolve()
    identity = json.loads(args.identity.read_text())
    generation = identity['generation']
    from research_math.complete_unit_contract import validate_generation_profile
    profile = validate_generation_profile(generation)
    expected = (42, 'float16' if profile == 'fp16-lowram-v1' else 'bfloat16',
                profile == 'fp16-lowram-v1', 100, 30, 40000, .02, [7.5], 0)
    actual = (args.seed, args.dtype, args.low_ram, args.stage_0_steps,
              args.stage_1_steps, args.face_decimation, args.floaters_threshold,
              args.guidance_scales, args.anchor_idx)
    if actual != expected:
        raise ValueError('Observer must preserve every admitted native generation parameter')
    archive = args.capture_root.with_suffix('.tar.gz')
    if archive.exists() or args.capture_root.exists():
        raise FileExistsError('Preserve previous decoder capture')
    sys.path.insert(0, str(source/'third_party/TripoSG'))
    sys.path.insert(0, str(source))
    # Load the real entry without triggering its __main__ argument parser.
    official = runpy.run_path(str(source/'inference/video_to_animated_mesh.py'),
                             run_name='official_observer_entry')
    import torch
    pipeline = official['ActionMeshPipeline'](
        config_name=Path(generation['config']).name,
        config_dir=str(source/'actionmesh/configs'),
        dtype=getattr(torch, args.dtype), lazy_loading=args.low_ram)
    pipeline.to('cuda')
    with PipelineDecoderCapture(pipeline, args.capture_root, identity):
        official['run_actionmesh'](
            pipeline, input=str(args.input), output_dir=str(args.output_dir),
            seed=args.seed, stage_0_steps=args.stage_0_steps,
            stage_1_steps=args.stage_1_steps, face_decimation=args.face_decimation,
            floaters_threshold=args.floaters_threshold,
            guidance_scales=args.guidance_scales, anchor_idx=args.anchor_idx)
    # The receipt owns one closed archive plus a readable manifest. All windows
    # are inside it; no glob-selected subset and no tensor/mesh files are omitted.
    with tarfile.open(archive, mode='x:gz') as bundle:
        for path in sorted(args.capture_root.rglob('*')):
            if path.is_symlink():
                raise ValueError('Capture archive cannot contain symlinks')
            if path.is_file():
                bundle.add(path, arcname=path.relative_to(args.capture_root).as_posix(),
                           recursive=False)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())

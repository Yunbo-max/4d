"""New conditional motion readouts for completed direction arms; CPU only.

Reuses existing sampling, alignment and fixed maps. Official CD scalars are not
recomputed. No new model, ICP, learned parameter, or direction selection.
"""
import argparse
import json
import os
from pathlib import Path
import signal
import time

from research_census_eval import digest

os.environ['CUDA_VISIBLE_DEVICES'] = ''


def revalidate_sources(hashes):
    for filename, expected in hashes.items():
        if digest(Path(filename)) != expected:
            raise ValueError('Source changed during motion readout: '+filename)


def qualify_sources(args):
    """Bind current artifacts to the independent evaluator's frozen evidence.

    Use exact canonical paths from its provenance, never a basename/UID match.
    This intentionally refuses an unrelated attempt or a relocated archive.
    """
    from research_census_motion_controls import ALLOWED_UIDS
    hashes = {}

    def snapshot(path):
        path = path.resolve()
        actual = digest(path)
        if str(path) in hashes and hashes[str(path)] != actual:
            raise ValueError('Source changed during qualification: '+str(path))
        hashes[str(path)] = actual
        return path

    def read(path):
        return json.loads(snapshot(path).read_text())

    generation = read(args.case_dir/'report.json')
    direction = read(args.direction_dir/'report.json')
    evaluation = read(args.direction_eval_dir/'report.json')
    provenance = read(args.direction_eval_dir/'provenance.json')
    stage = read(args.stage_gt_dir/'report.json')
    uid = generation.get('uid')
    if uid not in ALLOWED_UIDS:
        raise ValueError('Require one of the two frozen direction diagnostic assets')
    for label, report in (('generation', generation), ('direction', direction),
                          ('evaluation', evaluation), ('stage-GT', stage)):
        if report.get('status') != 'completed':
            raise ValueError('Incomplete '+label+' report')
        if (report.get('uid'), report.get('seed')) != (uid, 42):
            raise ValueError('UID/seed mismatch in '+label+' report')
    if direction.get('controls_passed') is not True:
        raise ValueError('Direction controls are not qualified')
    if direction.get('GT_read') is not False or direction.get('training') is not False:
        raise ValueError('Direction generation must declare no GT access or training')
    if evaluation.get('independent_control_gates_passed') is not True or evaluation.get('all_arms_retained') is not True:
        raise ValueError('Independent evaluator did not retain and qualify all arms')
    if provenance.get('native_cache_qualified') is not True:
        raise ValueError('Independent evaluator did not qualify the native cache')
    matrix = args.stage_gt_dir/'shared-anchor-transform.npy'
    gt = args.gt_dir/uid/'surfaces.npy'
    required = [args.case_dir/'report.json', args.case_dir/'sequence.npz', gt, matrix,
                args.stage_gt_dir/'report.json', args.direction_dir/'report.json']
    required += [args.native_cache_dir/name for name in ('report.json', 'provenance.json',
                 'protocol.json', 'controls.json', 'per-frame-shape.json', 'sampling-and-maps.npz')]
    required += [args.direction_dir/'variants'/arm/name
                 for arm in ('forward', 'row_permutation', 'time_reversal')
                 for name in ('report.json', 'sequence.npz')]
    recorded = {}
    for filename, expected in provenance.get('source_files_sha256', {}).items():
        key = str(Path(filename).resolve())
        if key in recorded:
            raise ValueError('Ambiguous evaluation source path: '+key)
        recorded[key] = expected
    for path in required:
        current = snapshot(path)
        if hashes[str(current)] != recorded.get(str(current)):
            raise ValueError('Independent evaluation source hash mismatch: '+str(current))
    source_hashes = direction.get('source_hashes', {})
    if not all(name in source_hashes for name in ('prepared.npz', 'denoised.npz', 'sequence.npz', 'report.json')):
        raise ValueError('Missing direction parent source identity')
    for name, expected in source_hashes.items():
        path = snapshot(args.case_dir/name)
        if hashes[str(path)] != expected:
            raise ValueError('Parent direction source hash mismatch: '+name)
    if digest(args.case_dir/'sequence.npz') != generation.get('sha256', {}).get('sequence.npz'):
        raise ValueError('Generation sequence hash mismatch')
    if digest(matrix) != stage.get('shared_matrix_sha256') or digest(matrix) != provenance.get('same_matrix_sha256'):
        raise ValueError('Shared alignment provenance mismatch')
    arm_reports = {}
    for arm in ('forward', 'row_permutation', 'time_reversal'):
        report = read(args.direction_dir/'variants'/arm/'report.json')
        if report.get('status') != 'completed' or (report.get('uid'), report.get('seed')) != (uid, 42):
            raise ValueError('Incomplete or mismatched UID/seed in '+arm+' arm')
        if arm != 'time_reversal' and report.get('control_pass') is not True:
            raise ValueError('Unqualified '+arm+' control')
        arm_reports[arm] = report
    snapshot(args.native_cache_dir/'aligned-material-clouds.npz')
    for name in ('research_census_direction_motion.py', 'research_census_motion_readout.py',
                 'research_census_time_direction_eval.py', 'research_census_eval.py',
                 'research_census_motion_controls.py', 'research_census_stage_gt.py',
                 'research_census_time_direction.py', 'research_three_ideas.py'):
        snapshot(Path(__file__).with_name(name))
    revalidate_sources(hashes)
    return dict(generation=generation, direction=direction, evaluation=evaluation,
                arm_reports=arm_reports, source_sha256=hashes,
                evaluator_provenance_sha256=hashes[str((args.direction_eval_dir/'provenance.json').resolve())])


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for key in ('root','case-dir','direction-dir','direction-eval-dir','stage-gt-dir','native-cache-dir','gt-dir','output'):
        parser.add_argument('--'+key,type=Path,required=True)
    args=parser.parse_args()
    args.output.mkdir(parents=True,exist_ok=False)
    started=time.monotonic()
    def timeout(*unused):raise TimeoutError('180-second CPU motion readout budget exceeded')
    signal.signal(signal.SIGALRM,timeout);signal.alarm(180)
    qualified=qualify_sources(args)
    import numpy as np
    import torch
    import trimesh
    from research_census_eval import load_official,load_arrays,OFFICIAL_FILES
    from research_census_time_direction_eval import qualify_cache,validate_sequence,exact
    from research_census_motion_controls import array_hash
    from research_census_motion_readout import motion_readout,protocol
    torch.set_num_threads(1)
    generation=qualified['generation']
    matrix_path=args.stage_gt_dir/'shared-anchor-transform.npy'
    gt_path=args.gt_dir/generation['uid']/'surfaces.npy'
    _,cache_provenance,_,_,maps=qualify_cache(args,generation,gt_path,matrix_path)
    native,faces,gt=load_arrays(args.case_dir/'sequence.npz',gt_path)
    with np.load(args.case_dir/'sequence.npz',allow_pickle=False) as f:ids=f['query_vertex_ids'].copy()
    with np.load(args.native_cache_dir/'aligned-material-clouds.npz',allow_pickle=False) as f:native_cloud=f['native_aligned'].copy()
    assert array_hash(native_cloud)==cache_provenance['native_material_cloud_sha256']
    for name in OFFICIAL_FILES:
        path=(args.root/'repo'/'actionbench'/name).resolve()
        qualified['source_sha256'][str(path)]=digest(path)
    _,official=load_official(args.root/'repo',cpu_rng_fix=True)
    assert official['sha256']==cache_provenance['official']['sha256']
    import sample_mesh
    from pytorch3d.transforms import Transform3d
    transform=Transform3d(matrix=torch.from_numpy(maps['shared_alignment_matrix']))
    sample_faces=torch.from_numpy(maps['material_face_indices'])
    barycentric=torch.from_numpy(maps['material_barycentric'])
    qg=np.random.RandomState(45).permutation(len(gt[0]))[:10000]
    report=dict(status='running',uid=generation['uid'],seed=42,protocol=protocol(),arms={},
                source_sha256=qualified['source_sha256'],
                evaluator_provenance_sha256=qualified['evaluator_provenance_sha256'],
                independent_evaluation_sources_qualified=True,
                scope='Developmental auxiliary readout; exact first-frame NN remains a conditional correspondence, not semantic GT identity.',
                official_CD_recomputed=False,ICP_run=False,model_loaded=False,all16frames=True)
    for arm in ('row_permutation','time_reversal'):
        folder=args.direction_dir/'variants'/arm
        arm_report=qualified['arm_reports'][arm]
        vertices=validate_sequence(folder/'sequence.npz',arm_report,native,faces,ids,arm)
        meshes=[trimesh.Trimesh(frame,faces,process=False) for frame in vertices]
        samples=sample_mesh.apply_baryc_sampling_on_meshes(
            sample_mesh.join_meshes_as_batch([sample_mesh.trimesh_to_pytorch3d(m) for m in meshes]),sample_faces,barycentric)
        cloud=transform.transform_points(samples).numpy()
        assert exact(cloud[0],native_cloud[0]),'Changed first-frame sample identity'
        values=motion_readout(cloud,gt,maps['pred_to_gt_firstframe'],maps['gt_to_pred_firstframe'],maps['query_predicted_ids'],qg)
        out=args.output/arm;out.mkdir()
        (out/'motion-readouts.json').write_text(json.dumps(values,indent=2)+'\n')
        np.savez_compressed(out/'aligned-material-clouds.npz',aligned_material_cloud=cloud)
        report['arms'][arm]=dict(status='completed',summaries={k:v['primary_summary'] for k,v in values.items()},
                                cloud_array_sha256=array_hash(cloud),source_sequence_sha256=digest(folder/'sequence.npz'))
    revalidate_sources(report['source_sha256'])
    report.update(status='completed',seconds=time.monotonic()-started,cuda_initialized=torch.cuda.is_initialized())
    assert not report['cuda_initialized']
    (args.output/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    signal.alarm(0)
    print(json.dumps({'status':'completed','output':str(args.output),'seconds':report['seconds']}))


if __name__=='__main__':main()

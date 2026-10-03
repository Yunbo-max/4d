"""CPU descriptive error ranking from three frozen model-only material scores.

No inference/training/ICP. Model scores are saved in canonical coordinates before
GT evaluation. Cached frame0 NN assignments are fixed; target is conditional
forward displacement EPE at two times, NOT official CD-M or known material truth.
"""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import resource
import sys
import time
import traceback

import numpy as np

UIDS=('000-037_1358c424008a43cbaa35eba5e58551ac','000-043_061697e330d44524bd11f8cf95772e2d')
FRAMES=(8,15)
SCORES=('reversal_disagreement','native_motion_amplitude','stageI_sample_distance')
RECON_ATOL=2e-6
RECON_RTOL=2e-6
PROTOCOL_ID='census-disagreement-rank-20261002-v1'
PROTOCOL_SHA256='b0a6699b4b07ed767b055ebac536db6442b07ef565614534853b3091b2385595'


def digest(path):
    value=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(8*1024*1024),b''):value.update(block)
    return value.hexdigest()


def array_hash(value):return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()


def write_json(path,data):
    temporary=path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(data,indent=2,allow_nan=False)+'\n')
    temporary.replace(path)


def exact(a,b):return a.dtype==b.dtype and a.shape==b.shape and a.tobytes()==b.tobytes()


def finite_vector(value):
    value=np.asarray(value,dtype=np.float64)
    if value.ndim!=1 or not len(value) or not np.isfinite(value).all():raise ValueError('Expected nonempty finite vector')
    return value


def average_ranks(value):
    value=finite_vector(value);order=np.argsort(value,kind='stable')
    sorted_values=value[order];rank=np.empty(len(value),dtype=np.float64)
    starts=np.r_[0,np.flatnonzero(sorted_values[1:]!=sorted_values[:-1])+1]
    ends=np.r_[starts[1:],len(value)]
    for first,last in zip(starts,ends):rank[order[first:last]]=(first+1+last)/2.
    return rank


def spearman(score,outcome):
    score,outcome=finite_vector(score),finite_vector(outcome)
    if score.shape!=outcome.shape:raise ValueError('Rank lengths differ')
    a,b=average_ranks(score),average_ranks(outcome)
    a-=a.mean();b-=b.mean()
    denominator=np.linalg.norm(a)*np.linalg.norm(b)
    if denominator==0:return None
    return float(np.clip(np.dot(a,b)/denominator,-1.,1.))


def risk_diagnostic(score,outcome,query_ids):
    score,outcome=finite_vector(score),finite_vector(outcome)
    ids=np.asarray(query_ids)
    if score.shape!=outcome.shape or ids.shape!=score.shape or len(np.unique(ids))!=len(ids):
        raise ValueError('Scores/outcomes/unique material IDs must match')
    if not np.issubdtype(ids.dtype,np.integer) or (outcome<0).any():raise ValueError('Integer IDs and nonnegative EPE required')
    base=float(outcome.mean())
    result={'status':'defined','n_material_queries':len(score),'spearman':spearman(score,outcome),
            'overall_risk':base,'risk_enrichment':None,'retained_risk':None,'retained_risk_ratio':None,
            'top_fraction':.1,'score_unique_values':int(len(np.unique(score))),
            'tie_policy':'Exact ties; arbitrary ascending original material ID breaks selection ties, not a predictive signal'}
    if np.ptp(score)==0 or np.ptp(outcome)==0 or base==0:
        result.update(status='undefined',reason='constant_score' if np.ptp(score)==0 else 'constant_or_zero_outcome',
                      rejected_query_ids=[],boundary_tie_total=None,boundary_tie_rejected=None,boundary_tie_is_split=None)
        return result
    count=max(1,int(math.ceil(.1*len(score))))
    if count>=len(score):raise ValueError('Need a nonempty retained set')
    order=np.lexsort((ids,-score));reject,retain=order[:count],order[count:]
    boundary=score[reject[-1]];tied=score==boundary
    rejected_ties=int(tied[reject].sum());all_ties=int(tied.sum())
    high=float(outcome[reject].mean());remaining=float(outcome[retain].mean())
    result.update(rejected_count=count,retained_count=len(retain),rejected_query_ids=ids[reject].tolist(),
        rejected_risk=high,risk_enrichment=high/base,retained_risk=remaining,retained_risk_ratio=remaining/base,
        boundary_score=float(boundary),boundary_tie_total=all_ties,boundary_tie_rejected=rejected_ties,
        boundary_tie_is_split=0<rejected_ties<all_ties,
        rejected_epe_mass_fraction=float(outcome[reject].sum()/outcome.sum()))
    return result


def material_samples(vertices,faces,face_ids,barycentric):
    """Same fixed face/barycentric samples; explicit float32 eager arithmetic."""
    vertices=np.asarray(vertices,dtype=np.float32);faces=np.asarray(faces);face_ids=np.asarray(face_ids)
    barycentric=np.asarray(barycentric,dtype=np.float32)
    if vertices.ndim!=3 or vertices.shape[-1]!=3 or faces.ndim!=2 or faces.shape[1]!=3:
        raise ValueError('Bad mesh shape')
    if face_ids.ndim!=1 or barycentric.shape!=(len(face_ids),3) or not len(face_ids):raise ValueError('Bad fixed sample shape')
    if not np.isfinite(vertices).all() or not np.isfinite(barycentric).all():raise ValueError('Nonfinite sampling input')
    if face_ids.min()<0 or face_ids.max()>=len(faces) or faces.min()<0 or faces.max()>=vertices.shape[1]:
        raise ValueError('Face/sample indices out of range')
    if barycentric.min() < -1e-7 or not np.allclose(barycentric.sum(axis=-1),1,atol=1e-6,rtol=0):
        raise ValueError('Invalid barycentric weights')
    triangles=vertices[:,faces[face_ids]]
    return (triangles[:,:,0]*barycentric[None,:,0,None]+triangles[:,:,1]*barycentric[None,:,1,None]
            +triangles[:,:,2]*barycentric[None,:,2,None])


def model_only_scores(native_raw,reverse_raw,stageI_surfaces,check):
    """Only canonical model predictions enter. No GT/matrix/map arguments."""
    from scipy.spatial import cKDTree
    if native_raw.shape!=reverse_raw.shape or native_raw.shape[0]!=3:raise ValueError('Require anchor/8/15 sample arrays')
    native,reverse=native_raw.astype(np.float64),reverse_raw.astype(np.float64)
    per_frame={'reversal_disagreement':np.linalg.norm(reverse[1:]-native[1:],axis=-1),
               'native_motion_amplitude':np.linalg.norm(native[1:]-native[0],axis=-1)}
    distances=[]
    for index,frame in enumerate(FRAMES):
        check(f'model_only_stageI_sample_distance_frame{frame}')
        surface=np.asarray(stageI_surfaces[frame],dtype=np.float64)
        if surface.shape!=(50000,3) or not np.isfinite(surface).all():raise ValueError('Require cached50k Stage-I surface')
        distances.append(cKDTree(surface).query(native[index+1],k=1)[0])
    per_frame['stageI_sample_distance']=np.stack(distances)
    scores={name:value.mean(axis=0) for name,value in per_frame.items()}
    if any(not np.isfinite(value).all() for value in scores.values()):raise ValueError('Nonfinite model score')
    return scores,per_frame


def conditional_target(native_aligned,gt_frames,locked_nn):
    """Prediction rows and GT rows contain ONLY physical frames0/8/15."""
    pred=np.asarray(native_aligned,dtype=np.float64);gt=np.asarray(gt_frames,dtype=np.float64)
    nn=np.asarray(locked_nn)
    if pred.ndim!=3 or pred.shape[0]!=3 or pred.shape[-1]!=3 or gt.ndim!=3 or gt.shape[0]!=3 or gt.shape[-1]!=3:
        raise ValueError('Target requires anchor plus two selected times')
    if nn.shape!=(pred.shape[1],) or not np.issubdtype(nn.dtype,np.integer) or nn.min()<0 or nn.max()>=gt.shape[1]:
        raise ValueError('Invalid cached frame0 matching map')
    matched=gt[:,nn]
    per_frame=np.linalg.norm((pred[1:]-pred[0])-(matched[1:]-matched[0]),axis=-1)
    oracle=np.linalg.norm(pred[0]-matched[0],axis=-1)
    if not np.isfinite(per_frame).all() or not np.isfinite(oracle).all():raise ValueError('Nonfinite evaluation target')
    return per_frame.mean(axis=0),per_frame,oracle


def source_hash(provenance,filename):
    matches=[value for path,value in provenance.get('source_files_sha256',{}).items() if Path(path).name==filename]
    if len(matches)!=1:raise ValueError('Expected exactly one source hash for '+filename)
    return matches[0]


def read_report(path,uid=None):
    result=json.loads(path.read_text())
    if result.get('status')!='completed':raise ValueError('Incomplete parent: '+str(path))
    if uid is not None and (result.get('uid'),result.get('seed'))!=(uid,42):raise ValueError('UID/seed mismatch: '+str(path))
    return result


def run(args,state,check):
    from research_census_time_direction_eval import validate_sequence,difference
    from research_census_time_direction import array_digest
    from research_census_stage_gt import transform_numpy
    sources={}
    def record(path):
        value=digest(path);sources[str(path)]=value;return value
    check('model_only_source_validation')
    generation=read_report(args.case_dir/'report.json');uid=generation.get('uid')
    if uid not in UIDS or generation.get('seed')!=42:raise ValueError('Only firsttwo frozen seed42 cases permitted')
    state.update(uid=uid,seed=42,independent_assets_in_report=1,declared_independent_cohort_assets=2)
    direction=read_report(args.direction_dir/'report.json',uid)
    if not direction.get('controls_passed') or direction.get('GT_read') is not False or direction.get('training') is not False:
        raise ValueError('Direction parent lacks required controls/no-GT/no-training declarations')
    for name in ('prepared.npz','denoised.npz','sequence.npz','report.json'):
        if record(args.case_dir/name)!=direction.get('source_hashes',{}).get(name):raise ValueError('Direction native source mismatch: '+name)
    if record(args.case_dir/'sequence.npz')!=generation.get('sha256',{}).get('sequence.npz'):raise ValueError('Native sequence hash mismatch')
    with np.load(args.case_dir/'sequence.npz',allow_pickle=False) as saved:
        native,faces,vertex_ids=(saved[key].copy() for key in ('vertices','faces','query_vertex_ids'))
        if not np.array_equal(saved['frame_indices'],np.arange(16)):raise ValueError('Bad physical timeline')
    if native.ndim!=3 or native.shape[0]!=16 or native.shape[2]!=3 or not np.isfinite(native).all():raise ValueError('Bad native geometry')
    if not np.array_equal(vertex_ids,np.arange(native.shape[1])) or array_digest(vertex_ids)!=direction.get('query_vertex_ids_sha256'):
        raise ValueError('Source query identity mismatch')
    diagonal=float(np.linalg.norm(np.ptp(native[0].astype(np.float64),axis=0)))
    if diagonal<=0 or not np.isfinite(diagonal):raise ValueError('Bad anchor size')
    reverse=None
    for arm in ('forward','row_permutation','time_reversal'):
        folder=args.direction_dir/'variants'/arm;report=read_report(folder/'report.json',uid)
        value=validate_sequence(folder/'sequence.npz',report,native,faces,vertex_ids,arm)
        if arm=='forward' and (not exact(value,native) or not report.get('control_pass')):raise ValueError('Forward reproduction failed')
        if arm=='row_permutation':
            diff=difference(value,native,diagonal)
            if diff['moving_rms_xyz_over_D']>1e-4 or diff['moving_max_abs_coordinate_over_D']>1e-3 or not report.get('control_pass'):
                raise ValueError('Permutation control failed')
        if arm=='time_reversal':reverse=value
        record(folder/'report.json');record(folder/'sequence.npz')
    record(args.direction_dir/'report.json')
    cache=read_report(args.native_cache_dir/'report.json',uid)
    if cache.get('positive_control_status')!='invariant':raise ValueError('Native material cache unqualified')
    provenance=json.loads((args.native_cache_dir/'provenance.json').read_text())
    if record(args.case_dir/'sequence.npz')!=source_hash(provenance,'sequence.npz'):raise ValueError('Material cache source mismatch')
    maps_path=args.native_cache_dir/'sampling-and-maps.npz'
    if record(maps_path)!=provenance.get('sampling_and_maps_sha256'):raise ValueError('Cached sampling map hash mismatch')
    # Read only inference-available sampling identity here; GT fields remain unopened.
    with np.load(maps_path,allow_pickle=False) as saved:
        query_ids=saved['query_predicted_ids'].copy();sample_faces=saved['material_face_indices'].copy();bary=saved['material_barycentric'].copy()
    if not np.array_equal(query_ids,np.random.RandomState(44).permutation(100000)[:10000]):raise ValueError('Frozen10k seed44 query IDs differ')
    if sample_faces.shape!=(1,100000) or bary.shape!=(1,100000,3):raise ValueError('Bad material-sampling cache')
    selected_faces=sample_faces[0,query_ids];selected_bary=bary[0,query_ids]
    physical=(0,*FRAMES)
    native_raw=material_samples(native[list(physical)],faces,selected_faces,selected_bary)
    reverse_raw=material_samples(reverse[list(physical)],faces,selected_faces,selected_bary)
    if not exact(native_raw[0],reverse_raw[0]):raise ValueError('Sampled anchor mismatch')
    probe=read_report(args.probe_dir/'report.json',uid)
    probe_provenance=json.loads((args.probe_dir/'provenance.json').read_text())
    for name in ('sequence.npz','denoised.npz','report.json'):
        if record(args.case_dir/name)!=probe_provenance.get('source_hashes',{}).get(name):raise ValueError('Probe/native source mismatch')
    surfaces={};raw_mesh_hashes={}
    for frame in FRAMES:
        folder=args.probe_dir/f'frame_{frame:02d}';measurement=json.loads((folder/'measurement.json').read_text())
        path=folder/'surface-samples.npz'
        if measurement.get('frame')!=frame or measurement.get('samples_per_mesh')!=50000 or measurement.get('sample_seed')!=20261002+frame:
            raise ValueError('Stage-I cached sampling frame/count/seed differs')
        if record(path)!=measurement.get('sha256',{}).get(path.name):raise ValueError('Stage-I sample hash differs')
        with np.load(path,allow_pickle=False) as saved:
            surfaces[frame]=saved['stageI_xyz'].copy()
            if int(saved['seed'])!=20261002+frame:raise ValueError('Stage-I sample seed differs')
        raw=folder/'stageI-raw-mesh.npz';raw_mesh_hashes[frame]=record(raw)
        if raw_mesh_hashes[frame]!=measurement.get('sha256',{}).get(raw.name):raise ValueError('Stage-I mesh lineage differs')
        record(folder/'measurement.json')
    scores,per_scores=model_only_scores(native_raw,reverse_raw,surfaces,check)
    np.savez_compressed(args.output/'model-only-scores.npz',query_material_ids=query_ids,
        material_face_indices=selected_faces,material_barycentric=selected_bary,physical_frames=np.asarray(physical),
        native_raw=native_raw,reverse_raw=reverse_raw,**scores,
        **{name+'_by_frame':value for name,value in per_scores.items()})
    state['model_only_scores_sha256']=digest(args.output/'model-only-scores.npz')
    state['model_only_scores_saved_before_GT_arrays_or_alignment_load']=True
    check('evaluation_only_cached_GT_and_alignment')
    stage_gt=read_report(args.stage_gt_dir/'report.json',uid)
    stage_provenance=json.loads((args.stage_gt_dir/'provenance.json').read_text())
    matrix_path=args.stage_gt_dir/'shared-anchor-transform.npy';matrix_hash=record(matrix_path)
    if matrix_hash!=stage_gt.get('shared_matrix_sha256') or matrix_hash!=stage_provenance.get('shared_matrix_sha256') or matrix_hash!=source_hash(provenance,matrix_path.name):
        raise ValueError('Exact cached shared transform lineage mismatch')
    gt_path=args.gt_dir/uid/'surfaces.npy';gt_hash=record(gt_path)
    if gt_hash!=source_hash(provenance,'surfaces.npy') or gt_hash!=source_hash(stage_provenance,'surfaces.npy'):
        raise ValueError('Locked GT source differs')
    if source_hash(stage_provenance,'sequence.npz')!=record(args.case_dir/'sequence.npz'):raise ValueError('Stage-GT native source differs')
    for frame,expected in raw_mesh_hashes.items():
        values=[v for p,v in stage_provenance.get('source_files_sha256',{}).items()
                if Path(p).name=='stageI-raw-mesh.npz' and Path(p).parent.name==f'frame_{frame:02d}']
        if values!=[expected]:raise ValueError('Stage-I target differs from shared Stage-GT lineage')
    matrix=np.load(matrix_path,allow_pickle=False)
    with np.load(maps_path,allow_pickle=False) as saved:
        if not exact(saved['shared_alignment_matrix'],matrix):raise ValueError('Locked transform bytes differ')
        nn_all=saved['pred_to_gt_firstframe'].copy()
    if nn_all.shape!=(100000,):raise ValueError('Bad locked anchor matching array')
    cloud_path=args.native_cache_dir/'aligned-material-clouds.npz'
    with np.load(cloud_path,allow_pickle=False) as saved:aligned_all=saved['native_aligned'].copy()
    if aligned_all.shape!=(16,100000,3) or array_hash(aligned_all)!=provenance.get('native_material_cloud_sha256'):
        raise ValueError('Hash-qualified native aligned cloud required')
    aligned=aligned_all[list(physical)][:,query_ids];del aligned_all
    reconstructed=transform_numpy(native_raw,matrix)
    difference_max=float(np.max(np.abs(reconstructed-aligned)))
    if not np.allclose(reconstructed,aligned,rtol=RECON_RTOL,atol=RECON_ATOL):
        raise ValueError('Canonical material reconstruction fails frozen float32 cache tolerance')
    state['reconstruction_cache_check']={'atol':RECON_ATOL,'rtol':RECON_RTOL,'maximum_absolute_difference':difference_max,'passed':True}
    gt=np.load(gt_path,mmap_mode='r',allow_pickle=False)
    if gt.ndim!=3 or gt.shape[:2]!=(16,100000) or gt.shape[2]<3:raise ValueError('Bad GT trajectory array')
    # Match research_census_eval.load_arrays and the locked native-cache GT
    # representation before promoting arithmetic to float64 in the target.
    gt_selected=np.asarray(gt[list(physical),:,:3],dtype=np.float32);nn=nn_all[query_ids]
    outcome,per_outcome,oracle=conditional_target(aligned,gt_selected,nn)
    check('descriptive_within_asset_ranks')
    diagnostics={name:risk_diagnostic(scores[name],outcome,query_ids) for name in SCORES}
    oracle_report={'label':'ORACLE_GT_anchor_NN_distance_NOT_available_at_inference',
        'diagnostic':risk_diagnostic(oracle,outcome,query_ids),
        'spearman_with_model_only_scores':{name:spearman(scores[name],oracle) for name in SCORES},
        'interpretation':'Anchor-matching confound analysis only; not a usable score or conditional-independence test'}
    comparisons={}
    for name in SCORES[1:]:
        a,b=diagnostics[SCORES[0]],diagnostics[name]
        defined=a['status']=='defined' and b['status']=='defined'
        comparisons[name]={'defined':defined,'spearman_delta':a['spearman']-b['spearman'] if defined else None,
            'risk_enrichment_delta':a['risk_enrichment']-b['risk_enrichment'] if defined else None,
            'retained_risk_ratio_delta':a['retained_risk_ratio']-b['retained_risk_ratio'] if defined else None,
            'descriptive_beats_baseline':bool(defined and a['spearman']>b['spearman'] and a['risk_enrichment']>b['risk_enrichment'])}
    np.savez_compressed(args.output/'evaluation-only-outcomes.npz',query_material_ids=query_ids,
        locked_gt_anchor_ids=nn,conditional_displacement_epe=outcome,conditional_displacement_epe_by_frame=per_outcome,
        ORACLE_anchor_gt_NN_distance=oracle)
    for folder in (args.native_cache_dir,args.probe_dir,args.stage_gt_dir):
        for filename in ('report.json','provenance.json'):record(folder/filename)
    record(cloud_path)
    record(args.native_cache_dir/'protocol.json')
    for helper in ('research_census_time_direction_eval.py','research_census_time_direction.py','research_census_stage_gt.py'):
        record(Path(__file__).with_name(helper))
    for path,expected in sources.items():
        check('final_source_hash_recheck')
        if digest(path)!=expected:raise ValueError('Source changed during diagnostic: '+path)
    state.update(status='completed',model_only_diagnostics=diagnostics,oracle_confound_only=oracle_report,
        comparisons=comparisons,descriptive_screen_pass=all(v['descriptive_beats_baseline'] for v in comparisons.values()),
        frames=list(FRAMES),query_count=len(query_ids),n_independent_assets=1,
        target_mean_conditional_displacement_epe=float(outcome.mean()),
        mean_epe_per_frame={str(t):float(per_outcome[i].mean()) for i,t in enumerate(FRAMES)},
        GT_used_only_for_evaluation=True,GPU_used=False,training=False,ICP_run=False,
        scientific_scope='Descriptive one-asset conditional forward EPE rank diagnostic, no pointwise p-values, no calibrated uncertainty or novel-method claim',
        output_sha256={name:digest(args.output/name) for name in ('model-only-scores.npz','evaluation-only-outcomes.npz')})
    write_json(args.output/'provenance.json',{'source_files_sha256':sources,'script_sha256':digest(Path(__file__)),
        'protocol_sha256':state['protocol_sha256'],'exact_saved_transform_sha256':matrix_hash,
        'model_score_coordinates':'canonical, no GT-derived transform or scale',
        'material_reconstruction':'NumPy float32 eager barycentric operations, fixed cached identities',
        'numpy_version':np.__version__})


def aggregate(paths,state):
    rows=[]
    for path in paths:
        row=read_report(path)
        if row.get('uid') not in UIDS or row.get('seed')!=42 or row.get('protocol_sha256')!=state['protocol_sha256']:
            raise ValueError('Aggregation parent UID/seed/protocol mismatch')
        if (row.get('script_sha256')!=state['script_sha256'] or row.get('frames')!=list(FRAMES)
                or row.get('query_count')!=10000 or row.get('n_independent_assets')!=1
                or not row.get('GT_used_only_for_evaluation')
                or not row.get('model_only_scores_saved_before_GT_arrays_or_alignment_load')):
            raise ValueError('Aggregation parent implementation/scope mismatch')
        rows.append(row)
    if len(rows)!=2 or {row['uid'] for row in rows}!=set(UIDS):raise ValueError('Require exactly the two distinct frozen assets')
    state.update(status='completed',n_independent_assets=2,pointwise_independence_assumed=False,
        both_asset_descriptive_screen_pass=all(row['descriptive_screen_pass'] for row in rows),
        asset_results=[{'uid':row['uid'],'diagnostics':row['model_only_diagnostics'],'comparisons':row['comparisons'],
                       'oracle_confound_only':row['oracle_confound_only'],
                       'descriptive_screen_pass':row['descriptive_screen_pass']} for row in rows],
        source_reports_sha256={str(path):digest(path) for path in paths},
        interpretation='Exactly two assets; no pooled point-level significance, no population inference, no novelty clearance')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('root','case-dir','direction-dir','probe-dir','native-cache-dir','stage-gt-dir','gt-dir','protocol','output','extra-python'):
        parser.add_argument('--'+name,type=Path)
    parser.add_argument('--aggregate-reports',type=Path,nargs=2)
    parser.add_argument('--max-seconds',type=float,default=240.)
    args=parser.parse_args()
    if args.extra_python:sys.path.insert(0,str(args.extra_python.expanduser().resolve()))
    if args.output is None or args.protocol is None:parser.error('--output and --protocol required')
    if not 0<args.max_seconds<=240:parser.error('Frozen CPU budget is positive and <=240seconds')
    if not args.aggregate_reports and any(getattr(args,name) is None for name in ('case_dir','direction_dir','probe_dir','native_cache_dir','stage_gt_dir','gt_dir')):
        parser.error('All source directory arguments required')
    for name,value in vars(args).items():
        if isinstance(value,Path):setattr(args,name,value.expanduser().resolve())
    protocol=json.loads(args.protocol.read_text())
    if digest(args.protocol)!=PROTOCOL_SHA256:parser.error('Frozen protocol bytes changed')
    if protocol.get('protocol_id')!=PROTOCOL_ID or protocol.get('frames')!=list(FRAMES) or tuple(protocol.get('cohort_uids',[]))!=UIDS:
        parser.error('Wrong frozen protocol')
    if protocol.get('reconstruction_verification')!={'relative_tolerance':RECON_RTOL,'absolute_tolerance':RECON_ATOL,'reason':'NumPy float32 barycentric and cached affine arithmetic versus original PyTorch float32 cache; verification only, never score/GT tuning'}:
        parser.error('Frozen reconstruction tolerance differs')
    args.output.mkdir(parents=True,exist_ok=False)
    started=time.monotonic()
    state={'status':'running','started_utc':datetime.now(timezone.utc).isoformat(),'protocol_id':PROTOCOL_ID,
           'protocol_sha256':digest(args.protocol),'script_sha256':digest(Path(__file__))}
    write_json(args.output/'protocol.json',protocol)
    write_json(args.output/'command.json',{'argv':sys.argv,'max_seconds':args.max_seconds,'script_sha256':state['script_sha256']})
    def check(stage):
        state.update(stage=stage,elapsed_seconds=time.monotonic()-started)
        write_json(args.output/'progress.json',state)
        if state['elapsed_seconds']>args.max_seconds:raise TimeoutError('CPU ranking deadline exceeded at '+stage)
    try:
        if args.aggregate_reports:aggregate(args.aggregate_reports,state)
        else:run(args,state,check)
        check('completed')
    except Exception as exc:
        state.update(status='failed',error_type=type(exc).__name__,error=str(exc),traceback=traceback.format_exc())
    finally:
        usage=resource.getrusage(resource.RUSAGE_SELF)
        state.update(elapsed_seconds=time.monotonic()-started,resources={'cpu_user_seconds':usage.ru_utime,
            'cpu_system_seconds':usage.ru_stime,'peak_rss_bytes':int(usage.ru_maxrss if sys.platform=='darwin' else usage.ru_maxrss*1024),'GPU_used':False})
        write_json(args.output/'report.json',state)
    return int(state['status']!='completed')


if __name__=='__main__':raise SystemExit(main())

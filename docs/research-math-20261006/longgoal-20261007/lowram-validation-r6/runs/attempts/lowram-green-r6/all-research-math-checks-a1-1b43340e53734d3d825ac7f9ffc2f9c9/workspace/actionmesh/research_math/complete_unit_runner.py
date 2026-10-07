"""Harness-only complete default-release generation/export/three-arm scorer.

No fallback or automatic retry. A failure retains logs and partial artifacts.
This engineering unit is not a candidate experiment or full-128 qualification.
"""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

from official_actionbench_adapter import digest, write_json
from research_math.complete_unit_contract import generation_argv, require_three_scores, validate_generation_profile
from research_math.complete_unit_export import restore_vertices, validate_sequence


def run_stage(name, command, cwd, output, timeout, environment=None):
    started=time.monotonic()
    record={'command':command,'cwd':str(cwd),'exit_code':None,'status':'failed',
            'timeout_seconds':timeout}
    try:
        with (output/(name+'.stdout.log')).open('w') as out, (output/(name+'.stderr.log')).open('w') as err:
            import psutil
            process=subprocess.Popen(command,cwd=cwd,env=environment,stdout=out,stderr=err)
            try:
                returncode=process.wait(timeout=timeout)
            except subprocess.TimeoutExpired:
                # Enumerate while the parent still exists; subprocess.run would kill
                # it first and leave its scoring descendants orphaned.
                try:children=psutil.Process(process.pid).children(recursive=True)
                except psutil.NoSuchProcess:children=[]
                record['timeout_descendants']=[child.pid for child in children]
                for child in reversed(children):
                    try:child.kill()
                    except psutil.NoSuchProcess:pass
                process.kill();process.wait()
                psutil.wait_procs(children,timeout=1)
                raise
        record['exit_code']=returncode
        if returncode:raise RuntimeError(name+' exited '+str(returncode))
        record['status']='completed'
        return record
    except subprocess.TimeoutExpired as error:
        record['status']='timeout'
        raise RuntimeError(name+' exceeded its retained timeout') from error
    finally:
        record['elapsed_seconds']=time.monotonic()-started
        write_json(output/(name+'.execution.json'),record)


def collect_native_sequence(output,uid):
    import numpy as np
    import trimesh
    vertices=restore_vertices(np.load(output/'deformations_vertices.npy',allow_pickle=False))
    faces=np.load(output/'deformations_faces.npy',allow_pickle=False)
    arrays=validate_sequence(vertices,faces)
    expected={f'mesh_{i:02d}.glb' for i in range(16)}
    if {p.name for p in output.glob('mesh_*.glb')} != expected:
        raise ValueError('Exactly the 16 official GLB frames required')
    for i in range(16):
        mesh=trimesh.load(output/f'mesh_{i:02d}.glb',force='mesh',process=False)
        if (not np.array_equal(np.asarray(mesh.vertices,dtype=np.float32),vertices[i])
                or not np.array_equal(np.asarray(mesh.faces),faces)):
            raise ValueError('Official GLB/deformation disagreement at frame '+str(i))
    sequence=output/'sequence.npz'
    if sequence.exists():raise FileExistsError('Preserve previously converted sequence')
    np.savez_compressed(sequence,**arrays)
    report={'status':'completed','uid':uid,'seed':42,'frames':16,
            'coordinate_conversion':'inverse official (-z,x,y) export; exact GLB equality',
            'sha256':{p.name:digest(p) for p in output.iterdir() if p.is_file()},
            'scientific_effect_qualification':False}
    write_json(output/'report.json',report)
    return report


def verify_prerequisites(args,output):
    from research_math.actionbench_unit_manifest import freeze
    from research_math.snapshot_admission import snapshot_manifest, verify_source
    load=lambda p:json.loads(p.read_text())
    contract,population,snapshot,semantics,manifest=map(load,(args.contract,args.population,
        args.snapshot_admission,args.dataset_semantics,args.unit_manifest))
    if manifest.get('status')!='frozen_engineering_current_release_unit':
        raise ValueError('Frozen unit manifest required')
    if manifest['prerequisite_receipts']!={'snapshot_admission_sha256':digest(args.snapshot_admission),
            'dataset_semantics_sha256':digest(args.dataset_semantics)}:
        raise ValueError('Prerequisite receipt hashes changed')
    checked=freeze(contract,population,snapshot,semantics,args.source_root,args.dataset_root,
        output/'revalidated-unit-manifest.json',digest(args.snapshot_admission),digest(args.dataset_semantics))
    if checked!=manifest:raise ValueError('Frozen unit inputs or configuration changed')
    verify_source(args.source_root,load(args.snapshot_contract)['source'])
    generation=manifest['generation']
    validate_generation_profile(generation)
    model_dirs={'actionmesh':'ActionMesh','triposg':'TripoSG','dinov2':'dinov2','rmbg':'RMBG'}
    for key,name in model_dirs.items():
        root=args.weights_root/name
        expected=snapshot['snapshots'][key]
        actual=snapshot_manifest(root,expected)
        if actual!=expected:raise ValueError('Model bytes or revision changed: '+key)
        if (args.source_root/'pretrained_weights'/name).resolve()!=root.resolve():
            raise ValueError('Official model cache path differs from verified root: '+key)
    if os.environ.get('CUDA_VISIBLE_DEVICES')!=args.gpu_uuid:
        raise ValueError('Harness must allocate the exact physical GPU UUID')
    return manifest


def execute(args):
    from research_math.control_scoring import DeviceSamples
    from research_math.simple_mesh_controls import export_controls
    from research_math.unit_resources import HostSamples
    output=args.output.resolve();output.mkdir(parents=True,exist_ok=False)
    started=time.monotonic();monitor=DeviceSamples(args.gpu_uuid,output/'device-samples.jsonl')
    host=HostSamples(output,output/'host-samples.jsonl')
    result={'status':'failed','scientific_effect_qualification':False,'candidate_methods_tested':False,
            'source_root':str(args.source_root),'gpu_uuid':args.gpu_uuid,'stages':{}}
    try:
        host.start()
        manifest=verify_prerequisites(args,output)
        monitor.start()
        uid=manifest['calibration_unit']['uid']
        environment=os.environ.copy()
        environment.update(HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',HF_HUB_DISABLE_IMPLICIT_TOKEN='1',
            PYTHONPATH=str(args.source_root)+os.pathsep+str(args.source_root/'third_party/TripoSG'))
        result['offline_environment']={k:environment[k] for k in ('HF_HUB_OFFLINE','TRANSFORMERS_OFFLINE','PYTHONPATH')}
        native=output/'native-generation'
        profile=validate_generation_profile(manifest['generation'])
        result['runtime_profile']=profile
        command=generation_argv(args.source_root,args.dataset_root/'data'/uid/'imgs',native,profile=profile)
        result['stages']['generation']=run_stage('generation',command,args.source_root,output,
                                                max(1,args.wall_seconds-(time.monotonic()-started)),environment)
        begin=time.monotonic();collect_native_sequence(native,uid)
        result['stages']['export']={'elapsed_seconds':time.monotonic()-begin,'status':'completed'}
        begin=time.monotonic();controls=output/'controls'
        summary=export_controls(native,controls,uid=uid,expected_sequence_sha256=digest(native/'sequence.npz'),sigma=1.)
        if summary['status']!='completed':raise ValueError('At least one control failed')
        result['stages']['controls']={'elapsed_seconds':time.monotonic()-begin,'status':'completed'}
        remaining=args.wall_seconds-(time.monotonic()-started)
        if remaining<=1:raise TimeoutError('Complete-unit budget exhausted before scoring')
        score=output/'official-scores.json'
        adapter=Path(__file__).resolve().parents[1]/'official_actionbench_adapter.py'
        command=[sys.executable,str(adapter),'--case-dir',str(controls),'--gt-dir',str(args.dataset_root/'data'),
            '--output',str(score),'--manifest',str(controls/'manifest.json'),'--repo-root',str(args.source_root),
            '--device','cuda:0','--seed','44','--cpu-knn-backward','--timeout-seconds',str(max(1,int(remaining/3)))]
        result['stages']['official_scoring']=run_stage('official-scoring',command,adapter.parent,output,remaining)
        result['scores']=require_three_scores(json.loads(score.read_text()),uid)
        # Detect mutations across inference and scoring without replacing the original input receipt.
        final_dir=output/'final-integrity';final_dir.mkdir()
        verify_prerequisites(args,final_dir)
        result['status']='completed'
    except Exception as error:
        result.update(error=type(error).__name__+': '+str(error),traceback=traceback.format_exc())
    finally:
        monitor.close()
        host.close()
        result['host_resources']=host.summary()
        result['device_memory']={'sample_interval_seconds':1,'samples':len(monitor.memory),
            'observed_peak_mib':max(monitor.memory,default=None),'exact_peak':False,'errors':monitor.errors}
        result['outputs']=[{'path':str(p.relative_to(output)),'bytes':p.stat().st_size,'sha256':digest(p)}
                           for p in sorted(output.rglob('*')) if p.is_file()]
        if monitor.errors or host.errors:result['status']='failed'
        result['elapsed_seconds']=time.monotonic()-started
        if result['elapsed_seconds']>args.wall_seconds:result.update(status='failed',budget_exceeded=True)
        write_json(output/'result.json',result)
    return 0 if result['status']=='completed' else 1


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('contract','population','snapshot-contract','snapshot-admission','dataset-semantics',
                 'unit-manifest','source-root','dataset-root','weights-root','output'):
        parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--gpu-uuid',required=True)
    parser.add_argument('--wall-seconds',type=int,default=27000)
    args=parser.parse_args()
    if not 1<=args.wall_seconds<=27000:parser.error('Bounded unit wall budget required')
    return execute(args)


if __name__=='__main__':raise SystemExit(main())

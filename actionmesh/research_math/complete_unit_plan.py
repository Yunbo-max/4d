"""Build, never execute, one full default-release GPU calibration unit."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
from research_math.control_scoring import file_ref
from research_math.complete_unit_contract import complete_unit_output_paths

INPUT_NAMES=('contract','population','snapshot_contract','snapshot_admission',
             'dataset_semantics','unit_manifest','environment')


def validate_inventory(args):
    expected={'snapshot_admission':'admitted_engineering_snapshot',
              'dataset_semantics':'admitted_engineering_dataset_semantics',
              'unit_manifest':'frozen_engineering_current_release_unit'}
    records={}
    for key,status in expected.items():
        record=json.loads(getattr(args,key).read_text())
        if record.get('status')!=status:raise ValueError('Successful prerequisite required: '+key)
        records[key]=record
    return records


def complete_output_inventory(records):
    manifest=records.get('unit_manifest',{})
    unit=manifest.get('calibration_unit',{})
    return complete_unit_output_paths(unit.get('uid'))


def build_plan(args):
    root=args.root.resolve();plan_dir=args.plan_dir.resolve();plan_dir.relative_to(root)
    records=validate_inventory(args)
    environment=json.loads(args.environment.read_text())
    if environment['python_executable']!=sys.executable or environment['gpu_uuid']!=args.gpu_uuid:
        raise ValueError('Fresh current interpreter and allocated physical GPU required')
    if not 60<=args.wall_seconds<=27000:raise ValueError('Complete unit wall limit is 60..27000')
    if not args.gpu_uuid.startswith('GPU-'):raise ValueError('Physical GPU UUID required')
    for path in (args.source_root,args.dataset_root,args.weights_root):
        if not path.is_dir():raise FileNotFoundError(path)
    if plan_dir.exists():raise FileExistsError('Preserve existing plan')
    scripts=args.skill_dir.resolve()/'scripts';sys.path.insert(0,str(scripts))
    import run_experiments as native
    import run_harness as harness
    inputs=[file_ref(root,getattr(args,name)) for name in INPUT_NAMES]
    inputs+=environment['dependency_lock_refs']
    sources=sorted((root/'actionmesh/research_math').rglob('*.py'))
    sources += [root/'actionmesh'/name for name in ('official_actionbench_adapter.py',
        'deterministic_actionbench_entry.py','research_census_eval.py')]
    code=[file_ref(root,p) for p in sources]
    command=[sys.executable,'-m','research_math.complete_unit_runner']
    for name in INPUT_NAMES:
        if name!='environment':command+=['--'+name.replace('_','-'),str(getattr(args,name).resolve())]
    for name in ('source_root','dataset_root','weights_root'):
        command+=['--'+name.replace('_','-'),str(getattr(args,name).resolve())]
    command+=['--gpu-uuid',args.gpu_uuid,'--wall-seconds',str(args.wall_seconds),'--output','unit-output']
    revision=subprocess.check_output(['git','rev-parse','HEAD'],cwd=root,text=True).strip()
    plan=native.make_plan(root,run_id=args.run_id,purpose='engineering',evidence_mode='developmental',
        jobs=[{'trial_id':'complete-default-three-arm-unit','command':command,'cwd':'actionmesh',
            'input_refs':inputs,'code_refs':code,'output_paths':complete_output_inventory(records),
            'seed':42,'group':'engineering','arm_role':'complete-current-release-calibration'}],
        provenance={'git_revision':revision,'model_revision':'four immutable manifests in snapshot admission',
            'data_revision':records['unit_manifest']['population']['revision'],
            'environment_digest':file_ref(root,args.environment)['sha256']},
        limits={'max_attempts':1,'max_development_trials':1,'max_confirmation_trials':0,'max_retries_per_trial':0,
                'wall_time_seconds':args.wall_seconds,'attempt_timeout_seconds':args.wall_seconds})
    plan_dir.mkdir(parents=True)
    inner=plan_dir/'native.json';inner.write_text(json.dumps(plan,indent=2)+'\n')
    outer=harness.make_plan(root,batch_id=args.run_id,
        tasks=[{'task_id':'complete-default-three-arm-unit','idea_id':'baseline-qualification','depends_on':[],
            'priority':1,'plan_ref':file_ref(root,inner),
            'resources':{'cpu_cores':8,'ram_mib':32768,'gpu_count':1,'gpu_peak_mib':None,
                'allow_gpu_share':False,'memory_profile_ref':None,'exclusive_keys':['actionbench-complete-unit']}}],
        limits={'total_wall_seconds':args.wall_seconds+1800,'window_seconds':args.wall_seconds+1800,
                'max_parallel_tasks':1,'cpu_cores':8,'ram_mib':32768,'max_gpu_task_seconds':args.wall_seconds},
        gpus={'uuids':[args.gpu_uuid],'safety_margin_mib':1024,'max_tasks_per_gpu':1})
    target=plan_dir/'harness.json';target.write_text(json.dumps(outer,indent=2)+'\n')
    return {'plan':str(target),'approved_plan_digest':outer['plan_digest'],'execution_started':False,
            'scope':'one complete default-release calibration; no candidates or full128 scientific claim'}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in (*INPUT_NAMES,'root','plan_dir','skill_dir','source_root','dataset_root','weights_root'):
        parser.add_argument('--'+name.replace('_','-'),type=Path,required=True)
    parser.add_argument('--run-id',required=True);parser.add_argument('--gpu-uuid',required=True)
    parser.add_argument('--wall-seconds',type=int,default=27000)
    print(json.dumps(build_plan(parser.parse_args())))


if __name__=='__main__':main()

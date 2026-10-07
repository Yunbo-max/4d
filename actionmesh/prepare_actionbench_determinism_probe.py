"""Emit an engineering harness plan diagnosing the existing scorer backend.

The only runtime change is PyTorch's strict deterministic-algorithm guard.
All source inputs, official metric parameters and sampling budgets are retained.
A backend rejection is diagnostic evidence, never an accepted benchmark score.
"""
import argparse
import json
from pathlib import Path
import subprocess
import sys

from research_math.control_scoring import file_ref, resolve_ref, verify_request


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ('root', 'skill-dir', 'request', 'environment', 'plan-dir'):
        parser.add_argument('--'+key, type=Path, required=True)
    parser.add_argument('--run-id', required=True)
    parser.add_argument('--gpu-uuid', required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    plan_dir = args.plan_dir.resolve()
    plan_dir.relative_to(root)
    request_path = args.request.resolve()
    request = json.loads(request_path.read_text())
    verify_request(root, request)
    environment = json.loads(args.environment.read_text())
    if environment['python_executable'] != sys.executable or environment['gpu_uuid'] != args.gpu_uuid:
        raise ValueError('Current interpreter and GPU runtime identity required')
    scripts = args.skill_dir.resolve()/'scripts'
    sys.path.insert(0, str(scripts))
    import run_experiments as native
    import run_harness as harness
    plan_dir.mkdir(parents=True, exist_ok=False)
    manifest = plan_dir/'manifest.json'
    manifest.write_text(json.dumps({'cases':[{'case_id':request['uid']+'-native',
        'uid':request['uid'], 'case_dir':'native'}]}, indent=2)+'\n')
    program = ('import os,sys,runpy;'
               'os.environ["CUBLAS_WORKSPACE_CONFIG"]=":4096:8";'
               'import torch;torch.use_deterministic_algorithms(True);'
               'print("Strict deterministic guard enabled; no accepted benchmark score",flush=True);'
               'sys.argv=sys.argv[1:];runpy.run_path(sys.argv[0],run_name="__main__")')
    command = [sys.executable, '-c', program, str(root/'actionmesh/research_census_eval.py'),
        '--case-dir', str(resolve_ref(root,request['manifest_ref']).parent),
        '--gt-dir', str(resolve_ref(root,request['ground_truth_ref']).parent.parent),
        '--repo-root', str(root/request['repo_root']), '--manifest', str(manifest),
        '--output', 'deterministic-probe/scores.json', '--device', 'cuda:0', '--seed', '44']
    inputs = request['input_refs']+[file_ref(root,request_path), file_ref(root,manifest),
        file_ref(root,args.environment)]+environment['dependency_lock_refs']
    for ref in inputs:
        resolve_ref(root,ref)
    code = request['code_refs']+[file_ref(root,Path(__file__).resolve())]
    revision = subprocess.check_output(['git','rev-parse','HEAD'],cwd=root,text=True).strip()
    plan = native.make_plan(root,run_id=args.run_id,purpose='engineering',evidence_mode='developmental',
        jobs=[{'trial_id':'strict-backend-determinism','command':command,'cwd':'actionmesh',
            'input_refs':list({x['path']:x for x in inputs}.values()),'code_refs':code,
            'output_paths':['actionmesh/deterministic-probe/scores.json'],
            'seed':44,'group':'engineering','arm_role':'backend-diagnostic'}],
        provenance={'git_revision':revision,'model_revision':'none; frozen native prediction',
            'data_revision':request['ground_truth_ref']['sha256'],
            'environment_digest':file_ref(root,args.environment)['sha256']},
        limits={'max_attempts':1,'max_development_trials':1,'max_confirmation_trials':0,
            'max_retries_per_trial':0,'wall_time_seconds':180,'attempt_timeout_seconds':180})
    native_path=plan_dir/'native.json'
    native_path.write_text(json.dumps(plan,indent=2)+'\n')
    outer=harness.make_plan(root,batch_id=args.run_id,
        tasks=[{'task_id':'strict-backend-determinism','idea_id':'scorer-reproducibility',
            'depends_on':[],'priority':1,'plan_ref':file_ref(root,native_path),
            'resources':{'cpu_cores':8,'ram_mib':8192,'gpu_count':1,'gpu_peak_mib':None,
                'allow_gpu_share':False,'memory_profile_ref':None,
                'exclusive_keys':['actionbench-native-scorer']}}],
        limits={'total_wall_seconds':210,'window_seconds':210,'max_parallel_tasks':1,
            'cpu_cores':8,'ram_mib':8192,'max_gpu_task_seconds':180},
        gpus={'uuids':[args.gpu_uuid],'safety_margin_mib':1024,'max_tasks_per_gpu':1})
    (plan_dir/'harness.json').write_text(json.dumps(outer,indent=2)+'\n')
    print(json.dumps({'plan':str(plan_dir/'harness.json'),
        'approved_plan_digest':outer['plan_digest'],'execution_started':False,
        'scope':'diagnostic; nonzero exit preserves backend rejection, not a scientific outcome'}))


if __name__=='__main__':
    main()

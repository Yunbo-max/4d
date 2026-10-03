from pathlib import Path
import json,subprocess,os,time,hashlib
r=Path('/root/rivermind-data/actionmesh-repro');s=r/'research/census-20261002';o=r/'outputs/census-20261002'
p=s/'disagreement-batch-v1.json';assert not p.exists()
state=dict(status='running',pid=os.getpid(),jobs=[])
def save():
 q=p.with_suffix('.tmp');q.write_text(json.dumps(state,indent=2)+'\n');q.replace(p)
save()
env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1')
for name,uid,case,probe,direction in [('mannequin','000-037_1358c424008a43cbaa35eba5e58551ac','000-037_1358c424008a43cbaa35eba5e58551ac__seed42__attempt2','mannequin-seed42-v3','mannequin-seed42-v2'),('bat','000-043_061697e330d44524bd11f8cf95772e2d','000-043_061697e330d44524bd11f8cf95772e2d__seed42','bat-seed42-v1','bat-seed42-v1')]:
 argv=[str(r/'inference-env/bin/python'),'-u',str(s/'research_census_disagreement_rank.py')]
 for key,value in [('case-dir',o/'cases'/case),('direction-dir',o/'time-direction'/direction),('probe-dir',o/'shape-probes'/probe),('native-cache-dir',o/'motion-controls'/f'{name}-seed42-v1'),('stage-gt-dir',o/'stage-gt'/f'{name}-seed42-v1'),('gt-dir',r/'data/actionbench-census-20261002/data'),('protocol',s/'census-disagreement-rank-protocol-20261002.json'),('output',o/'disagreement-rank-v1'/f'{name}-seed42-v1'),('max-seconds',240)]:argv += ['--'+key,str(value)]
 job=dict(name=name,status='running',argv=argv,source_sha256=hashlib.sha256((s/'research_census_disagreement_rank.py').read_bytes()).hexdigest());state['jobs'].append(job);save();t=time.monotonic()
 with (s/f'disagreement-{name}-v1.log').open('x') as log:
  c=subprocess.Popen(argv,cwd=r,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True);job['pid']=c.pid;save()
  try:code=c.wait(timeout=240)
  except subprocess.TimeoutExpired:
   os.killpg(c.pid,15)
   try:c.wait(timeout=5)
   except subprocess.TimeoutExpired:os.killpg(c.pid,9);c.wait()
   code=-124
 job.update(status='completed' if code==0 else 'failed',exit_code=code,seconds=time.monotonic()-t);save()
 if code:break
state['status']='completed' if len(state['jobs'])==2 and all(j['status']=='completed' for j in state['jobs']) else 'stopped_after_failure';save()

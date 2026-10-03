"""Serial bounded runner for the declared exploratory 4D protocol."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import time


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    state=a.output/'queue.json'
    if state.exists():
        raise FileExistsError('Queue identity already exists; reconcile previous job rather than duplicate it')
    busy=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip()
    if busy: raise RuntimeError('GPU busy: '+busy)
    script=a.root/'research_three_ideas.py'
    py=a.root/'inference-env/bin/python'
    prepared=a.output/'prepare/prepared.npz'
    prefix=a.output/'prefix/prefix.npz'
    if not prepared.is_file(): raise FileNotFoundError(prepared)
    base=[str(py),'-u',str(script)]
    common=['--root',str(a.root),'--cache',str(prepared),'--seed','42']
    jobs=[('prefix',base+['prefix',*common,'--output',str(a.output/'prefix')])]
    for folder,mode in [('baseline','baseline'),('cfg_cap','cfg-cap'),('cfg_global','global-cap'),('moment_energy','moment-energy'),('rms_energy','rms-energy')]:
        jobs.append((folder,base+['run',*common,'--prefix',str(prefix),'--mode',mode,'--output',str(a.output/folder)]))
    started=time.monotonic()
    report={'started_utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),
        'status':'running','maximum_seconds':2500,'concurrency':1,
        'script_sha256':hashlib.sha256(script.read_bytes()).hexdigest(),'jobs':[]}
    def save():
        report['elapsed_seconds']=time.monotonic()-started
        tmp=state.with_suffix('.tmp'); tmp.write_text(json.dumps(report,indent=2)+'\n'); tmp.replace(state)
    save()
    for name,cmd in jobs:
        remaining=2500-(time.monotonic()-started)
        if remaining<60:
            report['status']='budget_exhausted'; save(); break
        row={'name':name,'argv':cmd,'status':'running','timeout_seconds':min(720,remaining)}
        report['jobs'].append(row); save()
        print('START',name,flush=True)
        t=time.monotonic()
        try:
            with (a.output/(name+'.log')).open('w') as log:
                r=subprocess.run(cmd,stdout=log,stderr=subprocess.STDOUT,timeout=row['timeout_seconds'])
            row.update(exit_code=r.returncode,status='completed' if r.returncode==0 else 'failed')
        except subprocess.TimeoutExpired:
            row.update(exit_code=None,status='timeout')
        row['elapsed_seconds']=time.monotonic()-t; save()
        print('END',name,row['status'],row['elapsed_seconds'],flush=True)
        if row['status']!='completed':
            report['status']='stopped_after_failure'; save(); break
    else:
        report['status']='completed'; save()
    print(json.dumps(report),flush=True)
    if report['status']!='completed': raise SystemExit(1)


if __name__=='__main__': main()

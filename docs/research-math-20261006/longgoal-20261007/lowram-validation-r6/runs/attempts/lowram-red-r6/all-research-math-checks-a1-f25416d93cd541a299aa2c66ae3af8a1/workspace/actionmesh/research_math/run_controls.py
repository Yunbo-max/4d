"""Run all constructed mathematical controls without downloading any model."""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
import time
import numpy as np


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',required=True,type=Path)
    parser.add_argument('--seed',default=42,type=int)
    args=parser.parse_args()
    args.output.mkdir(parents=True,exist_ok=False)
    from research_math.correlated import constructed_benchmark
    from research_math.controllability import constructed_demo
    from research_math.contact_events import run_demo
    started=time.monotonic()
    results={'A':constructed_benchmark(args.seed),'B':constructed_demo(),'C':run_demo()}
    def json_value(v):
        if isinstance(v,np.ndarray):return v.tolist()
        if isinstance(v,np.generic):return v.item()
        raise TypeError(type(v).__name__)
    manifest=[]
    for name,result in results.items():
        p=args.output/(name+'.json')
        p.write_text(json.dumps(result,default=json_value,indent=2,allow_nan=False)+'\n')
        manifest.append({'path':p.name,'sha256':hashlib.sha256(p.read_bytes()).hexdigest()})
    report={'status':'completed','evidence':'constructed numerical controls only',
            'natural_4d_effectiveness_tested':False,'gpu_used':False,
            'seed':args.seed,'python':platform.python_version(),'numpy':np.__version__,
            'ended_utc':datetime.now(timezone.utc).isoformat(),'elapsed_seconds':time.monotonic()-started,
            'results':manifest,'code_sha256':{p.name:hashlib.sha256(p.read_bytes()).hexdigest()
                for p in sorted(Path(__file__).parent.glob('*.py'))}}
    (args.output/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))


if __name__=='__main__':main()

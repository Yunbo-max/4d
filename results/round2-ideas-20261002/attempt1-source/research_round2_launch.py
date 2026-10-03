#!/usr/bin/env python3
"""Launch once, monitor, preserve nonzero exits. Not a permission wrapper."""
import argparse,hashlib,json,os,subprocess,threading,time
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--source',type=Path,required=True);p.add_argument('--output',type=Path,required=True);args=p.parse_args()
out=args.output;out.mkdir(parents=True,exist_ok=False)
source=args.source;start=time.perf_counter();stop=threading.Event();samples=[]
def monitor():
 while not stop.is_set():
  r=subprocess.run(['nvidia-smi','--query-gpu=memory.used,utilization.gpu','--format=csv,noheader,nounits'],capture_output=True,text=True)
  samples.append({'elapsed':time.perf_counter()-start,'value':r.stdout.strip(),'returncode':r.returncode})
  stop.wait(1)
cmd=[str(args.root/'inference-env/bin/python'),str(source/'research_round2_run.py'),'--input-root',str(args.root/'outputs'),'--output',str(out/'pilot'),'--actionbench-root',str(args.root/'data/actionbench-round2'),'--manifest',str(source/'actionbench-subset-manifest.json'),'--protocol',str(source/'round2-protocol-20261002.md')]
meta={'command':cmd,'source_sha256':{f.name:hashlib.sha256(f.read_bytes()).hexdigest() for f in source.iterdir() if f.is_file()},'start_unix':time.time(),'training':False,'subprocess_timeout_seconds':650}
(out/'launch.json').write_text(json.dumps(meta,indent=2));th=threading.Thread(target=monitor,daemon=True);th.start()
try:
 with (out/'run.log').open('w') as log:
  result=subprocess.run(cmd,stdout=log,stderr=subprocess.STDOUT,timeout=650)
  meta['exit_code']=result.returncode
except subprocess.TimeoutExpired:meta['exit_code']=124;meta['timeout']=True
finally:
 stop.set();th.join();meta['elapsed_seconds']=time.perf_counter()-start
 (out/'gpu-samples.json').write_text(json.dumps(samples,indent=2));(out/'launch.json').write_text(json.dumps(meta,indent=2))
print(json.dumps(meta,indent=2));raise SystemExit(meta['exit_code'])

import datetime,json,re
from pathlib import Path
root=Path('/root/rivermind-data/4d-native-preparation/inputs')
logs=Path('/root/rivermind-data/4d-asset-acquisition-20261009')
def observed_size(p):
    try: return p.stat().st_size
    except FileNotFoundError: return 0  # downloader can atomically rename while observed
rows=[]
for name,target,log in [('dataset',root/'actionbench-2796071c','dataset-xet-download')]+[(n,root.parent/'upstream/actionmesh/pretrained_weights'/dict(actionmesh='ActionMesh',triposg='TripoSG',dinov2='dinov2',rmbg='RMBG')[n],n+'-download') for n in ('actionmesh','triposg','dinov2','rmbg')]:
    content=[p for p in target.rglob('*') if p.is_file() and '.cache' not in p.relative_to(target).parts]
    partial=[p for p in target.rglob('*.incomplete') if p.is_file()]
    log = name+'-xet-download' if name in ('actionmesh','triposg','dinov2') else log
    log = name+'-xet-r2' if name in ('triposg','dinov2') else log
    exitpath=logs/(log+'.exit')
    rows.append({'name':name,'files_complete':len(content),'content_bytes':sum(observed_size(p) for p in content),'partial_bytes':sum(observed_size(p) for p in partial),'exit_code':int(exitpath.read_text()) if exitpath.exists() else None})
r={'at':datetime.datetime.now(datetime.timezone.utc).isoformat(),'snapshots':rows,'gpu_stop':True,'progress_is_approximate':True}
with (logs/'progress.jsonl').open('a') as f:f.write(json.dumps(r)+'\n')
print(json.dumps(r,indent=2))

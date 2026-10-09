import concurrent.futures,json,pathlib,time
from huggingface_hub import hf_hub_download,set_client_factory
from huggingface_hub.utils._http import default_client_factory
p=pathlib.Path(__file__).resolve().parent
rev='556637099bf4fa79ea7b239d0d9c328b8a2e9ac8'
lookup=json.loads((p/'actionbench-object-path-lookup.json').read_text())
paths=sorted(set(lookup.values())|{'metadata/'+v.split('/')[1]+'.json.gz' for v in lookup.values()})
def hook(r):
 if r.url.host in ('hf-mirror.com','huggingface.co') and '/xet-read-token/' in r.url.path:
  r.url=r.url.copy_with(host='hf-mirror.com').copy_add_param('_controller_refresh',str(time.time_ns()));r.headers['host']='hf-mirror.com'
def factory():
 c=default_client_factory();c.event_hooks['request'].insert(0,hook);return c
set_client_factory(factory)
def fetch(rel):
 try:
  target=hf_hub_download('allenai/objaverse',rel,repo_type='dataset',revision=rev,local_dir=p/'objaverse-xet')
  return {'path':rel,'status':'downloaded','local_path':target}
 except Exception as e:return {'path':rel,'status':'failed','error':type(e).__name__+': '+str(e)}
with open(p/'family-xet-acquisition.jsonl','x') as log:
 with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
  for future in concurrent.futures.as_completed([pool.submit(fetch,rel) for rel in paths]):
   result=future.result();log.write(json.dumps(result)+'\n');log.flush();print(result['path'],result['status'],flush=True)

import concurrent.futures,hashlib,json,pathlib,urllib.request,time
p=pathlib.Path(__file__).resolve().parent
rev='556637099bf4fa79ea7b239d0d9c328b8a2e9ac8'
lookup=json.loads((p/'actionbench-object-path-lookup.json').read_text())
paths=sorted(set(lookup.values())|{'metadata/'+v.split('/')[1]+'.json.gz' for v in lookup.values()})
def fetch(rel):
 dest=p/'objaverse'/rel
 url='https://hf-mirror.com/datasets/allenai/objaverse/resolve/'+rev+'/'+rel
 try:
  dest.parent.mkdir(parents=True,exist_ok=True)
  if not dest.exists():
   with urllib.request.urlopen(url,timeout=45) as r,open(str(dest)+'.partial','xb') as f:
    while chunk:=r.read(1024*1024):f.write(chunk)
   pathlib.Path(str(dest)+'.partial').rename(dest)
  return {'path':rel,'revision':rev,'size':dest.stat().st_size,'sha256':hashlib.sha256(dest.read_bytes()).hexdigest(),'status':'downloaded'}
 except Exception as e:return {'path':rel,'status':'failed','error':type(e).__name__+': '+str(e)}
with open(p/'family-assets-acquisition.jsonl','x') as log:
 with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
  for future in concurrent.futures.as_completed([pool.submit(fetch,rel) for rel in paths]):
   result=future.result();log.write(json.dumps(result)+'\n');log.flush();print(result['path'],result['status'],flush=True)

from pathlib import Path
import json,hashlib,shutil,tarfile,re,subprocess
from datetime import datetime,timezone
p=Path('/root/rivermind-data/4d-native-prerequisites-20261009');out=p/'delivery-v1';out.mkdir(exist_ok=False)
result={'kind':'local-native-prerequisites-window','window_id':'20261009-native-prerequisites','captured_at':datetime.now(timezone.utc).isoformat(),'gpu_stop':True,'native_results':0,'real_family_mapping_admitted':False,'attempts':[]}
for label,run in [('red','family-provenance-red-002'),('green','family-provenance-green-001')]:
 root=Path('/root/rivermind-data/4d-family-'+label);attemptroot=root/'runs/attempts'/run;attempts=[x for x in attemptroot.iterdir() if x.is_dir()];assert len(attempts)==1;attempt=attempts[0]
 dest=out/'evidence'/run;dest.mkdir(parents=True)
 paths={'native-plan.json':root/'plans'/run/'native.json','harness-plan.json':root/'plans'/run/'harness.json','harness-report.json':root/'runs/harness'/run/'report.json','receipt.json':attemptroot/'receipt.json','execution-context.json':root/'runs/harness'/run/'tasks/all-research-math-checks/execution-context.json'}
 for name in ['stdout.log','stderr.log','attempt.json','process-guard.json']:paths[name]=attempt/name
 refs=[]
 for name,path in paths.items():
  data=path.read_bytes();(dest/name).write_bytes(data);refs.append({'path':str((dest/name).relative_to(out)),'remote_path':str(path),'sha256':hashlib.sha256(data).hexdigest()})
 archive=p/(run+'-raw-evidence.tar.gz')
 with tarfile.open(archive,'x:gz') as tar:
  for rel in ['plans/'+run,'runs/attempts/'+run,'runs/harness/'+run]:tar.add(root/rel,arcname=rel)
 receipt=json.loads(paths['receipt.json'].read_text());log=paths['stderr.log'].read_text();count,seconds=re.search(r'Ran (\d+) tests in ([0-9.]+)s',log).groups();fail=re.search(r'FAILED \(failures=(\d+)',log)
 result['attempts'].append({'run_id':run,'root':str(root),'source_commit':subprocess.check_output(['git','-C',str(root),'rev-parse','HEAD']).decode().strip(),'status':receipt['status'],'tests':int(count),'test_seconds':float(seconds),'failures':int(fail[1]) if fail else 0,'errors':0,'skipped':6,'harness_digest':json.loads(paths['harness-plan.json'].read_text())['plan_digest'],'raw_archive':{'path':str(archive),'sha256':hashlib.sha256(archive.read_bytes()).hexdigest(),'bytes':archive.stat().st_size},'retained_refs':refs})
a=out/'setup';a.mkdir();setuprefs=[]
for f in sorted(p.iterdir()):
 if not f.is_file() or f.suffix in ['.bundle','.gz'] or f.name.startswith('family-') and 'harness' not in f.name:continue
 data=f.read_bytes()
 try:s=data.decode()
 except UnicodeDecodeError:continue
 s=re.sub(r'https?://[^\s\"<>]+',lambda m:'<redacted-download-url>' if any(k in m[0].lower() for k in ['token=','signature=','x-amz-','x-xet-','policy=']) else m[0],s)
 (a/f.name).write_text(s);setuprefs.append({'path':str((a/f.name).relative_to(out)),'sha256':hashlib.sha256(s.encode()).hexdigest(),'source_sha256':hashlib.sha256(data).hexdigest()})
result['setup_snapshot_refs']=setuprefs;result['setup_status']='Conda core installed; pip extension setup incomplete; asset download ongoing; no native runtime capture yet'
(out/'RESULT.json').write_text(json.dumps(result,indent=2)+'\n')
archive=p/'delivery-v1.tar.gz'
with tarfile.open(archive,'x:gz') as tar:tar.add(out,arcname='delivery-v1')
print(json.dumps({'archive':str(archive),'sha256':hashlib.sha256(archive.read_bytes()).hexdigest(),'result_sha256':hashlib.sha256((out/'RESULT.json').read_bytes()).hexdigest()}))

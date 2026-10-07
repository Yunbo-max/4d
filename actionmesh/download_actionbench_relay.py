"""Relay pinned public ActionBench files through a connected control machine.

Downloads only one UID at a time, preserving genuine Hugging Face metadata.
Transfers a checksummed archive over an existing authenticated SSH connection.
This is data staging, not experiment execution or snapshot admission.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import shlex
import shutil
import subprocess
import tarfile
import time


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(2**20), b''):
            h.update(block)
    return h.hexdigest()


REMOTE_EXTRACT = '''import hashlib,json,pathlib,sys,tarfile
archive,destination,expected=sys.argv[1:]
p=pathlib.Path(archive);h=hashlib.sha256()
with p.open('rb') as f:
 for b in iter(lambda:f.read(2**20),b''):h.update(b)
assert h.hexdigest()==expected,'Archive checksum mismatch'
root=pathlib.Path(destination);root.mkdir(parents=True,exist_ok=True)
with tarfile.open(p) as t:
 for m in t.getmembers():
  q=pathlib.PurePosixPath(m.name)
  assert m.isfile() and not q.is_absolute() and '..' not in q.parts,'Unsafe archive member'
  assert q.parts[0] in ('data','.cache'),'Unexpected archive prefix'
 t.extractall(root,filter='data')
print(json.dumps({'archive_sha256':expected,'extracted':True}))
p.unlink()
'''


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ('population', 'stage', 'ssh-target', 'control-path', 'known-hosts', 'remote-root'):
        parser.add_argument('--'+key, required=True)
    parser.add_argument('--port', type=int, required=True)
    args = parser.parse_args()
    os.environ['HF_HUB_DISABLE_IMPLICIT_TOKEN'] = '1'
    os.environ['HF_HUB_DISABLE_XET'] = '1'
    from huggingface_hub import snapshot_download
    population = json.loads(Path(args.population).read_text())
    revision = population['revision']
    assert re.fullmatch('[0-9a-f]{40}', revision)
    stage = Path(args.stage).resolve()
    stage.mkdir(parents=True, exist_ok=True)
    receipts = stage/'receipts'; receipts.mkdir(exist_ok=True)
    common = ['-o', 'ControlPath='+args.control_path, '-o', 'UserKnownHostsFile='+args.known_hosts,
              '-o', 'BatchMode=yes']
    ssh = ['ssh', *common, '-p', str(args.port), args.ssh_target]
    incoming = args.remote_root+'/relay-incoming'
    subprocess.run([*ssh, shlex.join(['mkdir', '-p', incoming])], check=True)
    for index, uid in enumerate(population['uids']):
        assert re.fullmatch('[0-9]{3}-[0-9]{3}_[0-9a-f]+', uid)
        receipt = receipts/(uid+'.json')
        if receipt.exists():
            continue
        if shutil.disk_usage(stage).free < 1024**3:
            raise RuntimeError('Keep at least 1 GiB free on the controller')
        started = time.monotonic()
        batch = stage/uid; batch.mkdir(exist_ok=True)
        files = ['data/'+uid+'/'+name for name in
                 ['camera.json','surfaces.npy', *[f'imgs/{i:02d}.png' for i in range(16)]]]
        snapshot_download(population['dataset'], repo_type='dataset', revision=revision,
                          local_dir=batch, allow_patterns=files, max_workers=4,
                          endpoint='https://huggingface.co', token=False)
        members=[]; evidence=[]
        for name in files:
            p=batch/name
            metadata='.cache/huggingface/download/'+name+'.metadata'
            lines=(batch/metadata).read_text().splitlines()
            if lines[0] != revision or not lines[1]:
                raise RuntimeError('Missing pinned Hub metadata: '+name)
            digest=sha(p)
            if re.fullmatch('[0-9a-f]{64}', lines[1]) and digest != lines[1]:
                raise RuntimeError('Hub LFS hash mismatch: '+name)
            members.extend([name,metadata])
            evidence.append({'path':name,'sha256':digest,'bytes':p.stat().st_size,'etag':lines[1]})
        archive=stage/(uid+'.tar')
        with tarfile.open(archive,'w') as t:
            for name in members:t.add(batch/name,arcname=name,recursive=False)
        digest=sha(archive); remote_archive=incoming+'/'+archive.name
        subprocess.run(['scp',*common,'-P',str(args.port),str(archive),
                        args.ssh_target+':'+remote_archive],check=True)
        result=subprocess.run([*ssh,shlex.join(['python3','-c',REMOTE_EXTRACT,
            remote_archive,args.remote_root,digest])],check=True,capture_output=True,text=True)
        remote=json.loads(result.stdout)
        assert remote=={'archive_sha256':digest,'extracted':True}
        row={'uid':uid,'revision':revision,'files':evidence,'transfer':remote,
             'elapsed_seconds':time.monotonic()-started,'scope':'download and transfer only'}
        receipt.write_text(json.dumps(row,indent=2)+'\n')
        archive.unlink();shutil.rmtree(batch)
        print(json.dumps({'completed':index+1,'total':len(population['uids']),
                          'uid':uid,'seconds':row['elapsed_seconds']}),flush=True)


if __name__=='__main__':
    main()

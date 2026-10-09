"""Asset transport only: mirror pagination and fresh public Xet read tokens."""
import sys,time
from huggingface_hub import set_client_factory
from huggingface_hub.utils._http import default_client_factory
from huggingface_hub.cli.hf import main

def repair_transport(request):
    host=request.url.host
    path=request.url.path
    if host in ('huggingface.co','hf-mirror.com') and '/xet-read-token/' in path:
        request.url=request.url.copy_with(host='hf-mirror.com').copy_add_param('_controller_refresh',str(time.time_ns()))
        request.headers['host']='hf-mirror.com'
        print('controller: refresh public Xet download token without stale mirror cache',flush=True)
    elif host=='huggingface.co' and path.startswith('/api/datasets/facebook/actionbench/tree/'):
        request.url=request.url.copy_with(host='hf-mirror.com')
        request.headers['host']='hf-mirror.com'
        print('controller: route dataset pagination through configured mirror',flush=True)

def client_factory():
    client=default_client_factory()
    client.event_hooks['request'].insert(0,repair_transport)
    return client

set_client_factory(client_factory)
sys.exit(main())

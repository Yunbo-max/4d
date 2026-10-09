"""Controller-only HF CLI transport repair; immutable asset revisions unchanged."""
import sys
from huggingface_hub import set_client_factory
from huggingface_hub.utils._http import default_client_factory
from huggingface_hub.cli.hf import main

def mirror_pagination(request):
    if request.url.host == 'huggingface.co' and request.url.path.startswith('/api/datasets/facebook/actionbench/tree/'):
        request.url = request.url.copy_with(host='hf-mirror.com')
        request.headers['host'] = 'hf-mirror.com'
        print('controller: route ActionBench metadata pagination through configured mirror', flush=True)

def client_factory():
    client = default_client_factory()
    client.event_hooks['request'].insert(0, mirror_pagination)
    return client

set_client_factory(client_factory)
sys.exit(main())

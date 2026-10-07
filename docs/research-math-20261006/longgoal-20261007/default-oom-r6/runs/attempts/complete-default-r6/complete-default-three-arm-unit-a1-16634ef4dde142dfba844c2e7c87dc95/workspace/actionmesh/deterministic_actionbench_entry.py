"""Explicit process entry for the versioned CPU-KNN-backward runtime policy.

Official metric source files and parameters are retained unchanged. Backend
provenance is written separately; this entry grants no scientific qualification.
"""
import json
import os
from pathlib import Path
import runpy
import sys


def main():
    if '--official-script' not in sys.argv:
        from research_math.deterministic_knn import run_census
        return run_census()
    index = sys.argv.index('--official-script')
    script = Path(sys.argv[index+1]).resolve()
    del sys.argv[index:index+2]
    output = Path(sys.argv[sys.argv.index('--output_csv')+1]).with_suffix('.backend.json')
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    import torch
    from research_math.deterministic_knn import cpu_knn_backward
    torch.use_deterministic_algorithms(True)
    sys.path.insert(0, str(script.parent))
    sys.argv[0] = str(script)
    with cpu_knn_backward() as metadata:
        try:
            runpy.run_path(str(script), run_name='__main__')
        finally:
            output.write_text(json.dumps(metadata, indent=2)+'\n')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())

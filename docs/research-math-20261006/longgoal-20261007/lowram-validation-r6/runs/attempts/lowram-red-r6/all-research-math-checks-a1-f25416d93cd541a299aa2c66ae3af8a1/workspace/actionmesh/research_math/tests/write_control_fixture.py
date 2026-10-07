"""Engineering integration fixture only; not an ActionBench sample or score."""
import hashlib
import json
from pathlib import Path

import numpy as np

root = Path('software-fixture/source'); root.mkdir(parents=True, exist_ok=False)
anchor = np.array([[0., 0., 0.], [2., 0., 0.], [0., 3., 0.], [0., 0., 4.]], dtype=np.float32)
vertices = np.repeat(anchor[None], 16, axis=0)
vertices[8, 3, 2] += 1.
np.savez_compressed(root/'sequence.npz', vertices=vertices, faces=np.array([[0, 1, 2], [0, 1, 3]]),
                    frame_indices=np.arange(16), timesteps=np.arange(16, dtype=np.float32), query_vertex_ids=np.arange(4))
(root/'report.json').write_text(json.dumps({'status': 'completed', 'uid': 'engineering-fixture-not-actionbench', 'seed': 0,
    'evidence_scope': 'Software-only staging and artifact lifecycle test; no scorer',
    'sha256': {'sequence.npz': hashlib.sha256((root/'sequence.npz').read_bytes()).hexdigest()}})+'\n')

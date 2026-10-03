# H6: video selection API

Implementation: `actionmesh/research_ten/m06_selection.py`.

All videos must have the same asset, prompt and view convention. Scores are observable inputs, not downstream GT. `action`, `text` and `appearance` must come from frozen scoring functions shared by every compared policy; this package does not train those scorers or assume text similarity equals action quality.

```python
import numpy as np
from research_ten.m06_selection import CandidateFeatures, track_features, select_video

# candidates.npz: tracks[C,T,P,2], visible[C,T,P] Boolean,
# reference_xy[P,2] with the same point IDs, action[C], text[C], appearance[C].
with np.load('candidates.npz', allow_pickle=False) as data:
    proxy = track_features(data['tracks'], data['visible'], data['reference_xy'])
    features = CandidateFeatures(
        tuple(f'video-{i}' for i in range(len(data['tracks']))),
        data['action'], data['text'], data['appearance'],
        proxy['proportion'], proxy['trackability'],
    )
    choice = select_video(features, valid=proxy['valid'], action_threshold=0.5)
print(choice['selected_id'])  # None means no eligible candidate: do not reconstruct.
```

`track_features` cancels image translation and global scale in pair-length ratios, then penalizes relative proportion changes and missing adjacent-frame tracks. It is not invariant to viewpoint/perspective or real articulation. Its score is a proxy to test, not a calibrated probability. Invisible track coordinates may be NaN, but visible coordinates must be finite. Missing evidence is invalid, not perfect stability. Pair sampling is deterministic and capped at 4096 by default.

Policies `feasibility`, `text`, `appearance` and `random` use one common action/validity gate. Feasibility is a weighted geometric mean of appearance, proportion and trackability. Fix weights/thresholds on development data before testing new assets.

For actual video generation and reconstruction, the callback pipeline performs the calls and records their wall times separately:

```python
from research_ten.m06_selection import run_pipeline

# Supply existing integrations: generator(seed) -> video path;
# scorer(video) -> dict(action,text,appearance,proportion,trackability,valid);
# actionmesh_backend(video) -> output path/object.
result = run_pipeline(generator, scorer, actionmesh_backend,
                      seeds=[42, 43, 44, 45], action_threshold=0.5)
```

Only the selected candidate enters the expensive backend. If none passes the gate, zero backend calls are made. Exceptions preserve failure rather than replacing a candidate. To diagnose ranking during development, separately reconstruct every candidate; do not pass those quality measurements to the selector or omit those diagnostic costs from reporting.

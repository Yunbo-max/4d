# Frozen 4D mathematical prototypes

Research code for three conditional hypotheses. All model weights stay frozen. NumPy controls establish implementation properties; they do not establish natural 4D effectiveness or novelty.

From the parent `actionmesh` directory:

```bash
python3 -m unittest discover -s research_math/tests -v
python3 -m research_math.run_controls --output ../results/math-controls-new --seed 42
```

The output directory must be new. The CPU controls require NumPy only. They preserve positive, negative, infeasible and no-change cases and baseline results.

| File | Purpose | Important limit |
|---|---|---|
| correlated.py | Low-rank shared-bias covariance and feasible local correction | Declared covariance budget is not calibrated uncertainty; backtracking is not a global constrained optimum |
| controllability.py | Output-metric reachable spaces, trust-ball least squares, model/output comparison | Local reachable rank does not prove global or semantic edit feasibility |
| contact_events.py | Typed event alignment and endpoint contact repair | Constructed endpoint problem, not complete multi-body physics |
| native_probe.py | Cached frozen ActionMesh decoder, finite differences, actual redecoding and GPU telemetry | Sparse query-interface probe with constructed edit requests; no full mesh or natural quality gain claim |
| protected_projection.py | C02 weighted action-subspace projection with redundant-constraint and pin handling | Declared action observations are not ground-truth motion; native qualification is still pending |
| actionbench_full_reproduction.py | Fail-closed checking for the conditional full-128 current-README reproduction contract | Runs no scorer, does not qualify one UID or a candidate effect, and cannot authorize dispatch |

C02 implementation status (2026-10-06): four local NumPy tests and the same four
tests in the remote inference environment pass. No candidate GPU generation or
official ActionBench comparison has been run yet.

The native probe uses the already installed ActionMesh inference environment, weights and complete census caches. It never downloads or trains a model. Example on the existing GPU server, from its new research directory:

```bash
/root/rivermind-data/actionmesh-repro/inference-env/bin/python -m research_math.native_probe \
  --root /root/rivermind-data/actionmesh-repro \
  --case-dir /root/rivermind-data/actionmesh-repro/outputs/census-20261002/cases/000-037_1358c424008a43cbaa35eba5e58551ac__seed42__attempt2 \
  --output /root/rivermind-data/actionmesh-repro/outputs/math-prototypes-20261002/native-mannequin-new \
  --points 24 --max-seconds 600
```

It requires an available CUDA device, checks original source hashes, validates native replay before interpreting edits, keeps all16 context frames, and limits the process allocator to12GiB. Each asset's report includes observed GPU use, every accepted/rejected edit trial and the original source identity. CUDA algebra parity for A is explicitly separated from B's native decoder test. C has no qualified natural multi-entity benchmark in this delivery.

The precise frozen developmental protocol is in `results/math-prototypes-20261002/protocol.md` at the repository root. Full FreeOrbit4D/PREX reproduction is outside the current22GiB qualification; constructed controls cannot replace those missing natural inputs.

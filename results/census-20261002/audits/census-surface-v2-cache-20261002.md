# Exact surface baseline v2: cache-only performance repair

V1 remains unchanged at `actionmesh/research_census_surface_baseline.py`, SHA256 `6c64929d3a9380bbf3ecf4e130ddd7a4a1418628855eb05e6c5009ec3dd5b4ff`. Its timed-out partial outputs remain evidence. This repair does not reinterpret their geometry or qualification status.

New implementation: `actionmesh/research_census_surface_baseline_v2.py`.
New bounded CPU qualification harness: `actionmesh/research_census_surface_query_check.py`.

The inspected installed dependency source is Trimesh 5.1.0, archived at `evidence/local/census-20261002/trimesh-proximity-source.txt`. Its `nearby_faces` creates a SciPy cKDTree on all referenced vertices on each call. With the frozen query batch of 128 this rebuild occurs approximately 156 times per 20k-vertex pass, although the target geometry is fixed.

V2 builds one identical tree per owned, fixed target mesh. It copies the original `nearby_faces` function object with a private globals dictionary, replacing only the cKDTree constructor with the cached instance factory. It similarly copies `closest_point`, binding its `nearby_faces` reference to that owned callable. It does not mutate the dependency module, copy/reimplement its triangle arithmetic, reorder candidates, change KDTree options, or alter tolerances. Trimesh's Rtree `intersection_v` path and fallback, closest-point arithmetic, and face-normal tie selection execute their original installed bytecode. V2 records source/signature hashes and tolerance for both dependency functions, plus cache counters.

Target vertex/face arrays are locked read-only; the cache rejects a different target, replaced geometry arrays, or unlocked arrays. These targets are private and never modified by the fitting algorithms. The official repeated referenced-vertex array selection remains, avoiding an unnecessary second optimization.

AST checks passed locally: all preexisting constants match; all geometry/correction/solver functions match; the `run` function is identical after removing its additional final provenance field. Syntax compilation and CLI help passed. Local Python lacks SciPy/Trimesh/Rtree, so the numerical harness could not execute here; its attempted run stopped at the dependency import.

The root orchestrator subsequently reported actual remote CPU qualification passing for **both assets**, including synthetic fixtures and all three actual frames with bitwise-equal candidates, closest points, distances, and triangle IDs: mannequin qualification 18.744 seconds, bat qualification 25.499 seconds. These are total qualification runtimes, not baseline speedup measurements. Root retains the actual reports and is preparing the unchanged 600-second engineering retry. Full-path identity/closest/normal outputs will additionally be compared against the completed v1 arms; ARAP v1 did not complete.

Independent implementation review by `time_reversal_collision` returned source-only PASS with no launch blocker. It verified private globals, original bytecode/defaults/closures, unchanged vectorized/fallback candidate ordering, owned immutable geometry checks, and lazy property materialization before locking arrays. The reviewer treated the reported real-data numerical qualification as resolving runtime property-identity uncertainty. No further edits were requested.

Qualification requires bitwise-identical closest points, distances and triangle IDs, and exactly equal candidate order. Synthetic fixtures cover multiple batches, equidistant parallel surfaces, duplicate/opposite-winding triangles, points on edges/vertices/surfaces, and an unreferenced vertex that must not count as surface. They also check cache ownership and unchanged public module functions. The actual-data mode uses fixed evenly spaced source vertex indices at frames 0, 8 and 15; it reads no GT or alignment. Every actual case must pass before its new baseline run.

Example commands in the remote script directory (paths supplied by the root orchestrator):

```sh
python research_census_surface_query_check.py \
  --extra-python "$EXTRA_PYTHON" --actual-data \
  --case-dir "$CASE_DIR" --probe-dir "$PROBE_DIR" \
  --points 256 --max-seconds 180 --output "$NEW_QUALIFICATION_JSON"

python research_census_surface_baseline_v2.py \
  --root /root/rivermind-data/actionmesh-repro \
  --case-dir "$CASE_DIR" --probe-dir "$PROBE_DIR" \
  --extra-python "$EXTRA_PYTHON" --output "$NEW_V2_OUTPUT" --max-seconds 600
```

Four arms, three frames, trust radius, normal updates, ARAP iterations, convergence threshold and batch size remain frozen. No GT-guided tuning, method-strength changes, GPU execution, or remote launch was performed by this agent.

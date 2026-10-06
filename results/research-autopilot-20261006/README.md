# Research_Autopilot supervised runs (2026-10-06)

These artifacts were produced on the authorized Linux GPU host with Research_Autopilot `23adf94`.

- `c02-engineering-20261006-v2`: two independent CPU workers ran in parallel. C02 tests and the complete `research_math` suite both completed successfully. GPU occupancy was zero.
- `native-probe-c02-20261006-v2`: one exclusive RTX 2080 Ti worker ran the frozen ActionMesh native decoder diagnostic. The harness recorded completion, fixed GPU UUID `GPU-b544b42e-15d3-c9c8-1bdb-4c339775a740`, about 22.5 GPU-occupied seconds, and approximately 2.8 GiB sampled peak physical memory.

The native probe is a developmental interface diagnostic: frozen weights, no training, no gradients, constructed edit requests, 24 query points, and target frames 1/8/15. It is not a full ActionBench quality comparison and does not advance a scientific gate. The retained `report.json` contains the parity, cache replay, Jacobian and CUDA algebra checks.

The first engineering batch and first native probe attempt are retained on the remote host as failed/reconciliation evidence; they are not overwritten by the v2 runs. The first native probe itself completed, but its first harness record had an incorrect staged output declaration; v2 corrected the declaration and reran with a fresh run ID.

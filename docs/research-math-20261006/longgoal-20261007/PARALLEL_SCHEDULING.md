# Same-card concurrency decision, 2026-10-07

The user explicitly requested filling spare capacity with useful parallel work.
The remote installed research-autopilot is newer than the local skill copy. Its
SKILL.md requires qualified peak VRAM, headroom, and the implemented harness;
dependency-resource-scheduling.md also requires real useful-throughput evidence.

At 15:39:08 UTC the active first population unit used 10,255 MiB and reported
100% GPU utilization on GPU-4910bb04-2d00-ca81-f64a-2031568758ad (22,528 MiB).
The earlier 4,817 MiB observation was an earlier generation stage. Two copies
at the observed 10,255 MiB plus the current 1,024 MiB device margin would use
21,534 MiB, leaving only 994 MiB. These samples are not conservative enforced
peak bounds; neither safe two-job admission nor a speedup has been established.

Current batch population-gpu-current-r9 has max_parallel_tasks=1,
max_tasks_per_gpu=1, allow_gpu_share=false, null memory profiles and a common
exclusive key. Do not mutate a frozen running batch or bypass its host lock.
The first complete unit supplies the next current-device timing/peak evidence.

Next operational qualification: prepare a separately identified bounded
co-location pilot at a safe batch boundary with isolated outputs and unchanged
scientific parameters, compare two complete useful units against their serial
cost, retain OOM/slowdown records, and pin workload/device/source/input identity.
Do not manufacture profile measurements to get past sharing checks. Before
sharing admission, check worst-case residency and framework bounds across all
stages, and resolve the existing common exclusive key by inspecting actual
mutable assets. Use one harness owner. If the conservative envelopes do not fit
or completed-unit throughput does not improve, retain exclusive GPU execution.
Independent design/code preparation can proceed while GPU work runs.

No parallel GPU pilot has been launched; no speedup is claimed. Candidate
comparisons remain 0/15. This is a scheduling obligation, not a method result.

# Dependency and resource scheduling for research workers

Use with rolling-research-batches.md and multi-route-supervision.md when the human
wants independent work to overlap during a research window. These are engineering
planning/admission contracts. The [local execution harness](local-execution-harness.md)
implements a single-host executor; reading a skill neither connects a GPU nor
installs Orca. Preserve G01/E04 and the scientific gates.

## Task DAG and resource admission

New/reopened discovery first mathematically generates about 20 candidates and
verifies top-15 ranking/selection before code design under method-verification.md.
Decompose the selected 15 ideas, or the smaller later slate, into concrete tasks:
reading/design, implementation, qualification, training/inference, native scoring,
E04 verification and export. Keep at most 20 active independent ideas per project/
navigation goal. Idea count, agent count and running GPU-job count are different.
Each admitted task has a bounded owner/spec; one worker can serve several tasks.

Build a task DAG from actual artifact/decision dependencies. The research map's
84 task types and 221 relationships are an exploration graph, not this execution
DAG: graph adjacency neither orders all work nor establishes readiness. Repeated
research loops can produce new versioned tasks; each execution DAG remains acyclic.
Share a qualified baseline/upstream artifact only when versions and the scientific
contract permit; pin it read-only and avoid duplicated qualification or training.

Admission requires all of the following:

1. The task is scientifically/operationally ready, authorized and not already
   running. All required inputs/decisions exist at the pinned revisions and pass
   their applicable acceptance checks; a worker's completion message alone does
   not certify an upstream scientific result.
2. Its actual host, dependencies, resource allocation and cumulative budget are
   valid. Reserve capacity atomically through one host-enforced allocator; several
   agents independently reading free memory and launching is not a reservation.
3. Its branch/workspace/output paths and mutable shared assets do not conflict
   with active writers; retain protocol/confirmation isolation and one integrator.
4. Its conservative peak resource envelope fits available CPU/RAM/storage/I/O,
   GPU memory/compute and agent/API limits, with recorded operating headroom.
   Unknown GPU peaks or exclusive/multi-GPU jobs wait for bounded native-workload
   profiling or exclusive allocation, not guessed sharing.

Independent ready tasks may start together within the admitted limits, even when
their idea appears later in the slate. A dependent task waits for its own inputs,
not for unrelated ideas to finish. Event-driven dispatch considers completions,
failures, blocked dependencies and actual resource changes. Fill spare capacity
with useful ready work that does not delay a higher-priority reserved task; keep
its stop/admission rules. No unapproved new idea or adaptive experiment is created
to fill a slot. If no eligible task fits, retain the queue and report the reason.

Prefer decision-driving, consequential questions and dependency-critical work;
use remaining time/resources to place independent tasks without starving deferred
work. Record the priority rationale, wait age and actual dependencies rather than
a universal numerical score. Do not serialize everything by idea number or force
all 20 ideas to become 20 continuously running agents.

### Separate agent and execution concurrency

Use a bounded reusable worker pool with one coordinator/integrator and distinct
agent/API-token limits. Independent readers, isolated code writers and CPU-native
scoring may overlap a GPU experiment when their inputs are ready. Scoring that
uses a GPU enters the GPU queue. Locally hosted agent models also consume GPU
resources; API-hosted agents do not imply local model residency.

Every worker spec identifies target files/artifacts, exact task/change, invariants,
ownership, pinned inputs, prerequisites, resource/attempt/time bounds, expected
outputs and observable acceptance. Retain actual task, dispatch/attempt, route,
job, PID/provider identity and start/end evidence. Heartbeats/contact establish
liveness, not correctness. Completion is accepted against the active attempt and
actual outputs; stale messages cannot complete a newer attempt. Reconcile unknown
SSH/process status before retry or resource release. An accepted completed worker
may be reused/released under its real host contract; no duplicate job on timeout.

### Single-GPU sharing

A reported 22 GB GPU is an illustrative capacity, not a confirmed live inventory.
Read actual usable capacity and external usage. For co-located GPU jobs, require:

`sum(admitted conservative peak VRAM) + other accounted usage + operating headroom <= usable device VRAM`

Keep consistent bytes/MiB/GiB units, include model/optimizer/activation/cache/framework
overhead and transient peaks, and distinguish peak estimates from current free
memory. Validate that the actual frameworks honor the declared memory envelope;
a fractional GPU scheduling token does not isolate VRAM or guarantee compute.
If reliable bounds/enforcement are unavailable, allocate the GPU exclusively.

For illustration only, jobs with conservatively qualified 8 GB and 6 GB peaks on
an otherwise available 22 GB device may fit together with measured headroom;
a 20 GB job usually leaves little room for another GPU job, while independent
CPU/API work can still proceed. These numbers are not profiled project receipts,
a throughput claim or authorization to launch. Compare real total useful throughput
and slowdown: fitting memory does not imply concurrent training finishes faster.

Bound initial GPU concurrency, profile representative official development work
under the declared conditions, and retain co-location/host contention in run
metadata. On OOM or unacceptable contention, preserve failed outputs, stop further
admissions and apply frozen recovery rules; never secretly lower batch size,
precision, sample count or change seeds to make a scientific comparison fit.
Runtime/efficiency claims need comparable resource/contention conditions or
exclusive measurements. Confirmation remains frozen and prohibits inspected retries.

### Existing implementation boundary

The native runner is serial within each plan. `scripts/run_harness.py` now
coordinates independent native plans concurrently with actual host admission,
retained state and eight-hour reports; use local-execution-harness.md for its
frozen-plan, watchdog, recovery and qualification boundaries. The local supervisor registry reserves whole
GPU counts and does not support fractional GPU or VRAM subleases; do not change a
declared count, reuse another route's token or bypass conflicts to pretend otherwise.
One actual execution-owner route can hold the whole GPU and an implemented host
allocator can manage its internal jobs under that ownership; other coding/read
routes submit requests and do not independently launch against that lease.
Report subjob scheduling as host-managed metadata, not a capability of the existing
registry schema. Without using that implemented adapter, keep GPU jobs serial while independently
authorized CPU/API/agent work can overlap. Do not auto-install an execution framework.

## Eight-hour window handoff

At each default eight-hour boundary, persist the actual task DAG, completed outputs,
running jobs, ready/blocked queue, peak estimates, reservations and cumulative
time/spend. Carry eligible unfinished approved work into the next window with
stable idea/task/run/protocol identity; a window does not reset attempts or require
relaunching a live process. Frozen hard limits and lease expiry still apply.
Report useful completed comparisons separately from partial runs. Scientific
convergence, expansion and new validation/writing branches use human review;
runtime placement among already approved tasks needs no new scientific direction.
Three consecutive windows require a host that actually persists/executes them.

## Optional execution backends and primary references

Checked 2026-10-04. Orca is an ambiguous project name; these findings concern
[Stably Orca](https://github.com/stablyai/orca), not every project called ORCA.
No optional dependency is installed or connected by this update.

- [Orca orchestration](https://www.onorca.dev/docs/cli/orchestration) and its
  [actual guide](https://github.com/stablyai/orca/blob/main/skill-guides/orchestration.md)
  describe Run/task/dispatch identities, dependencies, isolated worker ownership,
  completion/heartbeat messages and decision gates. A Run does not schedule/place
  workers. Use its installed version-matched guide for real commands; do not copy
  retired scheduler commands or equate its engineering gates with scientific PASS.
- [Ray accelerator scheduling](https://docs.ray.io/en/latest/ray-core/scheduling/accelerators.html)
  supports fractional GPU resource declarations; applications remain responsible
  for staying within accelerator memory. This is an optional execution backend,
  not evidence that this skill currently schedules GPUs.
- [Ray fractional LLM serving](https://docs.ray.io/en/latest/serve/llm/user-guides/fractional-gpu.html)
  documents workload-specific memory preallocation and headroom. Its vLLM settings
  apply to that serving stack, not an automatic recipe for arbitrary training.

For a single GPU host, choose the smallest implemented admission/dispatch layer
that meets these contracts; an agent manager can manage workers while a separate
allocator manages GPU jobs. Recheck actual backend/version/capabilities before
implementation or deployment and retain real process/readback evidence.

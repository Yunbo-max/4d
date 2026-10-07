# Local execution harness for rolling research windows

Use `scripts/run_harness.py` on the actual Linux execution host. A local Codex
agent can prepare/validate it and start it through an already authorized SSH
connection. Python 3.11+ and the standard library are sufficient for CPU tasks;
GPU admission additionally requires working NVIDIA `nvidia-smi` telemetry. No
Ray/Orca install, continuously active chat worker, API call or paid service is
needed by the scheduler. The existing native runner remains the per-task executor.

## Preparation and frozen scope

The agent first joins the current papers, actual implementation and official
benchmark/scorer with G01 design. In the web/local workflow, reuse the approved
web-prepared linked reading/design record and inspect affected new dependencies;
local execution does not require regenerating the whole candidate investigation.
Confirm/reuse the current host/GPU inventory,
GitHub/HF destinations, time window and finite total budget. An eight-hour review
request alone is not an unlimited compute grant. Keep at most 20 active ideas
per navigation goal. New/reopened discovery derives about 20 mathematical
candidates and verifies ranked top-15 selection before code design under
method-verification.md; resume reuses that pool and later convergence keeps fewer.
Plan full controls/scoring/E04/export obligations; do not replace them with quick
tests, more seeds or arbitrary new ideas to occupy a GPU.

Read the host without launching a workload:

```bash
python3 "$SKILL_DIR/scripts/run_harness.py" --inspect-host
```

Prepare real reviewed `experiment-run-plan` files using native-runner.md. Each
harness task binds one such file by path and SHA256, with stable `task_id`,
`idea_id`, prerequisite task IDs, integer priority and a resource envelope.
Use one atomic foreground workload per native plan when task-level concurrency
is needed: jobs inside a native plan still run serially. Existing frozen train/
inference/evaluation entry points can perform several approved stages internally.
Keep one integration writer; pinned inputs are staged into isolated attempts.
Declare mutable caches/export targets as `exclusive_keys` when they may conflict.

`make_plan(root, *, batch_id, tasks, pool_dir=None, limits, gpus=None,
output_root='runs/harness')` validates and returns a digested `harness-plan`.
`window_seconds` defaults to 28800 in this builder; the remaining limits must be
specified from actual authorization. `schemas/harness-plan.schema.json` is the
exact strict file contract. A task resource envelope contains:

| Field | Meaning |
|---|---|
| `cpu_cores`, `ram_mib` | Conservative host reservations, including command descendants |
| `gpu_count` | 0 for CPU; otherwise actual UUID-based device count |
| `gpu_peak_mib` | Qualified conservative peak; null means unknown/exclusive |
| `allow_gpu_share` | False by default; true requires a pinned qualified profile |
| `memory_profile_ref` | Pinned `gpu-memory-profile`, or null |
| `exclusive_keys` | Shared mutable assets that cannot have simultaneous writers |

`limits` includes finite `total_wall_seconds`, `max_parallel_tasks`, `cpu_cores`,
`ram_mib`, and `max_gpu_task_seconds`. The last is a conservative planned native
task-time bound: the sum of each GPU task's native wall limit times GPU count
must fit. Reserve overhead in the external compute/spending budget. It is not
an invoice cap or measured physical GPU utilization. Retries stay inside each
native plan's attempt/development/time limits; confirmation never retries.

`gpus` freezes actual `uuids`, positive `safety_margin_mib`, and
`max_tasks_per_gpu`. Use a single common `pool_dir` for all cooperating workers
on a host (default `~/.local/state/research-autopilot`). One driver manages that
host pool; other agents submit approved work to it rather than launching their
own GPU processes. This layer operates under one execution-owner route's valid
whole-GPU lease; the existing supervisor registry has no VRAM sublease schema.
External provider budgets/lease expiry must already bound any paid infrastructure.

### Dependency scope

The DAG orders execution-ready, already pinned native plans. All input/code/
protocol files must exist and validate before execution. It does not invent the
future hash of a not-yet-produced checkpoint, prediction or native evaluation
record. A task's successful exit/declared files establish execution completion;
they do not qualify a scientific upstream result.

When a downstream plan needs newly generated artifacts or a scientific decision,
the agent must first collect/qualify those actual outputs, run E04/native checks
and bind them in a new ready frozen plan under existing authorized conditions.
Keep that item as a documented blocked obligation until then. Do not insert
placeholder hashes, auto-promote a scientific gate, or mutate the running batch.
An independently ready task can proceed during this wait. Automatically generated
artifact-to-input DAG binding is not implemented by this version.

## Dispatch and GPU admission

The scheduler considers completed prerequisites, descending frozen priority,
available task slots and CPU/RAM reservations. Independent tasks can overlap
regardless of idea numbering. Failed dependencies block their descendants and
leave other tasks eligible. RAM admission checks the aggregate conservative
reservations against both the declared envelope and live MemAvailable; this is
deliberately conservative under memory pressure. Reservations are admission
accounting, not cgroups or an OS memory partition. Commands/frameworks must honor
the reviewed envelope; do not silently alter batch size, dtype or seeds to fit.

GPU placement uses stable device UUIDs. CPU workers receive an empty
`CUDA_VISIBLE_DEVICES`; GPU workers receive only their admitted UUIDs. Retain the
actual allocation and declared envelope in `execution-context.json`. NVIDIA
query failure, unsupported/ambiguous memory values or unaccounted foreign GPU
processes block new GPU admissions. CPU tasks may still proceed.

Unknown peaks and multi-GPU workloads get exclusive devices. Same-card sharing
requires actual memory and useful-throughput qualification, pinned measurement/
throughput refs and `gpu-memory-profile` workload/device/concurrency bindings.
The profile also freezes allowed `co_location_workload_digests`: both profiles
must permit each other. Reuse a profile only for its exact workload/environment/
input/protocol identity. A JSON profile and hash do not themselves establish that
the measurements are scientifically valid; the agent retains that qualification.
Prefer exclusive runs for resource/efficiency claims or unqualified confirmation.

Admission accounts for the larger of conservative reserved peaks and observed
own allocation, plus foreign memory and operating headroom. An observed own
allocation above its reserved envelope prevents additional sharing. It does not
infer free capacity from a training job's small instantaneous allocation. No
memory partition/fractional accelerator token is claimed. Review real throughput:
co-location that fits VRAM can still be slower or change a timing comparison.

The exclusive driver lock serializes admission. Shared execution/device ownership
locks are bound to the current batch digest, inherited by its worker/watchdog/
command and retained while they run. A restarted driver can rejoin its own batch;
another batch must obtain exclusivity before changing the owner. Locks are
cooperative among users of the same host pool, not physical fencing of unrelated
accounts or programs. NVIDIA telemetry detects foreign GPU use before admission.

## Eight-hour reports and recovery

Default windows are eight hours. They generate consolidated reports and retain
unfinished work; they do not terminate a healthy task. A finite 24-hour approved
budget can cover three consecutive windows. The separate native job/attempt and
harness total deadlines still apply, using the host monotonic clock. Reboot,
host changes and uncertain process identity require reconciliation, rather than
resetting the clock or consuming a fresh budget. A batch that completed before
its deadline stays completed when inspected later.

Keep the driver on the execution host in an existing `tmux` session or an authorized
`nohup` invocation if SSH disconnects are expected. No background service is
installed by reading the skill. The Linux watchdog has its own deadline, responds
to worker death, kills the native process group, retains its receipt when possible
and holds ownership locks until cleanup. A command that leaves background
children after its foreground launcher exits is interrupted and cannot qualify;
scorer replay also rejects that result despite a zero launcher exit code.
Commands must remain in their foreground process group. Deliberately detached
sessions/daemons, privileged adversarial jobs and multi-host physical fencing
need a real container/cgroup/provider adapter outside this lightweight harness.

Validate a draft and print its canonical digest without starting jobs:

```bash
python3 "$SKILL_DIR/scripts/run_harness.py" draft-harness.json \
  --root "$PROJECT_DIR" --freeze > reviewed-harness.json
python3 "$SKILL_DIR/scripts/run_harness.py" reviewed-harness.json \
  --root "$PROJECT_DIR"
```

Execute only the already authorized exact finite scope, on the GPU host:

```bash
python3 "$SKILL_DIR/scripts/run_harness.py" reviewed-harness.json \
  --root "$PROJECT_DIR" --execute --approved-plan-digest REVIEWED_SHA256
```

The same command resumes the same batch. It verifies retained receipts and output
hashes, reconciles `/proc` PID/start-time/boot identities, reuses completed tasks
and keeps live jobs. A persisted dispatch intent with no provable process/result
stays `reconciliation_required`; it is never an automatic retry. Lost workers can
leave partial outputs; retain and investigate them. Do not delete state/locks or
invent a new run ID as a recovery shortcut. Apply existing confirmation policy.

`--stop-after-report` hands off the driver at the next boundary while existing
workers retain their deadlines/locks; restart with the same command to admit
further ready work. Leave the driver running for continuous useful dispatch.
`--status` reads the last retained state and launches nothing; it labels that
observation rather than claiming newly checked liveness.

### Retained artifacts and cost controls

Under `runs/harness/<batch_id>/`, retain `plan.json`, atomically replaced
`state.json`, `events.jsonl`, final/current `report.json`, `reports/window-*.json`
and each task's actual process/context/result/worker logs. Native attempt/run
receipts remain at their own frozen paths, with complete stdout/stderr, output
refs, failed attempts and watchdog refs. The scheduler verifies the canonical
run/attempt identities, actual declared outputs and receipt hashes. Reports
always retain `gate_advanced: false` and `scientific_result_verified: false`;
the agent must perform G01/E04/native scientific acceptance and human handoff.

The scheduler makes zero LLM calls. Reuse a small number of coding/reading agents
and let the process pool wait/dispatch. Completed task IDs do not repeat on
resume; share already qualified immutable baselines under their actual contracts.
Hash large files in bounded 1 MiB buffers, once per unique output per verification
boundary. Staging tries an isolated filesystem reflink, falling back to a normal
copy; it never uses writable hard links into pinned sources. This can reduce
checkpoint/data copying when the host filesystem supports copy-on-write.

Reports give elapsed/remaining wall time and the union of per-device task
ownership intervals (`gpu_device_occupied_seconds`), so overlapping jobs do not
double-count that duration. This measures reservation time, including staging,
not GPU utilization or paid billing. The planned native task-time bound is shown
separately. No price, total cost saving or GPU speedup is invented. Actual
GitHub/HF pushes remain the agent's authorized delivery work with readback,
not an automatic upload performed by this executor.
At each boundary and finite-queue completion, the local agent follows
[repository-round-trips.md](repository-round-trips.md)'s result-packet contract:
collect a coherent snapshot, record E04 status and unfinished work/budgets, push
to the saved project target, verify remote content and give the human a pinned
repo/commit/path for web analysis. A report with running jobs is a window handoff,
not a finished queue. Web-proposed code never mutates those active attempts.

## Engineering verification boundary

Use short CPU-only engineering commands to verify real overlap, dependency wait,
window continuation, stable IDs, hard deadlines, missing-output/failed dependency
handling, worker-death cleanup, mutable-asset locks, bounded-memory hashing and
CLI validation. Hardware doubles test resource admission and profile bindings;
they are not proof of live GPU throughput or SSH deployment. Running these tests
does not run research, qualify a method, create a benchmark or satisfy E04.

Primary interfaces checked 2026-10-04: [Python subprocess](https://docs.python.org/3/library/subprocess.html),
[Python fcntl](https://docs.python.org/3/library/fcntl.html),
[NVIDIA SMI](https://docs.nvidia.com/deploy/nvidia-smi/index.html).

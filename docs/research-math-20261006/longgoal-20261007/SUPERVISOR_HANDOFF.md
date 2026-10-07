# Finite campaign supervisor handoff

Status: **`generated_unexecuted`**. The supervisor source and its acceptance
checks require Local qualification. This authoring round has executed no project
tests, harness campaign, inference, scorer, or GPU workload. Historical receipts
do not qualify this source revision.

`scripts/research_supervisor.py` supervises a finite set of already frozen
`research-autopilot` harness plans. The installed `scripts/run_harness.py` remains
the executor and retains resource admission, native attempts, process ownership,
timeouts, receipts, and same-batch resume. The supervisor does not implement a
replacement executor, SSH transport, or an AI coding agent.

## Local entrypoint

The following commands describe the interface; they were not executed in this
authoring round. Run them on the local Linux execution host with the actual
absolute paths frozen in the manifest.

```bash
# Read-only inspection; launches no harness or workload.
python scripts/research_supervisor.py CAMPAIGN.json

# Read-only campaign digest output for review.
python scripts/research_supervisor.py CAMPAIGN.json --print-digest

# Execute only the exact reviewed finite campaign.
python scripts/research_supervisor.py CAMPAIGN.json \
  --execute --approved-campaign-digest SHA
```

`SHA` must be the reviewed campaign digest. `--poll-seconds` is an optional
polling interval between `0.05` and `30` seconds; it changes observation cadence,
not authorization, retry limits, deadlines, or collection reserve. Inspection
and digest output are not execution approval or software acceptance.

GPU dispatch remains stopped by default. For future Local GPU execution, the
optional `--allow-gpu-after-user-resume` flag acknowledges that the user has
explicitly resumed GPU work and that the exact plan and scientific admission
gates have independently been accepted. The flag is not permission proof and
cannot establish those prerequisites itself. It was not invoked in this
authoring round. There are no GPU permission fields in the manifest.

The import API defaults to
`run_campaign(..., allow_gpu_after_user_resume=False)`. The acknowledgement is
specific to the invocation; a saved campaign state or a prior invocation must
not automatically restore it. Existing live GPU state without the flag is
observation-only and must not cause a harness driver resume.

The manifest's project root, Python interpreter, skill root, and host pool must
resolve to their matching local Linux paths. A controller on another machine
cannot treat those strings as a remote connection. This CLI implements no SSH
login, file transfer, remote installation, or host migration.

## Manifest v1

The manifest has exactly these top-level keys:

| Key | Required value or meaning |
| --- | --- |
| `kind` | `"research-harness-campaign"` |
| `version` | `1` |
| `campaign_id` | Stable identity used for retained supervisor state |
| `root` | Absolute project root containing the pinned plans and their evidence |
| `python` | Absolute compatible Python interpreter path |
| `skill_root` | Absolute complete installed `research-autopilot` skill root |
| `skill_digest` | Digest of that complete skill tree using the runtime `digest_tree` algorithm |
| `pool_dir` | Absolute shared host-pool directory used by every campaign harness plan |
| `total_wall_seconds` | Finite campaign budget, at most `28800` seconds |
| `collection_reserve_seconds` | Collection reserve of at least `1800` seconds within that budget |
| `plans` | Finite list of pinned plan entries described below |
| `campaign_digest` | Digest binding the reviewed campaign manifest |

Every entry in `plans` has exactly these keys:

| Key | Meaning |
| --- | --- |
| `id` | Unique campaign plan identity |
| `plan_ref` | Exactly `{ "path": "project-relative/path.json", "sha256": "…" }` |
| `plan_digest` | The referenced harness plan's own validated digest |
| `dependencies` | List of campaign plan IDs that must complete before this plan may start |
| `on_failure_of` | `null` for ordinary work, or the exact parent plan ID whose known terminal failure gates this repair child |

All referenced harness plans, native plans, code, and inputs must already exist
and pass their pinned hash and plan checks. A manifest does not accept arbitrary
commands or shell snippets. Native run identities and trial IDs must be unique across the
whole campaign, including conditional repair children. Dependencies and repair
links must form a valid finite graph; they do not supply hashes for future
outputs or mutate an active plan.

`skill_digest` follows `research_autopilot.artifacts.digest_tree`: hash each
regular file's bytes, record its relative POSIX path and SHA-256, and hash the
canonical JSON inventory produced by sorted directory and filename traversal.
The runtime excludes `.git`, `__pycache__`, `.pytest_cache`, `.mypy_cache`,
`.ruff_cache`, `.venv`, `venv`, `.runtime`, `runtime-state`, and `node_modules`
directories, plus files ending in `.pyc`, `.pyo`, `.db`, `.sqlite`, `.sqlite3`,
`-wal`, or `-shm`. It rejects included symlinks and nonregular files. Pin the
actual complete installed skill; a digest of only `run_harness.py` is not this
contract.

## Failure, repair, and resume

A repair child is an already reviewed frozen plan, eligible only when its named
parent has exact status `failed`, backed by a verified native terminal receipt
with status `failed`, and its declared dependencies have completed. An
`interrupted` receipt does not qualify. It is
not permission to generate new code, alter inputs, rerun a scientific UID, reset
an attempt, or choose an unlisted command. An uncertain
process, lost receipt, or unresolved harness state requires reconciliation;
uncertainty is not a repair trigger. Independent eligible work can continue
after a known failure within the frozen campaign and remaining budget.

The native runner's existing internal retry limits remain unchanged. Current
one-attempt/no-retry plans keep `max_attempts=1` and
`max_retries_per_trial=0`; the supervisor does not widen them. Conditional repair
plans have their own reviewed identities and cannot be used to bypass the
project's no-retry, coverage, or scientific admission rules.

Supervisor records are retained under:

```text
runs/supervisor/CAMPAIGN/state.json
runs/supervisor/CAMPAIGN/manifest.json
runs/supervisor/CAMPAIGN/journal.jsonl
runs/supervisor/CAMPAIGN/STOP
```

Here `CAMPAIGN` is the manifest's `campaign_id`; `STOP` is a stop-request sentinel.
The persisted overall deadline survives supervisor restart, and downtime counts
against it. Preserve the collection reserve. Resume does not reset the campaign
clock or grant another 28,800-second window.

Resume attaches to the same canonical harness plan with
`--execute --approved-plan-digest` and its original digest. The harness retains
its original batch identity, deadlines, completed tasks, live workers, and
receipts. Do not delete state or locks, change active digests, or invent new run
IDs to escape a failed or uncertain prior dispatch. Host reboot, identity
mismatch, unprovable dispatch, and missing/changed evidence require explicit
reconciliation before any new work.

If the existing host driver still owns the shared pool, the supervisor observes
and waits within the original campaign deadline. It records one wait event per
target/lock period and never launches another driver against that lock. STOP and
the collection-reserve boundary end this wait without resetting any budget.

A `STOP` request or supervisor `SIGINT`/`SIGTERM` asks the active harness driver
to hand off through `SIGINT`. The canonical harness leaves live workers under
their own existing deadlines and ownership locks. This is a driver handoff,
not proof that all workloads have stopped; collect actual retained task/process
state before declaring the host idle or assigning another owner.

## Local acceptance and current blockers

The authored acceptance file is
`actionmesh/research_math/tests/test_research_supervisor.py`. Local must set
`RESEARCH_AUTOPILOT_SKILL_DIR` to the actual canonical complete installed skill
directory and include that environment binding in the reviewed acceptance
scope. The value must agree with the skill source selected and pinned for the
test. Run acceptance through the installed `run_harness.py` under the repository
runbook; do not execute the test file directly as a shortcut. Retain the actual
plan, digest, logs, attempts, receipt, and source revision before reporting any
acceptance result.

The skill environment variable is mandatory for these checks; a missing skill
must fail acceptance rather than skip it. The acceptance plan must explicitly
stage `scripts/research_supervisor.py` and its source dependencies alongside the
test. A plan that stages only `actionmesh` and root-level modules omits this
entrypoint and cannot qualify it. The source loader exposes
`skill_digest(root)` and `load_harness(root, expected_digest)` to bind the actual
canonical source tree before loading the harness.

The following blockers remain explicit:

- **The current GPU STOP remains in force.** Future Local dispatch requires the
  user's explicit resume, independently accepted plan/scientific gates, and the
  per-invocation `--allow-gpu-after-user-resume` acknowledgement. Neither that flag
  nor a campaign digest proves permission or resolves current qualification
  blockers. No GPU workload was launched by this delivery.
- **Local software and runtime qualification are pending.** Authored checks and
  source review do not establish a passed test, a working deployment, or an
  accepted harness campaign.
- **Free-form AI repair is not implemented.** Only the finite, pinned repair
  children in the approved manifest can become eligible after their exact
  failure condition. New code or a changed scientific direction needs a new
  reviewable artifact and the existing admission process.
- **Scientific qualification is unchanged.** This wrapper does not qualify the
  current native baseline, trusted replay, candidate methods, or scientific
  outcomes, and does not advance research gates.

Follow `AGENTS.md`, `LOCAL_AGENT_RUNBOOK.md`, the current round handoff, and
`docs/research-math-20261006/CURRENT.json` at the exact delivered revision before
Local acceptance or operational use. Preserve historical and running evidence;
this document does not authorize a new GPU launch.

Acceptance checks for the default GPU STOP, explicit future acknowledgement,
non-persistence of that acknowledgement, and observation-only handling of live
GPU state are authored source, not executed acceptance results.

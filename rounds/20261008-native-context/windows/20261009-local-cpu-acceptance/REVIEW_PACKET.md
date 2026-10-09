# Local CPU software acceptance — 20261009-local-cpu-acceptance

## Result and scope

The common Linux CPU acceptance suite completed at source commit
`62416056bfaaf7a82ff0d1826f1ce4b2bb0bebb8`: **849 tests, 843 passed, 6 skipped,
0 failures, 0 errors**. Test time was 126.296 seconds; the harness reported
`completed` after 130.455 seconds. This is software acceptance on engineering
fixtures. It is not retained-native method acceptance, scientific admission,
ActionBench scoring, inference, or evidence of a candidate effect.

The six skipped checks require real current-version C01/C03/C04/C06/C07/C08
artifacts. They remain explicitly unexecuted. All 15 selected ideas have source
chains; native results and Local method verification remain **0/15**. GPU STOP
remains active. No GPU experiment or running project supervisor was installed.

Final run: `debug-r6-cpu-001`.

- Remote root: `/root/rivermind-data/4d-debug-r6`
- Harness plan: `plans/debug-r6-cpu-001/harness.json`
- Approved digest: `a9c8201fe7d75da5c3bb1eee73dfe0ffb939d9a0c794e99164bf1945b22ba282`
- Receipt: `runs/attempts/debug-r6-cpu-001/receipt.json`
- Receipt SHA-256: `0e6f14504349543b759764b70b824958af75a1b485e84820652b9866b448b50f`

See [RESULT.json](RESULT.json) for every actual run, raw archive locator, hash,
plan, receipt, attempt record, execution context and stdout/stderr. Earlier
failures and the 120-second interrupted attempt were preserved in full. Each
repair was a new committed source revision and a new single-use run identity;
no failed attempt was overwritten or automatically retried.

## Repairs

- Restored missing staged dependencies; corrected cross-root calibration paths,
  undefined references, producer/consumer archive ordering and C05 GPU-plan
  inventory. The GPU plan was only tested as a plan, never executed.
- Restored C01 baseline source validation and C14 candidate role identity.
- Corrected C10 shared 3x3 indexing and mirrored the exporter's exact frame-zero
  restoration in its validator; exact byte comparisons remain required.
- Preserved C05 producer namespaces and bounded model-inventory validation while
  excluding external model weights from downstream CPU artifact staging.
- Accelerated C06/C07 balanced entropic duals and C08 endpoint scaling with
  bounded sparse Newton-CG steps. Added a box-constrained dual step for C07
  partial transport. Objectives, support, parameters, capacity constraints and
  final residual/KKT thresholds are unchanged. Large native runtime remains
  unmeasured; prior timing cannot price these implementations.
- Corrected stale synthetic fixtures and diagnostic expectations. C13
  mathematical checks use its float64 solver; exact affine inputs are
  representable in float32. A separate rejection check retains the known
  quantization failure and prohibits a scoreable export. Export thresholds
  were not relaxed.
- Preserved supervisor integrity errors and made intentionally rehashed test
  mutations internally consistent. G01 tests independently exercise admission,
  generation identity and post-score review rejection.
- Rebound reviewed source hashes and sorted the C03/C07 reference inventories.
  Scientific design fields, roles, criteria, population, seeds, denominator and
  authorization requirements are unchanged. Historical reviews are retained.

Independent source reviews were performed by `review_repairs`,
`review_numerics` and `review_g01`. The last two G01 test corrections were also
reviewed separately; the accepted final test file SHA-256 is
`2750640118ec76457c1ffcbc94d0224537db7a53b48866e419b373650eda9bab`.
The reviewed G01 digest is
`2d53b764958f89cb5edcc3949b721ac9a7ca484e7ec8c65d8b07649f7edeeb16`.

## Environment and runtime update

The current host has a **Tesla T4**, UUID
`GPU-305023ac-8457-0ac4-0432-4ff19b30de46`, rather than the historical RTX 2080 Ti.
All acceptance plans reserved one CPU, 2048 MiB RAM and zero GPUs; the harness
hid GPU visibility. The final attempt ceiling was explicitly 300 seconds plus
60 seconds of outer-harness allowance, with one attempt and no automatic retry.

CPU environment: Python 3.12.13, NumPy 1.26.4, SciPy 1.17.1, trimesh 5.1.0,
safetensors 0.8.0; base Torch is 2.12.1+cu130. Dependencies were installed in an
isolated environment using the Tsinghua PyPI mirror. This does not establish a
qualified native ActionMesh runtime.

Research_Autopilot was updated to GitHub revision
`bd368630ef623efa16c6c28fab894d692f82ee4d` and its skill family installed under
`/root/rivermind-data/research-autopilot-bd36863/skills/`.
The updated `run_harness.py` SHA-256 is
`3b8849c6c718427468eca0548c9edf1db804ad6f27ffea55d8735735546d3d42`.
The original installed plugin and dirty user checkout were preserved.

## Remaining execution boundary

The allowed continuation is asset acquisition and CPU engineering admission.
Scientific execution still requires the runbook's baseline/scorer qualification,
source-derived family evidence, candidate Gate 0/IPCG, exact real inputs,
prospective numeric criteria, strict runtime closure and an expiring
single-attempt authorization for the actual GPU. Software acceptance does not
supply any of those missing records and does not resume GPU execution.

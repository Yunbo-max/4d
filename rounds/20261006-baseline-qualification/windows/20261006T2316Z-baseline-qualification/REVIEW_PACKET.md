# Baseline qualification setup review

Window: `20261006T2316Z-baseline-qualification`
Delivered base revision: `fc51ba996c5b69b50afc35544914490e93e9addb`
Host: `jupyter-55p0lroqabt855y1`
Execution mode: native Conda-backed project venv; Docker not used

## Result

The remote checkout, bounded Autopilot harness and original one-asset inputs are prepared. Software acceptance passed **127/127 tests**. The native environment inventory was captured and hash-verified. The three CPU control arms were built and the scoring request validated. **No official score or GPU scorer replay was run.** The required frozen native protocol, official-versus-faithful-harness parity sidecar and nonce-bound trusted replay evidence are absent, and the handoff names these as mandatory admission inputs. This packet therefore closes setup only; it does not qualify a baseline or advance a scientific gate.

Candidate validation designs/results remain `0/15`.

Supporting plans, runtime files, original source/report/receipt, three-arm
outputs and harness/native receipts/logs are packaged in
[`supporting-evidence.tar.gz`](supporting-evidence.tar.gz), SHA-256
`06ccb3de0223a4578d2da2c3fdf493cd73d9c55f66ca6b1141d9acdebeadc6b6`. Extract
it at the repository root to restore the `inputs/`, `plans/` and `runs/`
paths cited below. The archive omits GT and duplicated staged attempt
workspaces; the latter remain on the GPU host.

## Source and runtime repair

The actual ActionMesh inference environment is `/root/rivermind-data/actionmesh-repro/inference-env/bin/python`, a `system-site-packages` venv based on `/opt/conda`. Its effective versions are NumPy 1.26.4, Torch 2.12.1+cu130, SciPy 1.17.1, trimesh 5.1.0 and PyTorch3D 0.7.9. The base Conda inventory also has NumPy 2.4.4, which is shadowed by the venv's NumPy 1.26.4 on `sys.path`.

The delivered collector initially rejected this real environment because it treated both visible NumPy metadata records as an unresolved conflict and only searched `sys.prefix` for Conda metadata. Two failed attempts were retained:

- `baseline-native-runtime-001`: the bare base interpreter lacked the project venv's SciPy, trimesh and PyTorch3D metadata.
- `baseline-native-runtime-002`: adding the venv with `PYTHONPATH` exposed both NumPy versions and triggered the collector's ambiguity rejection.

The applied patch makes the collector select distributions by Python `sys.path` order, keeps rejecting version conflicts with equal or unknown precedence, resolves the Conda base prefix separately from the Python prefix, and records both prefixes. The regression tests reproduced the failure before the implementation change. Patch: [runtime-overlay-fix.patch](runtime-overlay-fix.patch), SHA-256 `66d46f97f8fa4d4e4197ed501efde6b0ffb319118b629096195478481c07a10a`.

The current collector and tests were accepted at the delivered base plus this patch in harness run `control-scoring-integrity-software-003`:

- Outer plan digest: `4a4d6e4564802e4e63e3db5e8c5f7a70b4b936251dd53aa420a4657478bd079b`
- Native plan digest: `22ccf90112ea084d0fcfa79d572b6ad2255a37eeb1c558f565adb79ac8342ae8`
- State: `runs/harness/control-scoring-integrity-software-003/state.json`
- Attempt receipt SHA-256: `5c13039637536e1c7ac75d6c4c6ff66f67bfb2f16c037c50d62a753ca19a7c4a`
- Test log: `runs/attempts/control-scoring-integrity-software-003/`
- Result: 127 tests, 0 failures, CPU-only, under 3 seconds.

The prior 125-test run on the unmodified source and the first 127-test run after the repair are also retained in their corresponding `control-scoring-integrity-software-001` and `-002` plan/attempt directories.

Runtime capture succeeded in `baseline-native-runtime-003` through the outer harness (one CPU, 1,024 MiB, zero GPUs):

- Outer plan digest: `d54a6c5f17c6c305990d7a09e18f214586a241a17d7ba0d3bcb1d8e007e4be93`
- Native plan digest: `ba704ca231f1334f1b629aefdb8b805aec8e1118231a57c7b45a2bcb0e8e9e3b`
- Attempt receipt SHA-256: `3e5799d42337d764a5fbc5a2cd120c70b2648a8f0a51d577d9c2abd70d455ec1`
- `inputs/native-runtime/environment.json` SHA-256: `40326cc1e8ff263858f17c9a1bbd851215e0d3a523ffad9ebf906827b58233d3`
- `inputs/native-runtime/dependencies.json` SHA-256: `9e10d71f787adaa4e68af9084f51db1c04409df540973b0f1b658b9f14320a25`

The capture explicitly leaves `gpu_identity_verified=false` and `native_contract_qualified=false`; it is installed-package metadata, not CUDA or scorer qualification. `docs/research-math-20261006/NATIVE_RUNTIME_CAPTURE.md` now documents the Conda-backed venv case.

## Original asset and controls

The development asset is UID `000-048_45e57349f062416aaf11f2c31587da16`, from ActionBench revision `2796071cbe6248422fcbeab3101fa9f9886cb7b9`. The W0 protocol hash is `fa79ec3a2840fa9a4a71cb0ebfd52b138db810d4608756c798463dbc8dbc1885`; the original generation receipt is retained at `inputs/original-case/generation.receipt.json` (SHA-256 `506d401f33833344b1472a4d674f767ab9548f988b00b5e1e4063aa47a7aab8b`). The receipt binds the copied report and sequence; its fingerprint records the generation protocol. The generation source contains 16 frames, 19,936 vertices and 39,956 faces. All 16 images, camera metadata and GT were present and referenced by the W0 protocol. The six ActionBench evaluator source hashes in the clean checkout match the source closure recorded by that protocol.

Verified source hashes:

- `inputs/original-case/sequence.npz`: `0b4edd55f8b49ca8d821eb99606d9ca8e556c0f48883f88e23cd272cc867dad1`
- `inputs/original-case/report.json`: `af8fb23b4d19fa0b8ca78f77a8ada5957830053ad14347fe29dcbe369e7fdfdd`
- Released GT `surfaces.npy`: `25881f0d7a9f41578f77ba6be70f810ddcbcf4236a6834713ea197ea2916823e` (38,400,128 bytes; retained on the GPU host and intentionally not committed here).

The ordinary-controls plan ran through the harness with sigma 1.0, one CPU, 2,048 MiB and no GPU:

- Outer plan digest: `79becdc5e02a251b1c12228199bf48fedbc459f8fe83612854b6d344effb1848`
- Native plan digest: `cf76ad40493064fd651a7ffe8d5587afa075bd4fd25b4ee538016e28cd3220b7`
- Attempt receipt SHA-256: `3d2144f44e41c546ce38a74b96c05bb49ef9a07325fc392c300cc4e4d5f3f601`
- Native/world-Gaussian/body-Gaussian outputs all report `completed`.
- Manifest SHA-256: `738b5a2781fb987e2a9095e1aea2830fd369d576965002e26b17356ba2f0510d`
- Controls summary SHA-256: `eb40246841c2180a740707129b4f6ec09964ac86adfc94e495d2158735ed6d12`

The request was generated and strictly revalidated, but no scorer plan was admitted:

- `plans/control-scoring-request-000-048.json`
- Request digest: `49a27caf9f160d364b1e64e246aa1d9fb345f9185c7f4d54204c51713258c3e8`
- Request file SHA-256: `99f0c2f1526cb98584d6e37ff7fa45f71ed804f13366685ed44c6e1bcf493d22`
- `native_contract_qualified=false`; `candidate_methods_tested=false`.

## Host and resource observations

At the final host sample the GPU was an NVIDIA GeForce RTX 2080 Ti, driver 580.119.02, UUID `GPU-b544b42e-15d3-c9c8-1bdb-4c339775a740`, 22,528 MiB total, 22,002 MiB free, 0% utilization, with no compute processes listed. No GPU-scoring seconds were consumed.

The shared data volume was 49 GB with approximately 2.4 GB free (96% used). No dataset/model download or package installation was needed; existing assets and dependencies were present, so no mirror transfer was necessary. No old project data was deleted. Current free space was adequate for this one-asset preparation.

## Required next inputs

Before native scoring, Local still needs to provide and review a source-backed frozen qualification protocol, official-versus-faithful-harness parity evidence, nonce-bound trusted-host replay evidence, and the prospectively justified per-asset decision rules. The required protocol was not present in the checkout, and no substitute was created. After those inputs pass review, build a new scoring plan, recheck device/process and disk state, then run the one development asset and preserve all raw passes. This setup run does not establish full-unit timing, peak scorer VRAM, broad efficacy or candidate-method results.

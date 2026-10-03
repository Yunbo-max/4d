# Implementation ledger — plan: docs/research-ten/implementation-plan.md

2026-10-03: user provided ten-candidate spec and requested all code. Isolated branch `research/ten-methods-20261003` based on `fc501b6`. No prior ten-method implementation reused as if it already fulfilled this spec.

Interface review: workers share only `demo() -> JSON-safe dict`; numerical inputs and model callbacks documented per module. Root owns package/registry/CLI. Official model files remain unmodified. Tests must distinguish software correctness from scientific evidence.

Task root/H6: implementation and 11 behavioral tests pass; CLI 2 tests pass. Local dependencies: existing NumPy only. Baseline checkout unittest discovery found no tests (exit5); it did not establish a passing preexisting suite.
Native integration runner prepared for frozen decoder + separate 2-step three-branch guidance rollouts, pending adapter completion.

Integration first controls: all10 demos executed and emitted finite JSON. First fullsuite59tests failed17 due tracking test import paths; fixedbyowner, later fullsuite pending. This initial result is retained.
Review finding H6: huge finite weight sum overflow caused tiedscores/wrongselection. Added failing regression and stable max-normalization; all13H6 tests now pass.
Ruling H10: generated windowstarts are predictions, not immutable inputanchors. Earlier reverse-loop reconditioned on unchanged forward overlap and failed to transmit future-window evidence. Requested true reverse rollout starting at terminal reliable pseudo-boundary, then revised overlap propagation; preserve only explicit original inputanchors. This changes constructed protocol; keep firstcontrols as superseded developmental evidence.

Final: all10 modules/APIs/docs delivered. H4/H7 memorycaps and H8 integeroverflow regressions resolved. LocalPyTorch installed in /Users/yunbo/Documents/4d/.venv-research at explicit user request.71/71 local and71/71 remote pass,0skips.10/10 controls local/remote. NativeH2/H9 GPU170.31seconds,10229MiB peak; trainedweights unchanged. All Python delivery bytes match remote-testedcode-v2; native adapter subset matchesnative-code-v1. Fullnaturalbenchmark notrun.

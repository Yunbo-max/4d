# FP16 low-RAM complete-unit result

Harness: `complete-lowram-r7`; plan digest `a94aa69f8605266587f56f0977740001bbd21971463bb5b78d4e20eaf2bf574b`. Attempt `complete-fp16-lowram-v1-three-arm-unit-a1-198b8d1db25543eb968e2189bc7d9739`. Harness receipt SHA-256 `ec72d1aa2301d73cbc62eb61bda4e588ebc7ebbb949c17f542b7dc3e21f43909`; result SHA-256 `1cf6aea2aaaf39c9e8f1b535bd60e7f0fcf4b4f9c1ecaf72239878cd5ed13b10`.

One lexically frozen ActionBench UID; pinned current-public-release source/model/data. FP16 + official low-RAM preset. Generation seed 42, all 16 input frames, stage steps 100/30, face target 40,000, guidance 7.5, anchor 0. Official scorer seed 44, 100,000 Chamfer points, 10,000 ICP points, 24 rotations and 200 ICP iterations. All three official rows report success and 16 frames. The CPU KNN-backward compatibility patch affects only backend placement; prior parity evidence for its exact contract is preserved separately.

| Arm | cd_3d ↓ | cd_4d ↓ | cd_motion ↓ |
|---|---:|---:|---:|
| native | 0.038848036 | 0.075961557 | 0.168381736 |
| world_gaussian | 0.038530298 | 0.076854477 | 0.171246663 |
| body_gaussian | 0.038372575 | 0.075834003 | 0.169197276 |

The native arm has the lowest cd_motion. World Gaussian worsens cd_4d and cd_motion. Body Gaussian improves cd_3d and cd_4d slightly but worsens cd_motion by 0.000815541 (about 0.48%). This is one engineering UID, descriptive only; no confidence interval, method effect, or baseline qualification follows.

Complete unit wall: 1,330.466 s (22 min 10 s); generation 792.536 s; all three official scorings together 458.266 s; control construction 0.739 s; export validation 1.254 s. GPU sampled peak 10,255 MiB at 1 s interval (not exact peak). Host sampled RSS peak 11.54 GB, free disk minimum 24.77 GB. 121 receipt-bound output files passed completion/hash collection. GPU is now free.

This runtime is explicitly different from default BF16. It is not an exact published-row reproduction and cannot qualify that row. The single UID is not a training-free method result. Current native-method ledger remains 0/15.

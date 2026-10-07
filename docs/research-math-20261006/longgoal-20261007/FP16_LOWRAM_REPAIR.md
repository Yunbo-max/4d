# Explicit FP16 low-RAM repair v1

Parent: complete-default-r6, receipt SHA-256 7b4223a3fdcf4a738c1cec646ecb8f8172ece850a46aa8936aa96d8626cd62f9. Its default BF16, non-low-RAM run completed Stage 0 and failed at Stage I 0/30 in scaled_dot_product_attention, requesting 128.13 GiB. Observed GPU peak was 20,673 MiB. Full logs and receipts are retained in default-oom-r6/. No mesh or score succeeded.

Source inspection: the pinned official low-RAM YAML inherits the standard configuration and only enables clear_autocast and split_cfg_batch. The entrypoint additionally enables lazy loading. FP16 is an official CLI precision option. The attention processor checks global enable flags, not whether the hardware can actually use an efficient kernel. BF16 causing a Turing-incompatible efficient kernel and math fallback is a hypothesis, not a measured backend diagnosis.

Explicit child configuration: --dtype float16 --low_ram, no --fast. Keep UID, 16 frames, seed 42, 100/30 denoising steps, topology target, guidance, all three arms and full official evaluator settings (seed 44). This combined compatibility profile is an operational repair, not a causal ablation of dtype versus memory policy. Original contract, manifest and failure remain immutable. Do not call its values numerically equivalent to default or published-row results.

Plan: require software acceptance, create a separately bound child unit manifest, and run one complete three-arm unit through the existing supervised harness. Preserve failures without automatic retry. A successful unit only measures this specific runtime; the 15 candidate results remain pending.

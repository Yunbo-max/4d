"""Exploratory, frozen-model endpoint-statistic steering (2026-10-02).

correction returns (delta_velocity, logs); add delta_velocity to v.  The caller
must enable this only at step_index >= 20.  Only frame slots 1: are changed.
For additive ActionMesh flow, s=t/1000 and xhat=z+s*stopgrad(v).  This is a
constant-velocity extrapolation, not a verified clean-sample estimator.
Conditioning disagreement defines a tolerance, not calibrated uncertainty.
No denoiser graph, geometry decoder, or parameter training is used here.
"""

import json
import torch


def _rms(x):
    return x.square().mean(dim=(-2, -1), keepdim=True).sqrt()


def _statistic(x, mode):
    if mode == "moment_energy":
        return x.var(dim=-2, unbiased=False)
    if mode == "rms_energy":
        return x.square().mean(dim=(-2, -1)).clamp_min(1e-12).sqrt().unsqueeze(-1)
    raise ValueError("mode must be moment_energy or rms_energy")


def _objective_parts(anchor, x_anchor, x_cond, mode):
    target = _statistic(anchor, mode)
    # Per-channel scale has a global floor; a constant/zero anchor stays finite.
    global_scale = target.abs().mean(dim=-1, keepdim=True).clamp_min(1e-4)
    scale = torch.maximum(target.abs(), 0.1 * global_scale).clamp_min(1e-4)
    disagreement = (_statistic(x_cond, mode) - _statistic(x_anchor, mode)).abs()
    band = (0.1 * scale + disagreement).detach()
    return target.detach(), scale.detach(), band


def _energy(x, mode, target, scale, band):
    residual = ((_statistic(x, mode) - target).abs() - band).clamp_min(0)
    return (residual[:, 1:] / scale).square().mean()


def correction(z, v, v_anchor, v_cond, anchor_latent, s, mode):
    """Return delta_v (same dtype/device as z) and JSON-compatible diagnostics.

    z/v/v_anchor/v_cond: [B,T,N,D].  v_anchor and v_cond are unaggregated [01]
    and [11] branch velocities.  Clone them before aggregate_cfg's in-place +=.
    anchor_latent: [B,N,D], [B,1,N,D], or [B,T,N,D] (first frame is used).
    moment_energy uses centered per-channel token variance. rms_energy uses
    uncentered scalar RMS over tokens/channels, with identical band/cap rules.

    Cheap backtracking probes E(xhat+s*delta_v) with velocity held fixed; this
    diagnostic is not the energy after a new model evaluation.  The caller's
    actual Euler update is z + delta_s*(v+delta_v).
    """
    if mode not in ("moment_energy", "rms_energy"):
        raise ValueError("unknown energy mode")
    if z.ndim != 4 or any(a.shape != z.shape for a in (v, v_anchor, v_cond)):
        raise ValueError("z and all velocities must share [B,T,N,D] shape")
    if anchor_latent.ndim == 3:
        anchor_latent = anchor_latent.unsqueeze(1)
    if (anchor_latent.ndim != 4 or anchor_latent.shape[0] != z.shape[0]
            or anchor_latent.shape[-2:] != z.shape[-2:]):
        raise ValueError("anchor shape must match [B,1|T,N,D]")
    sf = float(s)
    if not 0.0 <= sf <= 1.001:
        raise ValueError("s must be normalized diffusion time t/1000")
    logs = dict(mode=mode, s=sf, energy_before=None, energy_probed_after=None,
                grad_rms=0.0, grad_norm=0.0, correction_rms=0.0,
                correction_norm=0.0, max_relative_correction_rms=0.0,
                active_fraction=0.0, backtracks=0, finite=True)
    # Inner context overrides the native pipeline's inference/no_grad contexts.
    with torch.inference_mode(False), torch.enable_grad():
        zf, vf, va, vc = [a.detach().to(dtype=torch.float32).clone()
                          for a in (z, v, v_anchor, v_cond)]
        anchor = anchor_latent[:, :1].detach().to(device=z.device,
                                                  dtype=torch.float32).clone()
        zero = torch.zeros_like(zf)
        if not all(bool(torch.isfinite(a).all()) for a in (zf, vf, va, vc, anchor)):
            logs.update(finite=False, reason="nonfinite_input")
            return zero.to(z.dtype), logs
        if z.shape[1] < 2 or sf == 0:
            logs.update(energy_before=0.0, energy_probed_after=0.0,
                        reason="no_unobserved_frames_or_zero_time")
            return zero.to(z.dtype), logs
        x = (zf + sf * vf).detach().clone().requires_grad_(True)
        target, scale, band = _objective_parts(anchor, zf + sf * va,
                                               zf + sf * vc, mode)
        before = _energy(x, mode, target, scale, band)
        grad, = torch.autograd.grad(before, x, create_graph=False)
        grad = grad.detach()
        grad[:, 0] = 0
        if not bool(torch.isfinite(before) & torch.isfinite(grad).all()):
            logs.update(finite=False, reason="nonfinite_energy_or_gradient")
            return zero.to(z.dtype), logs
        grad_rms = _rms(grad)
        cap = 0.1 * _rms(vf)
        delta = -grad * (cap / grad_rms.clamp_min(1e-12))
        delta[:, 0] = 0
        with torch.no_grad():
            active = ((_statistic(x, mode) - target).abs() > band)[:, 1:]
            after = _energy(x + sf * delta, mode, target, scale, band)
            for _ in range(12):
                if bool(torch.isfinite(after)) and float(after) <= float(before):
                    break
                delta.mul_(0.5)
                logs["backtracks"] += 1
                after = _energy(x + sf * delta, mode, target, scale, band)
            if not bool(torch.isfinite(after)) or float(after) > float(before):
                delta.zero_()
                after = before.detach()
                logs["reason"] = "no_descent_probe"
            # Margin covers FP16/BF16 rounding while retaining the 0.1 bound.
            delta.mul_(0.99)
            out = delta.to(z.dtype)
            after = _energy(x + sf * out.float(), mode, target, scale, band)
            ratio = _rms(out.float()) / _rms(vf).clamp_min(1e-12)
            logs.update(energy_before=float(before), energy_probed_after=float(after),
                        grad_rms=float(grad.square().mean().sqrt()),
                        grad_norm=float(grad.norm()),
                        correction_rms=float(out.float().square().mean().sqrt()),
                        correction_norm=float(out.float().norm()),
                        max_relative_correction_rms=float(ratio.max()),
                        active_fraction=float(active.float().mean()),
                        finite=bool(torch.isfinite(out).all() & torch.isfinite(after)))
        return out.detach(), logs


def cpu_canary():
    """CPU-only finite-difference, descent, permutation, and safety canaries."""
    generator = torch.Generator(device="cpu").manual_seed(1729)
    z = 1.8 * torch.randn(1, 4, 32, 8, generator=generator, device="cpu")
    v = 0.2 * torch.randn(z.shape, generator=generator, device="cpu")
    anchor = 0.8 * torch.randn(1, 1, 32, 8, generator=generator, device="cpu")
    results = {}
    for mode in ("moment_energy", "rms_energy"):
        x = (z + 0.2 * v).double().requires_grad_(True)
        target, scale, band = _objective_parts(anchor.double(), x.detach(),
                                               x.detach(), mode)
        energy = _energy(x, mode, target, scale, band)
        grad, = torch.autograd.grad(energy, x)
        direction = torch.randn(z.shape, generator=generator, device="cpu").double()
        direction[:, 0] = 0
        direction /= direction.norm()
        h = 1e-5
        fd = (_energy(x.detach()+h*direction, mode, target, scale, band)
              - _energy(x.detach()-h*direction, mode, target, scale, band)) / (2*h)
        analytic = (grad * direction).sum()
        relerr = float((fd-analytic).abs() / analytic.abs().clamp_min(1e-8))
        assert relerr < 1e-5, (mode, "directional_derivative", relerr)
        with torch.inference_mode():
            delta, logs = correction(z, v, v, v, anchor, 0.2, mode)
        assert logs["finite"] and not delta.requires_grad
        assert torch.count_nonzero(delta[:, 0]) == 0
        assert logs["max_relative_correction_rms"] <= 0.10001
        assert logs["energy_probed_after"] < logs["energy_before"]
        perm = torch.randperm(z.shape[-2], generator=generator)
        dp, lp = correction(z[:, :, perm], v[:, :, perm], v[:, :, perm],
                            v[:, :, perm], anchor[:, :, perm], 0.2, mode)
        assert torch.allclose(dp, delta[:, :, perm], rtol=1e-4, atol=1e-6)
        dz, lz = correction(z, v, v, v, torch.zeros_like(anchor), 0.2, mode)
        assert lz["finite"] and torch.isfinite(dz).all()
        results[mode] = dict(directional_relative_error=relerr, **logs)
    return results


if __name__ == "__main__":
    print(json.dumps(cpu_canary(), indent=2, allow_nan=False))

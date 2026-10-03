"""Edit-direction utility for fixed local/global row-vector rotations.

No fitting, model calls, selection, or GPU initialization occurs in this module.
Positive ``utility`` or ``heldout`` favors fallback from local to global.
The caller owns disjoint fit/held-out support and rotation validity.
"""

from __future__ import annotations

import argparse
import json
import math

import torch


@torch.no_grad()
def compute(observed, d0, query_indices, heldout_indices, local_rot, global_rot, scale):
    """Return action-comparison scores, each with shape [T,Q].

    Inputs: observed [T,N,3], d0 [Q,3], integer query_indices [Q], integer
    heldout_indices [Q,K], local_rot [T,Q,3,3], global_rot [T,3,3], and
    positive fixed scene scale. Rotations act on row vectors as ``d0 @ R``.

    ``utility`` is J_local - J_global, with J the mean squared difference
    between anchor and transported edit/spoke dot products, normalized by
    ||d0||^2 * scale^2. It is evaluated with unit edit vectors for numerical
    stability; zero edits yield zero utility. ``heldout`` is local minus
    global mean squared 3D spoke-registration error divided by scale^2.

    ``axis_sum`` separately sums the directional contrasts for unit x/y/z
    edits; for proper rotations it equals ``heldout``. ``axis_averaged`` is
    one third of that sum, the direction-agnostic ablation. Its sign yields
    exactly the same selector as ``heldout``, up to roundoff.

    Output dtype is float64 if any floating input is float64, else float32.
    Floating inputs must share observed's device. No inputs are mutated.
    This function cannot verify held-out identities were excluded from fits.
    """
    arrays = (observed, d0, local_rot, global_rot)
    if any(not isinstance(x, torch.Tensor) or not x.is_floating_point() for x in arrays):
        raise TypeError("geometry, edit and rotations must be floating-point tensors")
    if observed.ndim != 3 or observed.shape[-1] != 3 or min(observed.shape[:2]) < 1:
        raise ValueError("observed must have nonempty shape [T,N,3]")
    if d0.ndim != 2 or d0.shape[-1] != 3 or d0.shape[0] < 1:
        raise ValueError("d0 must have nonempty shape [Q,3]")
    frames, vertices, _ = observed.shape
    queries = d0.shape[0]
    if local_rot.shape != (frames, queries, 3, 3) or global_rot.shape != (frames, 3, 3):
        raise ValueError("rotation shapes must be [T,Q,3,3] and [T,3,3]")
    scene_scale = float(scale)
    if not math.isfinite(scene_scale) or scene_scale <= 0:
        raise ValueError("scale must be positive and finite")
    device = observed.device
    if any(x.device != device for x in arrays):
        raise ValueError("floating inputs must share the observed device")
    query = torch.as_tensor(query_indices, device=device)
    held = torch.as_tensor(heldout_indices, device=device)
    integer_types = (torch.int8, torch.int16, torch.int32, torch.int64, torch.uint8)
    if query.dtype not in integer_types or held.dtype not in integer_types:
        raise TypeError("query and held-out indices must have integer dtype")
    if query.shape != (queries,) or held.ndim != 2 or held.shape[0] != queries or held.shape[1] < 1:
        raise ValueError("index shapes must be [Q] and nonempty [Q,K]")
    if bool(((query < 0) | (query >= vertices)).any()) or bool(((held < 0) | (held >= vertices)).any()):
        raise ValueError("vertex index outside observed trajectory")
    query, held = query.long(), held.long()
    dtype = torch.float64 if any(x.dtype == torch.float64 for x in arrays) else torch.float32
    v, edit, local, glob = [x.to(dtype=dtype) for x in arrays]
    if not all(bool(torch.isfinite(x).all()) for x in (v, edit, local, glob)):
        raise ValueError("geometry, edit and rotations must be finite")

    # Normalizing spokes first is equivalent to dividing every squared score
    # by scale^2, avoiding a potentially overflowing squared scale scalar.
    spokes = (v[:, held] - v[:, query, None]) / scene_scale
    anchor_spokes = spokes[0]
    magnitude = torch.linalg.vector_norm(edit, dim=-1, keepdim=True)
    unit = edit / torch.where(magnitude > 0, magnitude, torch.ones_like(magnitude))
    local_edit = torch.einsum("qi,tqij->tqj", unit, local)
    global_edit = torch.einsum("qi,tij->tqj", unit, glob)
    anchor_response = (anchor_spokes * unit[:, None]).sum(-1)[None]
    local_utility = ((spokes * local_edit[:, :, None]).sum(-1) - anchor_response).square().mean(-1)
    global_utility = ((spokes * global_edit[:, :, None]).sum(-1) - anchor_response).square().mean(-1)

    local_prediction = torch.einsum("qki,tqij->tqkj", anchor_spokes, local)
    global_prediction = torch.einsum("qki,tij->tqkj", anchor_spokes, glob)
    local_full = (local_prediction - spokes).square().sum(-1).mean(-1)
    global_full = (global_prediction - spokes).square().sum(-1).mean(-1)
    heldout = local_full - global_full

    # Independently evaluate three actual edit directions. Each row of R is
    # the transported Cartesian basis vector, so these contractions form
    # dot-product responses rather than reusing the full residual above.
    local_axis_response = torch.einsum("tqkc,tqac->tqka", spokes, local)
    global_axis_response = torch.einsum("tqkc,tac->tqka", spokes, glob)
    local_axis_score = (local_axis_response - anchor_spokes[None]).square().mean(-2)
    global_axis_score = (global_axis_response - anchor_spokes[None]).square().mean(-2)
    axis_sum = (local_axis_score - global_axis_score).sum(-1)
    result = {"utility": local_utility - global_utility, "heldout": heldout,
              "axis_sum": axis_sum, "axis_averaged": axis_sum / 3,
              "axis_identity_error": axis_sum - heldout,
              "local_utility": local_utility, "global_utility": global_utility}
    if not all(bool(torch.isfinite(x).all()) for x in result.values()):
        raise ValueError("nonfinite utility score; check input magnitudes and scale")
    return result


def self_test():
    """CPU checks; analytic fixtures catch rotation, sign, and scale errors."""
    dtype, device = torch.float64, torch.device("cpu")
    identity = torch.eye(3, dtype=dtype, device=device)
    quarter_turn = torch.tensor(
        [[0., 1., 0.], [-1., 0., 0.], [0., 0., 1.]], dtype=dtype, device=device
    )
    points = torch.tensor(
        [[0., 0., 0.], [1., 0., 0.], [-1., 0., 0.],
         [0., 1., 0.], [0., -1., 0.], [0., 0., 1.], [0., 0., -1.]],
        dtype=dtype, device=device,
    )
    observed = torch.stack((points, points @ quarter_turn))
    before = observed.clone()
    query = torch.tensor([0], device=device)
    heldout = torch.tensor([[1, 2, 3, 4, 5, 6]], device=device)
    local = torch.stack((identity, quarter_turn))[:, None]
    glob = identity.expand(2, 3, 3).clone()
    edit_x = torch.tensor([[0.2, 0., 0.]], dtype=dtype, device=device)
    out = compute(observed, edit_x, query, heldout, local, glob, 2.)
    # Four mismatched unit spokes contribute full residual 4/3, divided by 4.
    torch.testing.assert_close(out["heldout"], torch.tensor([[0.], [-1./3.]], dtype=dtype))
    # Unit x edit has squared displacement error 2, isotropic spoke factor 1/3.
    torch.testing.assert_close(out["utility"], torch.tensor([[0.], [-1./6.]], dtype=dtype))
    torch.testing.assert_close(out["axis_sum"], out["heldout"], atol=1e-14, rtol=1e-14)
    torch.testing.assert_close(3 * out["axis_averaged"], out["axis_sum"])

    edit_z = torch.tensor([[0., 0., 0.2]], dtype=dtype, device=device)
    invariant = compute(observed, edit_z, query, heldout, local, glob, 2.)
    torch.testing.assert_close(invariant["utility"], torch.zeros(2, 1, dtype=dtype))
    torch.testing.assert_close(invariant["heldout"], out["heldout"])
    zero = compute(observed, torch.zeros_like(edit_x), query, heldout, local, glob, 2.)
    torch.testing.assert_close(zero["utility"], torch.zeros(2, 1, dtype=dtype))
    for scaled_points, scaled_edit, scaled_scale in (
        (observed, 7 * edit_x, 2.), (3 * observed, 3 * edit_x, 6.)
    ):
        scaled = compute(scaled_points, scaled_edit, query, heldout, local, glob, scaled_scale)
        for key in ("utility", "heldout", "axis_sum", "axis_averaged"):
            torch.testing.assert_close(scaled[key], out[key])
    torch.testing.assert_close(observed, before)

    # Asymmetric, nonrigid observations exercise independent contractions and
    # literal antithetic squared-distance probes, rather than the same formula.
    generator = torch.Generator(device="cpu").manual_seed(20261002)
    random_observed = torch.randn(3, 11, 3, generator=generator, dtype=dtype, device=device)
    random_edit = torch.randn(2, 3, generator=generator, dtype=dtype, device=device)
    q = torch.tensor([0, 1], device=device)
    h = torch.tensor([[2, 3, 4, 5], [6, 7, 8, 9]], device=device)
    rotations, _ = torch.linalg.qr(torch.randn(3, 2, 3, 3, generator=generator, dtype=dtype, device=device))
    rotations = rotations.clone()
    rotations[..., :, -1] *= torch.linalg.det(rotations)[..., None]
    global_random = rotations[:, 0].clone()
    random_out = compute(random_observed, random_edit, q, h, rotations, global_random, 3.)
    et = random_observed[:, h] - random_observed[:, q, None]
    unit_edit = random_edit / torch.linalg.vector_norm(random_edit, dim=-1, keepdim=True)
    initial = unit_edit[None, :, None]
    response0 = ((et[:1] + initial).square().sum(-1)
                 - (et[:1] - initial).square().sum(-1)) / 4
    probe_scores = []
    for rotated_edit in (
        torch.einsum("qi,tqij->tqj", unit_edit, rotations),
        torch.einsum("qi,tij->tqj", unit_edit, global_random),
    ):
        trial = rotated_edit[:, :, None]
        response = ((et + trial).square().sum(-1) - (et - trial).square().sum(-1)) / 4
        probe_scores.append((response - response0).square().mean(-1) / 9)
    torch.testing.assert_close(random_out["utility"], probe_scores[0] - probe_scores[1],
                               atol=1e-13, rtol=1e-13)
    torch.testing.assert_close(random_out["axis_sum"], random_out["heldout"],
                               atol=1e-13, rtol=1e-13)
    identical = compute(random_observed, random_edit, q, h,
                        global_random[:, None].expand(-1, 2, -1, -1), global_random, 3.)
    for key in ("utility", "heldout", "axis_sum"):
        torch.testing.assert_close(identical[key], torch.zeros(3, 2, dtype=dtype),
                                   atol=1e-14, rtol=0)
    try:
        compute(observed, edit_x, query, heldout, local, glob, 0.)
    except ValueError:
        pass
    else:
        raise AssertionError("zero scale must be rejected")
    return {"status": "passed", "device": "cpu", "torch_version": torch.__version__,
            "checks": ["analytic_row_rotation_and_sign", "direction_dependence",
                       "axis_sum_identity", "antithetic_distance_response",
                       "edit_amplitude_and_coordinate_scale_invariance",
                       "zero_edit", "identical_candidates", "input_not_mutated", "invalid_scale"],
            "max_axis_identity_error": float(random_out["axis_identity_error"].abs().max())}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--self-test", action="store_true", help="run CPU-only analytic checks")
    args = parser.parse_args()
    if not args.self_test:
        parser.error("specify --self-test; the compute API is imported by the experiment harness")
    print(json.dumps(self_test(), indent=2, allow_nan=False))

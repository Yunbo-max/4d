"""Finite algebra witnesses only; no benchmark, mesh, inference or native scorer."""
import json
import math
import numpy as np


def projection_witnesses():
    rng = np.random.default_rng(20261006)
    errors = []
    for _ in range(40):
        a = rng.normal(size=(9, 9))
        w = a.T @ a + np.eye(9)
        c0 = rng.normal(size=(3, 9))
        c = np.vstack([c0, c0[0]])  # Deliberately redundant protection row.
        d = rng.normal(size=9)
        winv_ct = np.linalg.solve(w, c.T)
        p = d - winv_ct @ np.linalg.pinv(c @ winv_ct) @ c @ d
        r = d - p
        d2, p2, r2 = (float(x @ w @ x) for x in (d, p, r))
        rho = r2 / d2
        s = math.sqrt(max(0., 1. - rho))
        q = lambda x: .5 * x @ w @ x - x @ w @ d
        errors.extend([np.linalg.norm(c @ p), abs(p @ w @ r),
                       abs(d2 - p2 - r2), abs(-q(p) - .5 * p2),
                       abs(q(p) - q(d) - .5 * r2)])
        assert -q(s * d) + 1e-9 >= -q(p)
        assert abs((s * d) @ w @ (s * d) - p2) < 1e-8
    assert max(errors) < 1e-8
    return {"finite_random_instances": 40, "max_identity_residual": max(errors),
            "redundant_rows_checked": True, "equal_norm_scalar_gain_inequality": True}


def identifiability_witnesses():
    values = []
    for theta in (.1, .3, .8, 1.2):
        u = np.array([[1.], [0.], [0.]])
        v = np.array([[math.cos(theta)], [math.sin(theta)], [0.]])
        m = np.eye(3) - u @ u.T
        s = v.T @ m @ v
        kappa = float(s[0, 0])
        gain = float(np.linalg.norm(np.linalg.pinv(m @ v) @ m, 2))
        assert abs(kappa - math.sin(theta) ** 2) < 1e-12
        assert abs(gain - 1. / math.sqrt(kappa)) < 1e-10
        values.append({"angle_radians": theta, "kappa": kappa, "noise_gain": gain})
    p = np.array([[1.], [0.]])
    b = p.copy()
    assert np.array_equal(p * 2. + b * 3., p * 1. + b * 4.)
    gram = b.T @ (np.eye(2) - p @ p.T) @ b
    assert float(gram[0, 0]) == 0.
    assert float((gram + .5 * np.eye(1))[0, 0]) > 0.
    return {"principal_angle_instances": values,
            "identical_prediction_different_phase_amplitude": True,
            "ridge_invertible_despite_unidentifiable_data": True}


if __name__ == "__main__":
    left, right = math.tanh(.5), math.tanh(1.) / 2.
    assert abs(left - right) > .05
    theta, radius = .1, 2.
    chord = 2. * radius * math.sin(theta)
    deficit = radius * (1. - math.cos(theta))
    assert abs(deficit - (radius - math.sqrt(radius**2 - chord**2/4.))) < 1e-12
    print(json.dumps({"kind": "finite-mathematical-witnesses", "numpy_version": np.__version__,
                      "projection": projection_witnesses(),
                      "phase_amplitude": identifiability_witnesses(),
                      "nonlinear_mean": {"decode_mean": left, "mean_decode": right},
                      "localized_surface_mean": {"radius": radius, "angle": theta,
                                                 "chord": chord, "radial_deficit": deficit},
                      "scope": "Finite arithmetic checks accompany written proofs; not a general theorem verifier or scientific benchmark evaluation.",
                      "native_scorer_executed": False, "candidate_method_implemented": False},
                     indent=2, allow_nan=False))

"""Finite mathematical witnesses only: no dataset, mesh, scorer or native evaluation."""
from fractions import Fraction as F
import json
import math

law = [(F(9, 20), F(0), F(1)), (F(9, 20), F(0), F(-1)),
       (F(1, 20), F(10), F(5)), (F(1, 20), F(-10), F(-5))]
assert sum(p for p, e, r in law) == 1
assert sum(p * e for p, e, r in law) == sum(p * r for p, e, r in law) == 0
var_r = sum(p * r * r for p, e, r in law)
cov = sum(p * e * r for p, e, r in law)
beta = cov / var_r
before2 = sum(p * e * e for p, e, r in law)
after2 = sum(p * (e - beta * r) ** 2 for p, e, r in law)
before1 = sum(p * abs(e) for p, e, r in law)
after1 = sum(p * abs(e - beta * r) for p, e, r in law)
assert (var_r, cov, beta) == (F(17, 5), F(5), F(25, 17))
assert (before2, after2, before1, after1) == (F(10), F(45, 17), F(1), F(27, 17))
assert after2 < before2 and after1 > before1

# A full-rank 3D embedding meets the original positive-definite covariance condition.
law3 = [(p / 4, (e, F(0), F(0)), (r0, F(u), F(v)))
        for p, e, r0 in law for u in (-1, 1) for v in (-1, 1)]
caa = [[sum(p * r[i] * r[j] for p, e, r in law3) for j in range(3)] for i in range(3)]
cta = [[sum(p * e[i] * r[j] for p, e, r in law3) for j in range(3)] for i in range(3)]
assert caa == [[F(17, 5), F(0), F(0)], [F(0), F(1), F(0)], [F(0), F(0), F(1)]]
assert cta == [[F(5), F(0), F(0)], [F(0), F(0), F(0)], [F(0), F(0), F(0)]]
scale = F(1, 10)
assert (before2 * scale**2, after2 * scale**2, before1 * scale, after1 * scale) == (F(1, 10), F(9, 340), F(1, 10), F(27, 170))

# A finite 2-output/1-regressor illustration of the general MM identity.
# These are algebraic numbers, not proxy native evaluation inputs.
r = [-3., -2., -1., 0., 1., 2., 3.]
y = [(1 + .4 * x + (4 if x == 3 else 0), -.3 + .2 * x) for x in r]
w = [1 / len(r)] * len(r)
tau, ridge = .25, .1
coef = [[0., 0.], [0., 0.]]

def squared_errors(theta):
    return [sum((target[j] - theta[j][0] - theta[j][1] * x) ** 2 for j in range(2))
            for x, target in zip(r, y)]

def objective(theta):
    return sum(a * math.sqrt(s + tau * tau) for a, s in zip(w, squared_errors(theta))) + ridge / 2 * sum(row[1] ** 2 for row in theta)

values = [objective(coef)]
min_det = float('inf')
max_upper_gap = 0.
for iteration in range(40):
    old_s = squared_errors(coef)
    q = [a / math.sqrt(s + tau * tau) for a, s in zip(w, old_s)]
    h00 = sum(q)
    h01 = sum(a * x for a, x in zip(q, r))
    h11 = sum(a * x * x for a, x in zip(q, r)) + ridge
    det = h00 * h11 - h01 * h01
    assert h00 > 0 and det > 0
    min_det = min(min_det, det)
    new = []
    for j in range(2):
        b0 = sum(a * target[j] for a, target in zip(q, y))
        b1 = sum(a * x * target[j] for a, x, target in zip(q, r, y))
        new.append([(h11 * b0 - h01 * b1) / det, (h00 * b1 - h01 * b0) / det])
    new_s = squared_errors(new)
    upper = sum(a * (math.sqrt(sk + tau * tau) + (s - sk) / (2 * math.sqrt(sk + tau * tau)))
                for a, s, sk in zip(w, new_s, old_s)) + ridge / 2 * sum(row[1] ** 2 for row in new)
    actual = objective(new)
    assert actual <= upper + 1e-12 and upper <= values[-1] + 1e-12
    max_upper_gap = max(max_upper_gap, upper - actual)
    coef = new
    values.append(actual)
assert all(b <= a + 1e-12 for a, b in zip(values, values[1:]))
assert all(0 <= math.sqrt(s + tau * tau) - math.sqrt(s) <= tau + 1e-12 for s in squared_errors(coef))
print(json.dumps({
    'status': 'passed',
    'scope': 'Finite mathematical witnesses, not general proof, candidate code, benchmark, scorer execution or efficacy',
    'counterexample': {'zero_means': True, 'reference_variance': str(var_r), 'cross_covariance': str(cov),
                      'squared_optimum': str(beta), 'squared_risk_before': str(before2), 'squared_risk_after': str(after2),
                      'unsquared_risk_before': str(before1), 'unsquared_risk_after': str(after1),
                      'full_rank_3d_embedding_checked': True, 'positive_reference_covariance_diagonal': ['17/5', '1', '1'],
                      'bounded_rescaling': {'scale': str(scale), 'errors_and_features_within': [-1, 1],
                          'squared_risk_before': str(before2 * scale**2), 'squared_risk_after': str(after2 * scale**2),
                          'unsquared_risk_before': str(before1 * scale), 'unsquared_risk_after': str(after1 * scale)}},
    'mm_witness': {'iterations': 40, 'initial_objective': values[0], 'final_objective': values[-1],
                   'minimum_gram_determinant': min_det, 'maximum_majorizer_gap': max_upper_gap,
                   'descent_inequality_checked': True, 'smoothing_bound_checked': True},
    'native_metric_gain_implied': False,
}, indent=2, allow_nan=False))

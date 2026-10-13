"""Loaders for the 30 instances, demand sequences and fluid-LP solutions, the LP start, and a check of each run."""
from fractions import Fraction
import math
from common import DATA, np
from model import Instance
from our_policy import recompute_aggregate


def load_model(case, T):
    """Instance (n, M) = case at horizon T, and its initial inventory B_T."""
    folder = DATA / 'instances' / case / str(T)
    with np.load(folder / 'inventory.npz') as z:
        cap = z['capacity'].copy()
    return Instance.load(folder / 'model.npz'), cap


def load_arrivals(case, path, T):
    """The first T requests of demand sequence `path` (0-29)."""
    with np.load(DATA / 'demand' / case / f'path_{path:03d}.npz') as z:
        return z['arrivals'][:T].copy()


def fluid_lp(case, T):
    """Optimal solution x* and dual prices of the fluid LP with demand T p and inventory B_T."""
    with np.load(DATA / 'fluid_lp' / f'{case}_{T}' / 'solution.npz') as z:
        return z['x'].copy(), z['price'].copy()


def lp_start(m, cap, T, x):
    """LP start of Section 6.2 (i): q_1,ja = x*_ja / (T p_j), the remaining fraction to rejection."""
    q = np.zeros(m.actions + m.n)
    for j in range(m.n):
        lo, hi = m.type_ptr[j], m.type_ptr[j + 1]
        frac = np.maximum(x[lo:hi], 0.0) / (T * m.p[j]) if m.p[j] > 0 else np.zeros(hi - lo)
        if frac.sum() > 1.0:
            frac /= frac.sum()
        q[lo + j] = 1.0 - frac.sum()
        q[lo + j + 1:hi + j + 1] = frac
    aggregate = recompute_aggregate(q, m.p.copy(), m.type_ptr, m.action_ptr, m.resources, len(cap))
    return q, aggregate


def epoch_flags(T):
    """Epoch boundaries T - ceil(T / 2^k), k = 1, 2, ..., of Ma et al. (2025), Algorithm 5 (epoch factor 1/2)."""
    rho, J, power = Fraction(1, 2), 0, Fraction(1)
    while power * T > 1:
        J += 1
        power *= rho
    out = [T - math.ceil(rho ** j * T) for j in range(1, J + 1)]
    flags = np.zeros(T + 1, dtype=np.bool_)
    flags[[t for t in out if 1 <= t <= T - 1]] = True
    return flags


def verify(m, arr, cap, revenue, remaining, decisions):
    """Executed actions belong to the arriving type, inventory never goes negative, and the reward adds up."""
    assert decisions.shape == arr.shape and np.all(decisions >= -1)
    take = decisions >= 0
    acts = decisions[take]
    assert np.all(acts >= m.type_ptr[arr[take]]) and np.all(acts < m.type_ptr[arr[take] + 1])
    counts = np.bincount(acts, minlength=m.actions)
    use = np.bincount(m.resources, weights=np.repeat(counts, np.diff(m.action_ptr)), minlength=len(cap))
    np.testing.assert_array_equal(cap - use, remaining)
    assert remaining.min() >= 0
    np.testing.assert_allclose(m.reward[acts].sum(), revenue, rtol=1e-9, atol=1e-6)

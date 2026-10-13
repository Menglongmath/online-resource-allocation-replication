"""The upper bound U of Section 6.4 on the hindsight LP, for each demand sequence (no LP is solved).

For every lambda >= 0 and arrival counts N, weak duality gives
    Phi(B_T, N) <= U(lambda) = lambda^T B_T + sum_j N_j max{0, max_a (r_ja - w_ja^T lambda)}.
lambda is chosen by L-BFGS-B on the log-sum-exp smoothing of U with tau = 1e-2, 1e-3, 1e-4 (an upper bound on U);
the bound is the exact U at the best lambda found. Sequences after the first start from the lambda of the previous
sequence (--init: from a lambda file, for example of a shorter horizon). U is valid for any lambda, so a recomputation
can give a slightly different valid bound; the paper uses results/huge/upper_bounds.csv.
Usage: python dual_bound.py --horizon T [--paths 0-9] [--threads 6] [--init lambda.npy]
Writes $OLP_SCRATCH/huge/upper_bounds.csv and $OLP_SCRATCH/huge/bounds/<T>/lambda_<k>.npy.
"""
import argparse
import csv
import numpy as np
from numba import njit, prange, set_num_threads
from scipy.optimize import minimize
from settings import SCRATCH, SEQUENCE_SEED
import instance as I
import implicit as im


@njit(cache=True, parallel=True)
def evaluate(lam, tau, types, N, orders, sizes, opts, nopts, shipping, penalty, pack, divisor, Z, F, capacity,
             bound, chunks):
    """tau > 0: smoothed U and its gradient; tau == 0: exact U and a subgradient."""
    d = len(lam)
    grads = np.zeros((chunks, d))
    vals = np.zeros(chunks)
    n = len(types)
    for c in prange(chunks):
        rew = np.empty(bound)
        dig = np.empty((bound, im.MAXS), dtype=np.uint8)
        score = np.empty(bound)
        price = np.zeros((im.MAXS, 12))
        g = grads[c]
        for k in range(c, n, chunks):
            j = types[k]
            b = j // Z
            q = sizes[b]
            A = im.menu(b, j % Z, orders, sizes, opts, nopts, shipping, penalty, pack, divisor, rew, dig)
            if A == 0:
                continue
            for s in range(q):
                i = orders[b, s]
                for dd in range(1, nopts[i] + 1):
                    price[s, dd] = lam[i * F + opts[i, dd - 1]]
            top, arg = 0.0, -1
            for a in range(A):
                v = rew[a]
                for s in range(q):
                    dd = dig[a, s]
                    if dd > 0:
                        v -= price[s, dd]
                score[a] = v
                if v > top:
                    top, arg = v, a
            w = N[k]
            if tau == 0.0:
                vals[c] += w * top
                if arg >= 0:
                    for s in range(q):
                        dd = dig[arg, s]
                        if dd > 0:
                            i = orders[b, s]
                            g[i * F + opts[i, dd - 1]] -= w
            else:
                z = np.exp(-top / tau)
                for a in range(A):
                    score[a] = np.exp((score[a] - top) / tau)
                    z += score[a]
                vals[c] += w * (top + tau * np.log(z))
                for a in range(A):
                    pa = w * score[a] / z
                    if pa > 0:
                        for s in range(q):
                            dd = dig[a, s]
                            if dd > 0:
                                i = orders[b, s]
                                g[i * F + opts[i, dd - 1]] -= pa
    grad = capacity.astype(np.float64).copy()
    for c in range(chunks):
        grad += grads[c]
    return lam @ capacity.astype(np.float64) + vals.sum(), grad


def realized(x, seed, T):
    """Types that arrive in the sequence with this seed, and their counts N_j."""
    counts = im.type_counts(seed, T, x['bacc'], x['bali'], x['zacc'], x['zali'], x['Z'], x['n'])
    types = np.flatnonzero(counts).astype(np.int64)
    return types, counts[types].astype(np.float64)


def bound(x, types, N, lam0, iters=(80, 60, 60)):
    args = (types, N, x['orders'], x['sizes'], x['opts'], x['nopts'], x['shipping'], x['penalty'], im.PACK,
            x['divisor'], x['Z'], x['F'], x['capacity'], x['bound'], 256)
    live = x['capacity'] > 0                     # pairs without inventory are in no menu: lambda = 0 there
    exact = lambda lam: evaluate(lam, 0.0, *args)[0]
    best_lam, best = lam0.copy(), exact(lam0)
    for tau, it in zip((1e-2, 1e-3, 1e-4), iters):
        if it == 0:
            continue

        def f(v):
            lam = np.zeros(len(live))
            lam[live] = v
            val, gr = evaluate(lam, tau, *args)
            return val, gr[live]
        res = minimize(f, best_lam[live], jac=True, method='L-BFGS-B', bounds=[(0, None)] * int(live.sum()),
                       options=dict(maxiter=it, maxfun=int(it * 1.3), maxcor=20))
        lam = np.zeros(len(live))
        lam[live] = res.x
        u = exact(lam)
        if u < best:
            best, best_lam = u, lam
    return best, best_lam


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--horizon', type=int, required=True)
    ap.add_argument('--paths', default='0-9')
    ap.add_argument('--threads', type=int, default=6)
    ap.add_argument('--init', default=None)
    ap.add_argument('--warm-iters', default='0,25,25', help='iterations per tau when starting from a previous lambda')
    a = ap.parse_args()
    set_num_threads(a.threads)
    x = I.load(SCRATCH / 'instances' / str(a.horizon))
    folder = SCRATCH / 'bounds' / str(a.horizon)
    folder.mkdir(parents=True, exist_ok=True)
    dest = SCRATCH / 'upper_bounds.csv'
    done = {(int(r['horizon']), int(r['path'])) for r in csv.DictReader(dest.open())} if dest.exists() else set()
    lo, _, hi = a.paths.partition('-')
    prev = sorted(folder.glob('lambda_*.npy'))
    lam = np.load(prev[-1]) if prev else (np.load(a.init) if a.init else np.zeros(len(x['capacity'])))
    for k in range(int(lo), int(hi or lo) + 1):
        if (a.horizon, k) in done:
            continue
        types, N = realized(x, SEQUENCE_SEED + k, a.horizon)
        iters = tuple(int(v) for v in a.warm_iters.split(',')) if lam.any() else (80, 60, 60)
        u, lam = bound(x, types, N, lam, iters)
        np.save(folder / f'lambda_{k}.npy', lam)
        new = not dest.exists()
        with dest.open('a', newline='') as h:
            w = csv.DictWriter(h, ['horizon', 'path', 'upper_bound'])
            if new:
                w.writeheader()
            w.writerow(dict(horizon=a.horizon, path=k, upper_bound=u))
        print(f'T={a.horizon:,} sequence {k}: U = {u:.3f}', flush=True)

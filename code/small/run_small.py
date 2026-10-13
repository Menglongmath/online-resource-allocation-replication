"""Sections 6.1-6.3: the hindsight LP, our policy and the price-based policies on the 30 instances, 6 horizons and
30 demand sequences. Writes results/small_network/runs.csv, one row per policy, instance, horizon and sequence.

Policies:
  ours_known            our policy with forecast p, starting solution of Section 6.2 (i)
  ours_known_lp_start   our policy with forecast p, LP start
  ours_estimated        our policy with forecast pi_t (unknown probabilities), starting solution of Section 6.2 (i)
  he_fallback, he       He et al. (2025), with and without the fallback rule
  dmd_fallback, dmd     DMD, step T^(-1/2), with and without the fallback rule
  ma2025_fallback, ma2025   Ma et al. (2025), Algorithm 5, with and without the fallback rule

Usage: python run_small.py [--workers 8]
"""
import argparse
import multiprocessing
from concurrent.futures import ProcessPoolExecutor, as_completed
from common import CASES, HORIZONS, PATHS, GAMMA, RESULTS, SCRATCH, np
import pandas as pd
from data import load_model, load_arrivals, fluid_lp, lp_start, epoch_flags, verify
from our_policy import heuristic_start, kappa_schedule, run as run_ours
from price_policies import he, dmd, ma2025
from column_solver import ColumnSolver

CHECKPOINTS = SCRATCH / 'small_network_checkpoints'


def cell(spec):
    case, T = spec
    out_file = CHECKPOINTS / f'{case}_{T}.csv'
    if out_file.exists():
        return pd.read_csv(out_file)
    m, cap = load_model(case, T)
    tp, ap, rr, r = m.type_ptr, m.action_ptr, m.resources, m.reward
    x, price0 = fluid_lp(case, T)
    kappas = np.ascontiguousarray(kappa_schedule(T, GAMMA))
    starts = {'ours_known': (heuristic_start(tp, ap, rr, r, m.p, cap, T, True), True),
              'ours_known_lp_start': (lp_start(m, cap, T, x), True),
              'ours_estimated': (heuristic_start(tp, ap, rr, r, m.p, cap, T, False), False)}
    rate = cap / T
    V = max(float(T), float(np.diff(ap).max() + np.sum(rate ** 2)))   # max{T, max ||w_ja||^2 + ||B_T / T||^2}
    flags = epoch_flags(T)
    solver = ColumnSolver(m, len(cap))
    rows = []
    for path in range(PATHS):
        arr = load_arrivals(case, path, T)
        bench = solver.solve(cap, np.bincount(arr, minlength=m.n).astype(float))[-1]   # hindsight LP
        assert bench['status'] == 'optimal'
        lp = float(bench['objective'])
        runs = {pol: run_ours(tp, ap, rr, r, m.p, m.ell, arr, cap, kappas, q, aggregate, known)
                for pol, ((q, aggregate), known) in starts.items()}
        for fb in (False, True):
            s = '_fallback' if fb else ''
            runs['he' + s] = he(tp, ap, rr, r, arr, cap, price0, rate, V, fb)
            runs['dmd' + s] = dmd(tp, ap, rr, r, arr, cap, T ** (-0.5), fb)
            runs['ma2025' + s] = ma2025(tp, ap, rr, r, arr, cap, flags, 0.1, fb)
        for pol, (rev, C, dec) in runs.items():
            verify(m, arr, cap, rev, C, dec)
            rows.append(dict(policy=pol, case=case, horizon=T, path=path, lp_hindsight=lp, revenue=float(rev),
                             regret=lp - float(rev), percentage_loss=100 * (lp - float(rev)) / lp))
    d = pd.DataFrame(rows)
    d.to_csv(out_file, index=False)
    return d


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--workers', type=int, default=8)
    args = ap.parse_args()
    CHECKPOINTS.mkdir(parents=True, exist_ok=True)
    specs = sorted(((c, T) for c in CASES for T in HORIZONS), key=lambda s: (-s[1], -int(s[0].split('_M')[1])))
    parts = []
    with ProcessPoolExecutor(max_workers=args.workers, mp_context=multiprocessing.get_context('spawn')) as pool:
        for i, f in enumerate(as_completed([pool.submit(cell, s) for s in specs]), 1):
            parts.append(f.result())
            if i % 20 == 0 or i == len(specs):
                print(f'{i}/{len(specs)} instance-horizon cells done', flush=True)
    d = pd.concat(parts).sort_values(['policy', 'case', 'horizon', 'path'])
    assert len(d) == 9 * len(CASES) * len(HORIZONS) * PATHS
    out = RESULTS / 'small_network' / 'runs.csv'
    out.parent.mkdir(parents=True, exist_ok=True)
    d.to_csv(out, index=False)
    print(f'wrote {out} ({len(d)} rows)')


if __name__ == '__main__':
    main()

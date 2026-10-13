"""Section 6.2, DMD: the step-size grid of Balseiro, Lu and Mirrokni (2023, Appendix H.1) on 30 other instances.

Steps T^(-1/2) (used in the paper) and s / sqrt(T d), s in {0.1, 1, 10, 100}, d = number of item-center pairs with
positive inventory; DMD without the fallback rule; total reward over the 30 instances and 20 demand sequences at
each horizon. Run make_instances.py --validation first.
Writes results/dmd_step_check/dmd_validation.csv and total_reward.csv.
Usage: python run_dmd_step_check.py [--workers 8]
"""
import argparse
import multiprocessing
from concurrent.futures import ProcessPoolExecutor, as_completed
from common import RESULTS, SCRATCH, CASES, HORIZONS, np
import pandas as pd
from model import Instance
from price_policies import dmd
from data import verify

S_GRID = [0.1, 1.0, 10.0, 100.0]
PATHS = 20
VALID = SCRATCH / 'validation'


def job(spec):
    case, T = spec
    folder = VALID / 'instances' / case / str(T)
    m = Instance.load(folder / 'model.npz')
    with np.load(folder / 'inventory.npz') as z:
        cap = z['capacity'].copy()
    d = int((cap > 0).sum())
    rules = [('paper', T ** -0.5)] + [(f's={s:g}', s / np.sqrt(T * d)) for s in S_GRID]
    rows = []
    for path in range(PATHS):
        with np.load(VALID / 'demand' / case / f'path_{path:03d}.npz') as z:
            arr = z['arrivals'][:T].copy()
        for name, step in rules:
            out = dmd(m.type_ptr, m.action_ptr, m.resources, m.reward, arr, cap, step, False)
            verify(m, arr, cap, out[0], out[1], out[2])
            rows.append(dict(case=case, horizon=T, path=path, rule=name, step=step, revenue=float(out[0])))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--workers', type=int, default=8)
    args = ap.parse_args()
    specs = sorted(((c, T) for c in CASES for T in HORIZONS), key=lambda s: -s[1])
    rows = []
    with ProcessPoolExecutor(args.workers, mp_context=multiprocessing.get_context('spawn')) as pool:
        for f in as_completed([pool.submit(job, s) for s in specs]):
            rows += f.result()
    d = pd.DataFrame(rows).sort_values(['rule', 'case', 'horizon', 'path'])
    out = RESULTS / 'dmd_step_check'
    out.mkdir(parents=True, exist_ok=True)
    d.to_csv(out / 'dmd_validation.csv', index=False)
    total = d.pivot_table(index='rule', columns='horizon', values='revenue', aggfunc='sum')
    total.to_csv(out / 'total_reward.csv')
    print(total.round(1).to_string())
    best = total.drop('paper').sum(axis=1).idxmax()
    print(f'best grid step {best}; earns less than T^(-1/2) at every horizon:',
          bool((total.loc[best] < total.loc['paper']).all()))


if __name__ == '__main__':
    main()

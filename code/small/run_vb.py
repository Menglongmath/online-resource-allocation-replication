"""Section 6.3: Vera and Banerjee (2021), Algorithm 4, with known probabilities, on the 30 instances, 6 horizons and
30 demand sequences (vb_policy.py). It solves T LPs per sequence, so the script keeps a checkpoint per run and accepts
a subset of instances and sequences.

Writes results/small_network/runs_vera_banerjee.csv, with the hindsight LP value taken from
results/small_network/runs.csv (run run_small.py first).
Usage: python run_vb.py [--workers 16] [--cases I10_M1,I10_M2] [--paths 0-29]
"""
import argparse
import json
import multiprocessing
from concurrent.futures import ProcessPoolExecutor, as_completed
from common import CASES, HORIZONS, PATHS, RESULTS, SCRATCH
import pandas as pd
from data import load_model, load_arrivals, verify
from vb_policy import run_vb

CHECKPOINTS = SCRATCH / 'vera_banerjee_checkpoints'


def job(spec):
    case, T, path, lp = spec
    f = CHECKPOINTS / f'{case}_{T}_{path}.json'
    if f.exists():
        return json.loads(f.read_text())
    m, cap = load_model(case, T)
    arr = load_arrivals(case, path, T)
    rev, C, dec = run_vb(m, arr, cap)
    verify(m, arr, cap, rev, C, dec)
    row = dict(policy='vera_banerjee', case=case, horizon=T, path=path, lp_hindsight=lp, revenue=float(rev),
               regret=lp - float(rev), percentage_loss=100 * (lp - float(rev)) / lp)
    f.write_text(json.dumps(row) + '\n')
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--workers', type=int, default=16)
    ap.add_argument('--cases', default=','.join(CASES))
    ap.add_argument('--paths', default=f'0-{PATHS - 1}')
    args = ap.parse_args()
    lo, hi = (int(v) for v in args.paths.split('-'))
    cases = args.cases.split(',')
    CHECKPOINTS.mkdir(parents=True, exist_ok=True)
    lp = pd.read_csv(RESULTS / 'small_network' / 'runs.csv', usecols=['policy', 'case', 'horizon', 'path', 'lp_hindsight'])
    lp = lp[lp.policy == 'ours_known'].set_index(['case', 'horizon', 'path']).lp_hindsight
    specs = sorted(((c, T, p, float(lp[(c, T, p)])) for c in cases for T in HORIZONS for p in range(lo, hi + 1)),
                   key=lambda s: -s[1])
    rows = []
    with ProcessPoolExecutor(max_workers=args.workers, mp_context=multiprocessing.get_context('spawn')) as pool:
        for i, f in enumerate(as_completed([pool.submit(job, s) for s in specs]), 1):
            rows.append(f.result())
            if i % 50 == 0 or i == len(specs):
                print(f'{i}/{len(specs)} runs done', flush=True)
    d = pd.DataFrame(rows).sort_values(['case', 'horizon', 'path'])
    out = RESULTS / 'small_network' / 'runs_vera_banerjee.csv'
    if out.exists():                       # merge with the rows of other subsets already in the file
        old = pd.read_csv(out)
        d = pd.concat([old, d]).drop_duplicates(['case', 'horizon', 'path'], keep='last').sort_values(
            ['case', 'horizon', 'path'])
    d.to_csv(out, index=False)
    print(f'wrote {out} ({len(d)} rows)')


if __name__ == '__main__':
    main()

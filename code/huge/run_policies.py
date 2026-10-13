"""Section 6.4: our policy, DMD and Ma et al. (2025), all with forecast pi_t / no knowledge of p and the fallback
rule, on the instance built by build_instance.py. Sequence k (seed SEQUENCE_SEED + k) is the same for all policies.
Records the reward (normalized units) and the CPU time of each run in $OLP_SCRATCH/huge/runs/ (one file per run).
Usage: python run_policies.py [--horizons 100000,1000000,10000000,100000000] [--paths 10] [--workers 6]
"""
import argparse
import json
import multiprocessing
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from fractions import Fraction
import math
import numpy as np
from settings import SCRATCH, HORIZONS, PATHS, SEQUENCE_SEED, GAMMA
import instance as I
import implicit as im

POLICIES = ('ours_estimated', 'dmd_fallback', 'ma2025_fallback')
RUNS = SCRATCH / 'runs'
_CACHE = {}


def boundaries(T):
    """Epoch boundaries T - ceil(T / 2^k) of Ma et al. (2025), Algorithm 5, sorted."""
    rho, J, power = Fraction(1, 2), 0, Fraction(1)
    while power * T > 1:
        J += 1
        power *= rho
    out = sorted(t for t in {T - math.ceil(rho ** j * T) for j in range(1, J + 1)} if 1 <= t <= T - 1)
    return np.array(out, dtype=np.int64)


def job(spec):
    policy, T, path = spec
    if T not in _CACHE:
        _CACHE.clear()
        _CACHE[T] = I.load(SCRATCH / 'instances' / str(T))
    x = _CACHE[T]
    seed = SEQUENCE_SEED + path
    common = (x['orders'], x['sizes'], x['opts'], x['nopts'], x['shipping'], x['penalty'], im.PACK, x['divisor'],
              x['Z'], x['F'])
    c0 = time.process_time()
    if policy == 'ours_estimated':
        revenue, remaining = im.ours(*common, x['n'], x['best_local'], x['best_dig'], x['load'], x['aggregate'],
                                     x['capacity'], T, GAMMA, seed, x['bacc'], x['bali'], x['zacc'], x['zali'],
                                     x['bound'])
    else:
        mode = 0 if policy == 'dmd_fallback' else 2
        bnd = boundaries(T) if mode == 2 else np.empty(0, dtype=np.int64)
        revenue, remaining = im.price_policy(*common, x['capacity'], T, mode, bnd, 0.1, seed, x['bacc'], x['bali'],
                                             x['zacc'], x['zali'], x['bound'])
    cpu = time.process_time() - c0
    assert remaining.min() >= 0
    row = dict(policy=policy, horizon=T, path=path, revenue=float(revenue), cpu_seconds=cpu)
    (RUNS / f'{policy}_{T}_{path}.json').write_text(json.dumps(row) + '\n')
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--horizons', default=','.join(str(T) for T in HORIZONS))
    ap.add_argument('--paths', type=int, default=None, help='first k sequences only (default: all)')
    ap.add_argument('--workers', type=int, default=6)
    args = ap.parse_args()
    RUNS.mkdir(parents=True, exist_ok=True)
    horizons = sorted((int(h) for h in args.horizons.split(',')), reverse=True)
    specs = [(p, T, k) for T in horizons for k in range(PATHS[T] if args.paths is None else min(args.paths, PATHS[T]))
             for p in POLICIES if not (RUNS / f'{p}_{T}_{k}.json').exists()]
    print(f'{len(specs)} runs', flush=True)
    with ProcessPoolExecutor(args.workers, mp_context=multiprocessing.get_context('spawn')) as pool:
        for f in as_completed([pool.submit(job, s) for s in specs]):
            r = f.result()
            print(f"{r['policy']:16s} T={r['horizon']:,} sequence {r['path']}: reward {r['revenue']:.3f}", flush=True)


if __name__ == '__main__':
    main()

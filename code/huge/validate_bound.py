"""Section 6.4: on a smaller instance of the same family, where the hindsight LP can be solved, compare U with
Phi(B_T, N). Instance: 100 items, 20 baskets of each size from two to five, catalogue seed 26093001, T = 20,000;
sequences 0 and 1. The hindsight LP is solved by column generation (code/small/column_solver.py).
Writes results/huge/upper_bound_validation.json.
Usage: python validate_bound.py
"""
import json
import sys
from types import SimpleNamespace
import numpy as np
from settings import ROOT, RESULTS, SEQUENCE_SEED
import instance as I
import implicit as im
from dual_bound import realized, bound

sys.path.insert(0, str(ROOT / 'code' / 'small'))
from column_solver import ColumnSolver  # noqa: E402

ITEMS, NPER, SEED, T = 100, 20, 26093001, 20000


def explicit_menus(x):
    """All menus of the instance as arrays (type_ptr, action_ptr, resources, reward)."""
    rew = np.empty(x['bound'])
    dig = np.empty((x['bound'], im.MAXS), dtype=np.uint8)
    tp, ap, res, rw = [0], [0], [], []
    for j in range(x['n']):
        b = j // x['Z']
        A = im.menu(b, j % x['Z'], x['orders'], x['sizes'], x['opts'], x['nopts'], x['shipping'], x['penalty'],
                    im.PACK, x['divisor'], rew, dig)
        for a in range(A):
            for s in range(x['sizes'][b]):
                d = dig[a, s]
                if d > 0:
                    i = x['orders'][b, s]
                    res.append(i * x['F'] + x['opts'][i, d - 1])
            ap.append(len(res))
            rw.append(rew[a])
        tp.append(len(rw))
    return np.array(tp), np.array(ap), np.array(res, dtype=np.int32), np.array(rw)


if __name__ == '__main__':
    x, _ = I.build(ITEMS, NPER, SEED, T)
    tp, ap, res, rw = explicit_menus(x)
    m = SimpleNamespace(type_ptr=tp, action_ptr=ap, resources=res, reward=rw, n=x['n'], actions=len(rw))
    out = []
    for path in range(2):
        types, N = realized(x, SEQUENCE_SEED + path, T)
        counts = np.zeros(x['n'])
        counts[types] = N
        lp = float(ColumnSolver(m, len(x['capacity'])).solve(x['capacity'].astype(float), counts)[-1]['objective'])
        u, _ = bound(x, types, N, np.zeros(len(x['capacity'])))
        assert u >= lp * (1 - 1e-9)
        out.append(dict(path=path, hindsight_lp=lp, upper_bound=u, excess_percent=100 * (u - lp) / lp))
        print(out[-1], flush=True)
    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / 'upper_bound_validation.json').write_text(json.dumps(out, indent=1) + '\n')

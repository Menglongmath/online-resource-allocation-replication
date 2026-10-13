"""The fluid LP with demand T p and inventory B_T for every instance and horizon (column_solver.py). Its dual
gives the initial prices of He et al. (2025), and its primal solution gives the LP start of our policy.
Writes data/small_network/fluid_lp/<case>_<T>/{solution.npz, certificate.json}; --check compares a recomputation
with the stored solutions instead.
Usage: python compute_fluid_lp.py [--check]
"""
import argparse
import json
from common import DATA, CASES, HORIZONS, np
from data import load_model
from column_solver import ColumnSolver


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--check', action='store_true')
    args = ap.parse_args()
    worst = 0.0
    for case in CASES:
        for T in HORIZONS:
            m, cap = load_model(case, T)
            x, price, _, cert = ColumnSolver(m, len(cap)).solve(cap, T * m.p)
            assert cert['status'] == 'optimal'
            folder = DATA / 'fluid_lp' / f'{case}_{T}'
            if args.check:
                with np.load(folder / 'solution.npz') as z:
                    worst = max(worst, float(np.abs(z['x'] - x).max()), float(np.abs(z['price'] - price).max()))
                continue
            folder.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(folder / 'solution.npz', x=x, price=price)
            (folder / 'certificate.json').write_text(json.dumps(cert, indent=2) + '\n')
    if args.check:
        print('largest difference from the stored solutions:', worst)


if __name__ == '__main__':
    main()

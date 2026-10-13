"""Collect the Section 6.4 runs (run_policies.py) and bounds (dual_bound.py) into results/huge/runs.csv and
results/huge/upper_bounds.csv. Percentage loss: 100 (U - reward) / U with the bound U of the sequence.
Usage: python assemble_results.py
"""
import json
import pandas as pd
from settings import SCRATCH, RESULTS

if __name__ == '__main__':
    U = pd.read_csv(SCRATCH / 'upper_bounds.csv').sort_values(['horizon', 'path'])
    u = U.set_index(['horizon', 'path']).upper_bound
    d = pd.DataFrame([json.loads(f.read_text()) for f in (SCRATCH / 'runs').glob('*.json')])
    d['upper_bound'] = [u[(h, k)] for h, k in zip(d.horizon, d.path)]
    d['loss_percent'] = 100 * (d.upper_bound - d.revenue) / d.upper_bound
    assert not d.duplicated(['policy', 'horizon', 'path']).any() and (d.revenue <= d.upper_bound * (1 + 1e-10)).all()
    RESULTS.mkdir(parents=True, exist_ok=True)
    d.sort_values(['policy', 'horizon', 'path']).to_csv(RESULTS / 'runs.csv', index=False)
    U.to_csv(RESULTS / 'upper_bounds.csv', index=False)
    print(d.groupby(['policy', 'horizon']).loss_percent.mean().unstack().round(3).to_string())

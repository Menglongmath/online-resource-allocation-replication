"""Paths and settings shared by the small-network scripts (Sections 6.1-6.3).

Every script in this folder imports this module first. It fixes one thread per process and keeps numba's
compilation cache in $OLP_SCRATCH (default ~/.olp_scratch).
"""
from pathlib import Path
import os
import sys

CODE = Path(__file__).resolve().parent
ROOT = CODE.parents[1]
DATA = ROOT / 'data' / 'small_network'
RESULTS = ROOT / 'results'
SCRATCH = Path(os.environ.get('OLP_SCRATCH', Path.home() / '.olp_scratch'))
for key in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMBA_NUM_THREADS',
            'VECLIB_MAXIMUM_THREADS'):
    os.environ[key] = '1'
os.environ.setdefault('NUMBA_CACHE_DIR', str(SCRATCH / 'numba_cache'))
sys.dont_write_bytecode = True
if str(CODE) not in sys.path:
    sys.path.insert(0, str(CODE))

import numpy as np  # noqa: E402

CASES = tuple(f'I{n}_M{m}' for n in (10, 20, 30, 40, 50, 60) for m in (1, 2, 3, 4, 5))  # (n, M)
HORIZONS = (1000, 5000, 10000, 20000, 50000, 100000)
PATHS = 30                                   # demand sequences per instance
GAMMA = 0.6                                  # kappa_t = min{t, T - t + 1}^(-gamma)

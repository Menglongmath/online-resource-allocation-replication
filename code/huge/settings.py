"""Paths and settings shared by the Section 6.4 scripts.

The instance arrays (about 6 GB per horizon), run checkpoints and bound files go to $OLP_SCRATCH/huge
(default ~/.olp_scratch/huge); the results reported in the paper are in results/huge/.
"""
from pathlib import Path
import os
import sys

CODE = Path(__file__).resolve().parent
ROOT = CODE.parents[1]
LARGE = ROOT / 'data' / 'large_network'          # regions and fulfillment centers of the large network of Ma (2023)
RESULTS = ROOT / 'results' / 'huge'
SCRATCH = Path(os.environ.get('OLP_SCRATCH', Path.home() / '.olp_scratch')) / 'huge'
SCRATCH.mkdir(parents=True, exist_ok=True)
os.environ['NUMBA_CACHE_DIR'] = str(SCRATCH / 'numba_cache')
for key in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
    os.environ[key] = '1'
sys.dont_write_bytecode = True
if str(CODE) not in sys.path:
    sys.path.insert(0, str(CODE))

ITEMS, NPER, SEED = 10000, 213000, 26093021      # items, baskets of each size from two to five, catalogue seed
HORIZONS = (10**5, 10**6, 10**7, 10**8)
PATHS = {10**5: 10, 10**6: 10, 10**7: 10, 10**8: 5}   # demand sequences per horizon
SEQUENCE_SEED = 26093200                         # sequence k is drawn from MT19937 seeded SEQUENCE_SEED + k
GAMMA = 0.6

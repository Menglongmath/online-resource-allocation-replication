# Section 6.4: the large instance

The large network of Ma (2023) (99 regions, 10 fulfillment centers; `data/large_network/`), 10,000 items and
213,000 baskets of each size from two to five, catalogue seed 26093021 (`generator.py`). Menus are not stored:
`implicit.py::menu` enumerates the menu of the arriving request. Demand sequence k is drawn inside each policy from
MT19937 seeded 26093200 + k, so all policies see the same requests.

Working files go to `$OLP_SCRATCH/huge` (default `~/.olp_scratch/huge`); each horizon's instance takes about 6 GB.

| Step | Command | Time |
|---|---|---|
| 1 | `python build_instance.py` | about 40 min, all cores |
| 2 | `python run_policies.py` — our policy, DMD and Ma et al. (2025), all with the fallback rule | up to 45 min per run at T = 10^8 |
| 3 | `python dual_bound.py --horizon T --paths 0-9` for each T (`--paths 0-4` at T = 10^8) — the upper bound U | minutes per sequence |
| 4 | `python validate_bound.py` — U against the hindsight LP on a smaller instance of the same family | minutes |
| 5 | `python assemble_results.py` — writes `results/huge/runs.csv` and `upper_bounds.csv` | seconds |

U is valid for every price vector. It is found by L-BFGS-B, started from zero prices or from the prices found for
another sequence, so a recomputation can give a slightly different valid bound; the paper uses
`results/huge/upper_bounds.csv`.

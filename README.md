# Numerical experiments: code, data and results

Code, data and results of Section 6 (and of the numerical paragraph of the introduction) of
"A First-Order Learning Algorithm for Online Resource Allocation with Constant Regret".

## Layout

```
data/                                     see data/README.md
  small_network/                          the 30 instances (n, M) = I<n>_M<M>, their demand sequences and fluid LPs,
                                          and the network of Ma (2023) with 10 regions and 5 centers
  large_network/                          the network of Ma (2023) with 99 regions and 10 centers (Section 6.4)
code/
  small/                                  Sections 6.1-6.3
  huge/                                   Section 6.4 (see code/huge/README.md)
  build_paper_tables.py                   Tables 1-3, Figure 1 and the numbers quoted in the text
results/
  small_network/runs.csv                  hindsight LP and the runs of our policy and the price-based policies
  small_network/runs_vera_banerjee.csv    the runs of Vera and Banerjee (2021)
  dmd_step_check/                         DMD step sizes on 30 other instances (Section 6.2)
  huge/                                   Section 6.4: runs, upper bounds U, instance statistics, check of U
  paper/                                  Tables 1-3 (LaTeX), Figure 1, numbers.txt (the numbers quoted in the text)
```

## Environment

Python 3.12 with the packages in `requirements.txt`. Set `OLP_SCRATCH` to a local folder for compilation caches,
checkpoints and the files of Section 6.4 (default `~/.olp_scratch`).

## Reproducing the results

From `code/small/`:

| Step | Command | Output | Time |
|---|---|---|---|
| 1 | `python run_small.py` | `results/small_network/runs.csv` | about 5 min with 16 workers |
| 2 | `python run_vb.py` | `results/small_network/runs_vera_banerjee.csv` | about 100 CPU hours (about 10 hours with 16 workers) |
| 3 | `python make_instances.py --validation` then `python run_dmd_step_check.py` | `results/dmd_step_check/` | a few minutes |

The instances, demand sequences and fluid LPs in `data/small_network` are produced by `python make_instances.py`
and `python compute_fluid_lp.py`; `make_instances.py --check` and `compute_fluid_lp.py --check` compare a
regeneration with the stored files.

For Section 6.4, follow `code/huge/README.md`. Then, from `code/`, run `python build_paper_tables.py` to write
`results/paper/`.

All policies are deterministic given a demand sequence.

## Policies (names in the result files)

| Name | Policy | Paper |
|---|---|---|
| `ours_known` | our policy, forecast p, starting solution of Section 6.2 (i) | Table 1 |
| `ours_known_lp_start` | our policy, forecast p, LP start | Table 1 |
| `ours_estimated` | our policy, forecast π_t (unknown p) | Tables 2, 3 |
| `vera_banerjee` | Vera and Banerjee (2021), Algorithm 4 | Table 1 |
| `he_fallback`, `he` | He et al. (2025), Algorithm 1, with and without the fallback rule | Table 1; text |
| `dmd_fallback`, `dmd` | DMD (Balseiro, Lu and Mirrokni 2023, Algorithm 1), with and without the fallback rule | Tables 2, 3; text |
| `ma2025_fallback`, `ma2025` | Ma et al. (2025), Algorithm 5 without the regularizer, with and without the fallback rule | Tables 2, 3; text |

Our policy always uses the fallback rule of Section 6.2. Rewards are divided by the largest reward of the instance.
Percentage loss is 100 (Φ(B_T, N) − reward) / Φ(B_T, N), with Φ the hindsight LP (Sections 6.1-6.3), or with Φ
replaced by the upper bound U (Section 6.4).

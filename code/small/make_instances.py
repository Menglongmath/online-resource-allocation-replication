"""Generate the instances and demand sequences of Section 6 (instance_generator.py).

  python make_instances.py               the 30 instances, horizons 1,000-100,000, and 30 demand sequences of
                                         100,000 requests each, written to data/small_network/{instances,demand}
  python make_instances.py --check       regenerate them in memory and compare with the files in data/small_network
  python make_instances.py --validation  the 30 other instances of the DMD step-size check (20 demand sequences each),
                                         written to $OLP_SCRATCH/validation

Seeds: the catalogue of n items uses SeedSequence([s, n, 1]) and the carried items SeedSequence([s, n, 2]); demand
sequence k of instance (n, M) uses SeedSequence([s', n, M, k, 920]). Reported instances: s = SEEDS[n], s' = 20260919.
Validation instances: s = 26092501, s' = 26092502.
"""
import argparse
from common import DATA, SCRATCH, CASES, HORIZONS, PATHS, np
from model import Instance
import instance_generator as gen

SEEDS = {10: 20260910, 20: 20260909, 30: 20260909, 40: 20260910, 50: 20260909, 60: 20260910}
DEMAND_SEED = 20260919
VALIDATION_SEED, VALIDATION_DEMAND_SEED, VALIDATION_PATHS = 26092501, 26092502, 20


def generate(case, seed, demand_seed, paths):
    n, M = int(case.split('_')[0][1:]), int(case.split('_M')[1])
    built = gen.build(n, M, seed, HORIZONS)
    longest = built[max(HORIZONS)][0]
    sequences = [gen.demand(longest, [demand_seed, n, M, k, 920]) for k in range(paths)]
    return built, sequences


def write(root, case, built, sequences):
    for T, (m, cap) in built.items():
        folder = root / 'instances' / case / str(T)
        folder.mkdir(parents=True, exist_ok=True)
        m.save(folder / 'model.npz')
        np.savez_compressed(folder / 'inventory.npz', capacity=cap)
    folder = root / 'demand' / case
    folder.mkdir(parents=True, exist_ok=True)
    for k, arrivals in enumerate(sequences):
        np.savez_compressed(folder / f'path_{k:03d}.npz', arrivals=arrivals)


def check(case, built, sequences):
    for T, (m, cap) in built.items():
        folder = DATA / 'instances' / case / str(T)
        stored = Instance.load(folder / 'model.npz')
        with np.load(folder / 'inventory.npz') as z:
            assert np.array_equal(z['capacity'], cap), (case, T)
        for name in ('type_ptr', 'action_ptr', 'resources', 'reward', 'p', 'ell'):
            assert np.array_equal(getattr(stored, name), getattr(m, name)), (case, T, name)
    for k, arrivals in enumerate(sequences):
        with np.load(DATA / 'demand' / case / f'path_{k:03d}.npz') as z:
            assert np.array_equal(z['arrivals'], arrivals), (case, k)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--check', action='store_true')
    ap.add_argument('--validation', action='store_true')
    args = ap.parse_args()
    for case in CASES:
        n = int(case.split('_')[0][1:])
        if args.validation:
            write(SCRATCH / 'validation', case, *generate(case, VALIDATION_SEED, VALIDATION_DEMAND_SEED,
                                                          VALIDATION_PATHS))
        elif args.check:
            check(case, *generate(case, SEEDS[n], DEMAND_SEED, PATHS))
        else:
            write(DATA, case, *generate(case, SEEDS[n], DEMAND_SEED, PATHS))
        print(case, 'done', flush=True)
    if args.check:
        print('all instances and demand sequences agree with data/small_network')


if __name__ == '__main__':
    main()

"""Build the Section 6.4 instance at T = 10^5, 10^6, 10^7 and 10^8 (about 20 minutes per build, all cores).

T = 10^6 and 10^8 reuse the T = 10^7 build (the same item-center pairs are stocked); T = 10^5 has fewer stocked pairs
and is built separately. Writes $OLP_SCRATCH/huge/instances/<T>/ and the instance statistics quoted in the paper to
results/huge/instance.json.
Usage: python build_instance.py
"""
import json
from settings import SCRATCH, RESULTS, ITEMS, NPER, SEED
import instance as I

if __name__ == '__main__':
    stats = {}
    base, stats[10**7] = I.build(ITEMS, NPER, SEED, 10**7)
    for T in (10**7, 10**6, 10**8):
        x = base if T == 10**7 else I.retarget(base, T)
        I.save(x, SCRATCH / 'instances' / str(T))
        stats[T] = dict(stats[10**7], stocked_pairs=int((x['capacity'] > 0).sum()))
        print('built T =', T, flush=True)
    small, stats[10**5] = I.build(ITEMS, NPER, SEED, 10**5)
    I.save(small, SCRATCH / 'instances' / str(10**5))
    print('built T =', 10**5, flush=True)
    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / 'instance.json').write_text(json.dumps({str(T): stats[T] for T in sorted(stats)}, indent=1) + '\n')
    print(json.dumps(stats, indent=1))

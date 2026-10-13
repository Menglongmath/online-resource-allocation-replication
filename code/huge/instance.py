"""The Section 6.4 instance at one horizon: catalogue, inventory, stocked centers of each item, reward divisor,
highest-reward action of each type, starting solution and alias tables. Menus stay implicit (implicit.menu).
"""
import json
import numpy as np
from settings import LARGE
import generator as g
import implicit as im

CHUNKS = 400


def options(capacity, items, F):
    """opts[i, :nopts[i]]: the centers with positive inventory of item i, ascending."""
    nopts = np.zeros(items, dtype=np.int64)
    opts = np.zeros((items, F), dtype=np.int64)
    for i in range(items):
        c = np.flatnonzero(capacity[i * F:(i + 1) * F] > 0)
        opts[i, :len(c)] = c
        nopts[i] = len(c)
    return opts, nopts


def build(items, n_per, seed, T):
    """Returns the instance and its statistics (types, type-action pairs, menu sizes, stocked pairs)."""
    dist, share = g.geography(LARGE / 'cities.csv', LARGE / 'fulfillment_centers.csv')
    st = g.structure(items, 5, seed, dist, share, n_per=n_per)
    Z, F = st['Z'], st['K']
    capacity = g.capacity_for(st['nominal'], T)
    opts, nopts = options(capacity, items, F)
    orders = np.ascontiguousarray(st['orders'], dtype=np.int64)
    sizes = np.asarray(st['sizes'], dtype=np.int64)
    inst = dict(items=items, T=T, Z=Z, F=F, n=len(st['p']), orders=orders, sizes=sizes, opts=opts, nopts=nopts,
                shipping=np.ascontiguousarray(st['shipping']), penalty=np.asarray(st['penalty']), capacity=capacity,
                nominal=st['nominal'])
    inst['bound'] = im.menu_bound(nopts)
    counts, top = im.pass_counts(orders, sizes, opts, nopts, inst['shipping'], inst['penalty'], im.PACK, Z,
                                 inst['bound'], CHUNKS)
    inst['divisor'] = float(top.max())          # largest reward of the instance
    inst['best_local'], inst['best_dig'] = im.pass_best(orders, sizes, opts, nopts, inst['shipping'],
                                                        inst['penalty'], im.PACK, inst['divisor'], Z, inst['bound'],
                                                        CHUNKS)
    inst['load'], inst['aggregate'] = im.warm_start(orders, sizes, opts, F, Z, inst['best_local'], inst['best_dig'],
                                                    capacity, T)
    inst['bacc'], inst['bali'] = im.alias_table(st['p'].reshape(-1, Z).sum(axis=1))
    inst['zacc'], inst['zali'] = im.alias_table(share)
    stats = dict(types=len(st['p']), type_action_pairs=int(counts.sum()), largest_menu=int(counts.max()),
                 mean_menu_size_under_p=float((st['p'] * counts).sum()), stocked_pairs=int((capacity > 0).sum()))
    return inst, stats


def retarget(inst, T):
    """The same instance at another horizon whose inventory leaves the stocked centers of every item unchanged
    (so the menus, the divisor and the highest-reward actions are unchanged): new inventory and starting solution."""
    capacity = g.capacity_for(inst['nominal'], T)
    opts, nopts = options(capacity, inst['items'], inst['F'])
    assert np.array_equal(opts, inst['opts']) and np.array_equal(nopts, inst['nopts'])
    out = dict(inst, T=T, capacity=capacity)
    out['load'], out['aggregate'] = im.warm_start(inst['orders'], inst['sizes'], inst['opts'], inst['F'], inst['Z'],
                                                  inst['best_local'], inst['best_dig'], capacity, T)
    return out


ARRAYS = ('orders', 'sizes', 'opts', 'nopts', 'shipping', 'penalty', 'capacity', 'best_local', 'best_dig', 'load',
          'aggregate', 'bacc', 'bali', 'zacc', 'zali')
SCALARS = ('items', 'T', 'Z', 'F', 'n', 'bound', 'divisor')


def save(inst, folder):
    folder.mkdir(parents=True, exist_ok=True)
    for k in ARRAYS:
        np.save(folder / f'{k}.npy', inst[k])
    (folder / 'scalars.json').write_text(json.dumps({k: inst[k] for k in SCALARS}, indent=1))


def load(folder):
    inst = json.loads((folder / 'scalars.json').read_text())
    for k in ARRAYS:
        inst[k] = np.load(folder / f'{k}.npy', mmap_mode='r' if k in ('best_local', 'best_dig') else None)
    return inst

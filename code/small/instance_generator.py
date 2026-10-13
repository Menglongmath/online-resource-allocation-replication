"""The instances of Section 6.1.

Network, costs and inventory rule of the small network of Ma (2023): 10 customer regions and 5 fulfillment centers
(data/small_network), package cost 8.759, shipping cost 0.423 + 0.000541 per mile. Catalogue: the n single-item baskets
and five random baskets of each size from two to five, with weights uniform on [0, 1]. Type 0 of every region is
the period without an order. Actions are all assignments of the basket's items to centers or to being unfulfilled,
restricted to rewards above zero and to item-center pairs with positive inventory.
"""
import csv
from itertools import product
import numpy as np
from common import ROOT
from model import Instance

PACK = 8.759
REGIONS, CENTERS = 10, 5


def geography():
    """Distances (miles) from the regions to the centers, and region probabilities proportional to population."""
    records = []
    for name in ['cities_warmup.csv', 'fulfillment_centers_warmup.csv']:
        with (ROOT / 'data/small_network' / name).open(encoding='utf-8-sig', newline='') as handle:
            records.append(list(csv.reader(handle)))
    cities, fcs = records
    city = np.radians(np.array([[float(r[2]), float(r[3])] for r in cities]))
    fc = np.radians(np.array([[float(r[2]), float(r[3])] for r in fcs]))
    lat = city[:, 0, None] - fc[None, :, 0]
    lon = city[:, 1, None] - fc[None, :, 1]
    hav = np.sin(lat/2)**2 + np.cos(city[:, 0, None])*np.cos(fc[None, :, 0])*np.sin(lon/2)**2
    distance = 2*6371/1.61*np.arcsin(np.sqrt(np.clip(hav, 0., 1.)))
    population = np.array([int(r[4]) for r in cities], dtype=float)
    return distance, population/population.sum()


def size_law(maximum):
    """Order size s = 1, ..., M with probability proportional to 2^-(s-1)."""
    p = 2.**-np.arange(maximum)
    return p/p.sum()


def catalogue(items, seed):
    """Baskets (the empty basket first), their weights, and which centers carry each item (probability 0.75)."""
    rng = np.random.default_rng(np.random.SeedSequence([seed, items, 1]))
    baskets = [()] + [(i,) for i in range(items)]
    weights = [1.] + rng.random(items).tolist()
    for size in range(2, 6):
        seen = set()
        while len(seen) < 5:
            candidate = tuple(sorted(rng.choice(items, size=size, replace=False).tolist()))
            if candidate not in seen:
                seen.add(candidate)
                baskets.append(candidate)
                weights.append(float(rng.random()))
    carry_rng = np.random.default_rng(np.random.SeedSequence([seed, items, 2]))
    carry = carry_rng.random((items, CENTERS)) < .75
    return baskets, np.array(weights), carry


def full_menus(items, maximum, seed, horizon=100000):
    """All actions on the item-center pairs stocked at T = 100,000, and the nearest-center demand rates.

    Type j = basket * REGIONS + region. Returns the menus (type_ptr, action_ptr, resources, reward), the arrival
    probabilities p, and nominal[i, c], the probability that a request contains item i and comes from a region
    whose closest center carrying i is c.
    """
    distance, zone_share = geography()
    all_baskets, all_weights, carry = catalogue(items, seed)
    mask = np.array([len(b) <= maximum for b in all_baskets])
    baskets = [b for b, keep in zip(all_baskets, mask) if keep]
    weights = all_weights[mask]
    sizes = np.array([len(b) for b in baskets])
    query_p = np.zeros(len(baskets))
    query_p[0] = .1
    for s, mass in enumerate(size_law(maximum), 1):
        take = sizes == s
        query_p[take] = .9*mass*weights[take]/weights[take].sum()
    p = (query_p[:, None]*zone_share).ravel()
    sku_zone = np.zeros((items, REGIONS))
    for b, basket in enumerate(baskets):
        for i in basket:
            sku_zone[i] += p[b*REGIONS:(b+1)*REGIONS]
    nominal = np.zeros((items, CENTERS))
    for i in range(items):
        available = np.flatnonzero(carry[i])
        if len(available):
            closest = available[np.argmin(distance[:, available], axis=1)]
            np.add.at(nominal[i], closest, sku_zone[i])
    capacity = inventory(nominal, horizon).reshape(items, CENTERS)
    shipping = .423 + .000541*distance
    penalty = 2*shipping.max(axis=1)
    type_ptr, action_ptr, resources, rewards = [0], [0], [], []
    for basket in baskets:
        options = [[CENTERS]+np.flatnonzero(capacity[i] > 0).tolist() for i in basket]
        assignments = [a for a in product(*options) if any(w < CENTERS for w in a)]
        local_resources = [[i*CENTERS+w for i, w in zip(basket, a) if w < CENTERS] for a in assignments]
        for z in range(REGIONS):
            baseline = 2*PACK + len(basket)*penalty[z] if basket else 0.   # cost of rejecting the order
            for assignment, r in zip(assignments, local_resources):
                wh = np.array([w for w in assignment if w < CENTERS], dtype=np.int32)
                packs, missing = len(np.unique(wh)), len(basket)-len(wh)
                cost = PACK*packs + shipping[z, wh].sum()
                if missing:
                    cost += 2*PACK + missing*penalty[z]
                resources.extend(r)
                action_ptr.append(len(resources))
                rewards.append(baseline-cost)
            type_ptr.append(len(rewards))
    return (np.array(type_ptr, dtype=np.int64), np.array(action_ptr, dtype=np.int64),
            np.array(resources, dtype=np.int32), np.array(rewards), p, nominal)


def inventory(nominal, T):
    """B_T: T omega + 0.5 sqrt(T omega (1 - omega)), rounded, for every item-center pair."""
    return np.rint(T*nominal + .5*np.sqrt(T*nominal*(1-nominal))).astype(np.int64).ravel()


def block_constant(action_resources):
    """L_j of (3.6): largest eigenvalue of W_j (I - 1 1^T / (K_j + 1)) W_j^T, and 1 if the menu is empty."""
    if not action_resources:
        return 1.
    used = np.unique(np.concatenate(action_resources))
    B = np.zeros((len(used), len(action_resources)))
    for column, resources in enumerate(action_resources):
        B[np.searchsorted(used, resources), column] = 1.
    gram = np.einsum('ik,jk->ij', B, B)
    total = B.sum(axis=1)
    tangent = gram-np.outer(total, total)/(len(action_resources)+1)
    return float(np.linalg.eigvalsh(tangent)[-1])*(1+1e-12)


def restrict(menus, capacity):
    """Keep the actions with positive reward that use only item-center pairs with positive inventory."""
    tp, ap, rr, reward = menus
    type_ptr, action_ptr, resources, keep, ell = [0], [0], [], [], []
    for j in range(len(tp)-1):
        local = []
        for action in range(tp[j], tp[j+1]):
            r = rr[ap[action]:ap[action+1]]
            if reward[action] > 0 and np.all(capacity[r] >= 1):
                resources.extend(r.tolist())
                action_ptr.append(len(resources))
                keep.append(action)
                local.append(r)
        ell.append(block_constant(local))
        type_ptr.append(len(keep))
    return (np.asarray(type_ptr, dtype=np.int64), np.asarray(action_ptr, dtype=np.int64),
            np.asarray(resources, dtype=np.int32), reward[np.asarray(keep, dtype=np.int64)], np.asarray(ell))


def build(items, maximum, seed, horizons):
    """The instance (n, M) = (items, maximum) at every horizon: {T: (Instance, B_T)}.

    Rewards are divided by the largest reward over the horizons, which is the largest reward at the longest one.
    """
    tp, ap, rr, reward, p, nominal = full_menus(items, maximum, seed)
    built = {}
    for T in horizons:
        cap = inventory(nominal, T)
        built[T] = (restrict((tp, ap, rr, reward), cap), cap)
    divisor = max(parts[3].max() for parts, _ in built.values())
    return {T: (Instance(t, a, r, w / divisor, p.copy(), ell), cap) for T, ((t, a, r, w, ell), cap) in built.items()}


def demand(instance, entropy, length=100000):
    """A demand sequence of i.i.d. types drawn with probabilities p."""
    rng = np.random.default_rng(np.random.SeedSequence(entropy))
    return rng.choice(instance.n, size=length, p=instance.p).astype(np.int64)

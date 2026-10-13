"""Catalogue, arrival probabilities and inventory of the Section 6.4 instance.

The procedure of Section 6.1 on a general network with Z regions and K centers and n_per baskets of each size:
  catalogue: all single-item baskets and n_per distinct random subsets of each size 2..5, weights U[0,1]
             (seed [base, items, 1]); each center carries each item with probability 0.75 (seed [base, items, 2]);
  arrivals:  no order with probability 0.1; size s with probability proportional to 2^-(s-1); basket by weight;
             region by population;
  inventory: rint(T omega + 0.5 sqrt(T omega (1 - omega))), omega from the closest center carrying the item;
  costs:     package 8.759 per center used, shipping 0.423 + 0.000541 per mile; if an item is unfulfilled,
             2 x 8.759 + (number unfulfilled) x penalty, penalty = 2 x the largest shipping cost to the region.
The menus are not stored; implicit.menu enumerates the menu of a type when it is needed.
"""
import csv
import numpy as np


def geography(cities_csv, fcs_csv):
    """Distances (miles) from regions to centers and region probabilities proportional to population."""
    rows = []
    for name in (cities_csv, fcs_csv):
        with open(name, encoding='utf-8-sig', newline='') as handle:
            rows.append(list(csv.reader(handle)))
    cities, fcs = rows
    city = np.radians(np.array([[float(r[2]), float(r[3])] for r in cities]))
    fc = np.radians(np.array([[float(r[2]), float(r[3])] for r in fcs]))
    lat = city[:, 0, None] - fc[None, :, 0]
    lon = city[:, 1, None] - fc[None, :, 1]
    hav = np.sin(lat / 2) ** 2 + np.cos(city[:, 0, None]) * np.cos(fc[None, :, 0]) * np.sin(lon / 2) ** 2
    distance = 2 * 6371 / 1.61 * np.arcsin(np.sqrt(np.clip(hav, 0., 1.)))
    population = np.array([int(r[4]) for r in cities], dtype=float)
    return distance, population / population.sum()


def catalogue(items, K, n_per, base):
    rng = np.random.default_rng(np.random.SeedSequence([base, items, 1]))
    baskets = [()] + [(i,) for i in range(items)]
    weights = [1.] + rng.random(items).tolist()
    for size in range(2, 6):
        seen = set()
        while len(seen) < n_per:
            candidate = tuple(sorted(rng.choice(items, size=size, replace=False).tolist()))
            if candidate not in seen:
                seen.add(candidate)
                baskets.append(candidate)
                weights.append(float(rng.random()))
    carry_rng = np.random.default_rng(np.random.SeedSequence([base, items, 2]))
    carry = carry_rng.random((items, K)) < .75
    return baskets, np.array(weights), carry


def size_law(maximum):
    p = 2. ** -np.arange(maximum)
    return p / p.sum()


def structure(items, maximum, base, distance, zone_share, n_per):
    """Baskets (as rows of item indices), type probabilities p (type = basket * Z + region), the nearest-center
    demand rates omega (nominal), and the shipping costs and penalties."""
    Z, K = distance.shape
    all_baskets, all_weights, carry = catalogue(items, K, n_per, base)
    mask = np.array([len(b) <= maximum for b in all_baskets])
    baskets = [b for b, keep in zip(all_baskets, mask) if keep]
    weights = all_weights[mask]
    sizes = np.array([len(b) for b in baskets])
    query_p = np.zeros(len(baskets))
    query_p[0] = .1
    for s, mass in enumerate(size_law(maximum), 1):
        take = sizes == s
        query_p[take] = .9 * mass * weights[take] / weights[take].sum()
    p = (query_p[:, None] * zone_share).ravel()
    sku_zone = np.zeros((items, Z))
    for b, basket in enumerate(baskets):
        for i in basket:
            sku_zone[i] += p[b * Z:(b + 1) * Z]
    nominal = np.zeros((items, K))
    for i in range(items):
        available = np.flatnonzero(carry[i])
        if len(available):
            closest = available[np.argmin(distance[:, available], axis=1)]
            np.add.at(nominal[i], closest, sku_zone[i])
    orders = np.full((len(baskets), max(maximum, 1)), -1, dtype=np.int64)
    for b, basket in enumerate(baskets):
        orders[b, :len(basket)] = basket
    shipping = .423 + .000541 * distance
    penalty = 2 * shipping.max(axis=1)
    return dict(sizes=sizes, p=p, nominal=nominal, orders=orders, shipping=shipping, penalty=penalty, Z=Z, K=K)


def capacity_for(nominal, T):
    return np.rint(T * nominal + .5 * np.sqrt(T * nominal * (1. - nominal))).astype(np.int64).ravel()

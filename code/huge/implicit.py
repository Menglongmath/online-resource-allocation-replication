"""Menus generated on demand, and the policies of Section 6.4 (numba kernels that touch only the menu of the
arriving request).

Type t = b * Z + z is basket b in region z; item-center pair (i, c) is resource i * F + c, and opts[i, :nopts[i]]
lists the centers with positive inventory of item i. Each policy draws its demand sequence inside the kernel from
numba's MT19937 seeded with `seed` (alias tables for basket and region), so all policies see the same requests.
"""
import numpy as np
from numba import njit, prange

PACK = 8.759
MAXS = 5


@njit(cache=True)
def menu(b, z, orders, sizes, opts, nopts, shipping, penalty, pack, divisor, reward, digits):
    """The menu of type (b, z): actions with positive reward, in itertools.product order (the options of an item
    are "unfulfilled" and then its stocked centers in ascending order; the last item varies fastest).
    reward[a] = reward / divisor; digits[a, s] = option of item s (0 = unfulfilled, c >= 1 = center
    opts[item, c - 1]). A branch is cut once its partial cost exceeds the cost of rejecting the order, which only
    removes actions with negative reward. Returns the number of actions."""
    q = sizes[b]
    if q == 0:
        return 0
    base = 2 * pack + q * penalty[z]
    dig = np.zeros(MAXS, dtype=np.int64)
    ship = np.zeros(MAXS + 1)
    packs = np.zeros(MAXS + 1, dtype=np.int64)
    miss = np.zeros(MAXS + 1, dtype=np.int64)
    used = np.zeros(MAXS + 1, dtype=np.int64)
    count, s = 0, 0
    while s >= 0:
        i = orders[b, s]
        if dig[s] > nopts[i]:
            s -= 1
            if s >= 0:
                dig[s] += 1
            continue
        d = dig[s]
        if d == 0:
            ship[s + 1] = ship[s]
            packs[s + 1] = packs[s]
            used[s + 1] = used[s]
            miss[s + 1] = miss[s] + 1
        else:
            k = opts[i, d - 1]
            ship[s + 1] = ship[s] + shipping[z, k]
            bit = np.int64(1) << k
            if used[s] & bit:
                packs[s + 1] = packs[s]
                used[s + 1] = used[s]
            else:
                packs[s + 1] = packs[s] + 1
                used[s + 1] = used[s] | bit
            miss[s + 1] = miss[s]
        lower = pack * packs[s + 1] + ship[s + 1]
        if miss[s + 1] > 0:
            lower += 2 * pack + miss[s + 1] * penalty[z]
        if lower > base + 1e-9:                 # every completion has negative reward
            dig[s] += 1
            continue
        if s == q - 1:
            if miss[q] < q:
                cost = pack * packs[q] + ship[q]
                if miss[q] > 0:
                    cost += 2 * pack + miss[q] * penalty[z]
                val = base - cost
                if val > 0:
                    reward[count] = val / divisor
                    for u in range(q):
                        digits[count, u] = dig[u]
                    count += 1
            dig[s] += 1
        else:
            s += 1
            dig[s] = 0
    return count


def menu_bound(nopts):
    return int(np.prod(np.sort(nopts)[-MAXS:] + 1))


@njit(cache=True, parallel=True)
def pass_counts(orders, sizes, opts, nopts, shipping, penalty, pack, Z, bound, chunks):
    """Per type: number of retained actions and largest raw reward."""
    nb = orders.shape[0]
    counts = np.zeros(nb * Z, dtype=np.int64)
    top = np.zeros(nb * Z)
    for c in prange(chunks):
        rew = np.empty(bound)
        dig = np.empty((bound, MAXS), dtype=np.uint8)
        for b in range(c * nb // chunks, (c + 1) * nb // chunks):
            for z in range(Z):
                A = menu(b, z, orders, sizes, opts, nopts, shipping, penalty, pack, 1.0, rew, dig)
                counts[b * Z + z] = A
                m = 0.0
                for a in range(A):
                    m = max(m, rew[a])
                top[b * Z + z] = m
    return counts, top


@njit(cache=True, parallel=True)
def pass_best(orders, sizes, opts, nopts, shipping, penalty, pack, divisor, Z, bound, chunks):
    """Per type: first action with the largest normalized reward (starting solution of Section 6.2 (i))."""
    nb = orders.shape[0]
    best_local = np.full(nb * Z, -1, dtype=np.int32)
    best_dig = np.zeros((nb * Z, MAXS), dtype=np.uint8)
    for c in prange(chunks):
        rew = np.empty(bound)
        dig = np.empty((bound, MAXS), dtype=np.uint8)
        for b in range(c * nb // chunks, (c + 1) * nb // chunks):
            for z in range(Z):
                A = menu(b, z, orders, sizes, opts, nopts, shipping, penalty, pack, divisor, rew, dig)
                t = b * Z + z
                best = 0.
                for a in range(A):
                    if rew[a] > best:
                        best_local[t], best = a, rew[a]
                if best_local[t] >= 0:
                    best_dig[t, :] = dig[best_local[t], :]
    return best_local, best_dig


@njit(cache=True)
def warm_start(orders, sizes, opts, F, Z, best_local, best_dig, capacity, T):
    """Starting solution of Section 6.2 (i) with equal probabilities 1/m: returns Lambda (load) and the vector
    sum_j W_j q_1,j (aggregate)."""
    n = len(best_local)
    load = np.zeros(len(capacity))
    denominator = float(n)
    for t in range(n):
        if best_local[t] < 0:
            continue
        b = t // Z
        weight = 1.
        for s in range(sizes[b]):
            d = best_dig[t, s]
            if d > 0:
                i = orders[b, s]
                load[i * F + opts[i, d - 1]] += weight
    load = load / denominator
    aggregate = np.zeros(len(capacity))
    for t in range(n):
        if best_local[t] < 0:
            continue
        b = t // Z
        fraction = warm_fraction(t, b, orders, sizes, opts, F, best_dig, load, capacity, T)
        weight = 1.
        for s in range(sizes[b]):
            d = best_dig[t, s]
            if d > 0:
                i = orders[b, s]
                aggregate[i * F + opts[i, d - 1]] += weight * fraction
    return load, aggregate


@njit(cache=True)
def warm_fraction(t, b, orders, sizes, opts, F, best_dig, load, capacity, T):
    fraction = 1.
    for s in range(sizes[b]):
        d = best_dig[t, s]
        if d > 0:
            i = orders[b, s]
            r = i * F + opts[i, d - 1]
            fraction = min(fraction, capacity[r] / T / load[r])
    return fraction


def alias_table(prob):
    prob = np.asarray(prob, dtype=np.float64)
    prob = prob / prob.sum()
    n = len(prob)
    scaled = prob * n
    return _vose(scaled)


@njit(cache=True)
def _vose(scaled):
    n = len(scaled)
    accept = np.ones(n)
    alias = np.arange(n)
    small = np.empty(n, dtype=np.int64)
    large = np.empty(n, dtype=np.int64)
    ns, nl = 0, 0
    for i in range(n):
        if scaled[i] < 1.0:
            small[ns] = i
            ns += 1
        else:
            large[nl] = i
            nl += 1
    work = scaled.copy()
    while ns > 0 and nl > 0:
        ns -= 1
        s = small[ns]
        nl -= 1
        l = large[nl]
        accept[s] = work[s]
        alias[s] = l
        work[l] = (work[l] + work[s]) - 1.0
        if work[l] < 1.0:
            small[ns] = l
            ns += 1
        else:
            large[nl] = l
            nl += 1
    return accept, alias


@njit(cache=True)
def draw(bacc, bali, zacc, zali, Z):
    """One request: basket and region from their alias tables (four uniforms)."""
    nb = len(bacc)
    b = min(int(np.random.random() * nb), nb - 1)
    if np.random.random() >= bacc[b]:
        b = bali[b]
    z = min(int(np.random.random() * Z), Z - 1)
    if np.random.random() >= zacc[z]:
        z = zali[z]
    return b * Z + z


@njit(cache=True)
def type_counts(seed, T, bacc, bali, zacc, zali, Z, n):
    """Arrival counts N_j of the sequence drawn with this seed (the same draws as the policies)."""
    np.random.seed(seed)
    out = np.zeros(n, dtype=np.int32)
    for t in range(T):
        out[draw(bacc, bali, zacc, zali, Z)] += 1
    return out


@njit(cache=True)
def simplex(z):
    """Euclidean projection onto the probability simplex."""
    ordered = np.sort(z)[::-1]
    total, threshold = 0., 0.
    for k in range(len(ordered)):
        total += ordered[k]
        threshold = (total - 1) / (k + 1)
        if k == len(ordered) - 1 or threshold >= ordered[k + 1]:
            break
    return np.maximum(z - threshold, 0.)


@njit(cache=True)
def capped_simplex(z):
    """Euclidean projection onto {v >= 0, sum v <= 1}."""
    values = np.maximum(z, 0.0)
    if values.sum() <= 1.0:
        return values
    return simplex(z)


@njit(cache=True)
def tangent_constant(q, A, digits, nopts_s):
    """L_j of (3.6) for one type; the rows are the item-center pairs used by its menu."""
    row = np.full((MAXS, 12), -1, dtype=np.int64)
    for a in range(A):
        for s in range(q):
            d = digits[a, s]
            if d > 0:
                row[s, d] = 0
    nr = 0
    for s in range(q):
        for d in range(1, nopts_s[s] + 1):
            if row[s, d] == 0:
                row[s, d] = nr
                nr += 1
    G = np.zeros((nr, nr))
    idx = np.empty(MAXS, dtype=np.int64)
    for a in range(A):
        k = 0
        for s in range(q):
            d = digits[a, s]
            if d > 0:
                idx[k] = row[s, d]
                k += 1
        for u in range(k):
            for v in range(k):
                G[idx[u], idx[v]] += 1.0
    total = np.diag(G).copy()
    M = G - np.outer(total, total) / (A + 1)
    return np.linalg.eigvalsh(M)[-1] * (1 + 1e-12)


@njit(cache=True)
def _grow_i(a, n):
    out = np.empty(max(n, 2 * len(a)), dtype=a.dtype)
    out[:len(a)] = a
    return out


@njit(cache=True)
def _grow_f(a, n):
    out = np.empty(max(n, 2 * len(a)), dtype=a.dtype)
    out[:len(a)] = a
    return out


@njit(cache=True)
def ours(orders, sizes, opts, nopts, shipping, penalty, pack, divisor, Z, F, n, best_local, best_dig, load,
         aggregate0, capacity, T, gamma, seed, bacc, bali, zacc, zali, bound):
    """Our policy with forecast pi_t (unknown probabilities), as on the small network (code/small/our_policy.py):
    kappa_t = min{t, T - t + 1}^(-gamma), L_j of (3.6), starting solution of Section 6.2 (i), fallback rule.
    The fractions of a type are stored sparsely, from its first arrival on; before that they equal the starting
    solution. Returns the total reward (normalized units) and the remaining inventory."""
    np.random.seed(seed)
    aggregate = aggregate0.copy()
    remaining = capacity.copy()
    slot = np.full(n, -1, dtype=np.int64)
    r_start = np.empty(1 << 16, dtype=np.int64)
    r_len = np.empty(1 << 16, dtype=np.int64)
    r_cap = np.empty(1 << 16, dtype=np.int64)
    r_ell = np.empty(1 << 16)
    r_H = np.empty(1 << 16, dtype=np.int64)
    pool_i = np.empty(1 << 20, dtype=np.int32)
    pool_v = np.empty(1 << 20)
    nrec, npool = 0, 0
    rew = np.empty(bound)
    dig = np.empty((bound, MAXS), dtype=np.uint8)
    old = np.zeros(bound)
    over = np.zeros((MAXS, 12))
    nopts_s = np.zeros(MAXS, dtype=np.int64)
    revenue = 0.0
    for t in range(T):
        periods = T - t
        tt = t + 1.0
        kappa = min(tt, T - tt + 1.0) ** (-gamma)
        scale = 1.0 / (t + n)
        j = draw(bacc, bali, zacc, zali, Z)
        b, z = j // Z, j % Z
        q = sizes[b]
        A = menu(b, z, orders, sizes, opts, nopts, shipping, penalty, pack, divisor, rew, dig)
        if A == 0:
            continue
        sl = slot[j]
        for a in range(A):
            old[a] = 0.0
        if sl < 0:
            H = 0
            if best_local[j] >= 0:
                old[best_local[j]] = warm_fraction(j, b, orders, sizes, opts, F, best_dig, load, capacity, T)
            for s in range(q):
                nopts_s[s] = nopts[orders[b, s]]
            ell = tangent_constant(q, A, dig, nopts_s)
        else:
            H = r_H[sl]
            ell = r_ell[sl]
            for u in range(r_start[sl], r_start[sl] + r_len[sl]):
                old[pool_i[u]] = pool_v[u]
        weight = (H + 1.0) * scale                      # pi_t,j = (H_j + 1) / (t - 1 + m)
        for s in range(q):
            i = orders[b, s]
            for d in range(1, nopts[i] + 1):
                r = i * F + opts[i, d - 1]
                over[s, d] = max(aggregate[r] * scale - min(remaining[r] / periods, 1.0), 0.0)
        accepting = np.empty(A)
        for a in range(A):
            gradient = -kappa * rew[a]
            for s in range(q):
                d = dig[a, s]
                if d > 0:
                    gradient += over[s, d]
            accepting[a] = old[a] - gradient / (weight * ell)
        accepting = capped_simplex(accepting)
        reject_value = max(0.0, 1.0 - accepting.sum())
        unnormalized = H + 1.0
        for a in range(A):
            if accepting[a] != old[a]:
                change = unnormalized * (accepting[a] - old[a])
                for s in range(q):
                    d = dig[a, s]
                    if d > 0:
                        i = orders[b, s]
                        aggregate[i * F + opts[i, d - 1]] += change
        best, selected = reject_value, -1
        for a in range(A):
            if accepting[a] > best:
                best, selected = accepting[a], a
        if selected >= 0 and not _fits(selected, q, b, orders, opts, F, dig, remaining):
            # fallback rule: prices lambda_t = (Wy - c_t)_+ / kappa_t at the updated allocation
            for s in range(q):
                i = orders[b, s]
                for d in range(1, nopts[i] + 1):
                    r = i * F + opts[i, d - 1]
                    over[s, d] = max(aggregate[r] * scale - min(remaining[r] / periods, 1.0), 0.0) / kappa
            best, selected = 0.0, -1
            for a in range(A):
                if not _fits(a, q, b, orders, opts, F, dig, remaining):
                    continue
                score = rew[a]
                for s in range(q):
                    d = dig[a, s]
                    if d > 0:
                        score -= over[s, d]
                if score > best:
                    best, selected = score, a
        if selected >= 0:
            for s in range(q):
                d = dig[selected, s]
                if d > 0:
                    i = orders[b, s]
                    remaining[i * F + opts[i, d - 1]] -= 1
            revenue += rew[selected]
        H += 1                                          # the arrival is counted after the decision
        for a in range(A):
            v = accepting[a]
            if v != 0.0:
                for s in range(q):
                    d = dig[a, s]
                    if d > 0:
                        i = orders[b, s]
                        aggregate[i * F + opts[i, d - 1]] += v
        support = 0
        for a in range(A):
            if accepting[a] != 0.0:
                support += 1
        if sl < 0:
            if nrec == len(r_start):
                r_start, r_len, r_cap = _grow_i(r_start, 0), _grow_i(r_len, 0), _grow_i(r_cap, 0)
                r_ell, r_H = _grow_f(r_ell, 0), _grow_i(r_H, 0)
            sl = nrec
            nrec += 1
            slot[j] = sl
            r_ell[sl] = ell
            r_cap[sl] = 0
            r_start[sl] = 0
        r_H[sl] = H
        if support > r_cap[sl]:
            need = npool + max(support, 2)
            if need > len(pool_i):
                pool_i, pool_v = _grow_i(pool_i, need), _grow_f(pool_v, need)
            r_start[sl] = npool
            r_cap[sl] = max(support, 2)
            npool = need
        u = r_start[sl]
        for a in range(A):
            if accepting[a] != 0.0:
                pool_i[u] = a
                pool_v[u] = accepting[a]
                u += 1
        r_len[sl] = support
    return revenue, remaining


@njit(cache=True)
def _fits(a, q, b, orders, opts, F, dig, remaining):
    for s in range(q):
        d = dig[a, s]
        if d > 0:
            i = orders[b, s]
            if remaining[i * F + opts[i, d - 1]] < 1:
                return False
    return True


@njit(cache=True)
def _prices(q, b, orders, opts, nopts, F, base, rate, clock, price):
    for s in range(q):
        i = orders[b, s]
        for d in range(1, nopts[i] + 1):
            r = i * F + opts[i, d - 1]
            price[s, d] = max(0.0, base[r] - rate[r] * clock)


@njit(cache=True)
def _advance(a, q, b, orders, opts, F, dig, base, rate, clock, step):
    next_clock = clock + step
    if a >= 0:
        for s in range(q):
            d = dig[a, s]
            if d > 0:
                i = orders[b, s]
                r = i * F + opts[i, d - 1]
                price = max(0.0, base[r] - rate[r] * clock)
                price = max(0.0, price + step * (1.0 - rate[r]))
                base[r] = price + rate[r] * next_clock
    return next_clock


@njit(cache=True)
def price_policy(orders, sizes, opts, nopts, shipping, penalty, pack, divisor, Z, F, capacity, T, mode,
                 boundaries, step_constant, seed, bacc, bali, zacc, zali, bound):
    """mode 0: DMD, step T^(-1/2); mode 2: Ma et al. (2025), Algorithm 5 without the regularizer (epoch boundaries
    in `boundaries`, step step_constant / (t - l + 1)). Both with the fallback rule (code/small/price_policies.py).
    Returns the total reward (normalized units) and the remaining inventory."""
    np.random.seed(seed)
    rate = capacity / T
    C = capacity.copy()
    base = np.zeros(len(C))
    clock = 0.0
    eta_dmd = T ** (-0.5)
    nxt, ell = 0, 0
    rew = np.empty(bound)
    dig = np.empty((bound, MAXS), dtype=np.uint8)
    price = np.zeros((MAXS, 12))
    revenue = 0.0
    for t in range(T):
        j = draw(bacc, bali, zacc, zali, Z)
        b, z = j // Z, j % Z
        q = sizes[b]
        A = menu(b, z, orders, sizes, opts, nopts, shipping, penalty, pack, divisor, rew, dig)
        _prices(q, b, orders, opts, nopts, F, base, rate, clock, price)
        best, a = 0.0, -1
        for x in range(A):
            score = rew[x]
            for s in range(q):
                d = dig[x, s]
                if d > 0:
                    score -= price[s, d]
            if score > best:
                best, a = score, x
        if a >= 0:
            executed = -1
            if _fits(a, q, b, orders, opts, F, dig, C):
                executed = a
            else:
                best_s = 0.0
                for x in range(A):
                    score = rew[x]
                    for s in range(q):
                        d = dig[x, s]
                        if d > 0:
                            score -= price[s, d]
                    if score > best_s and _fits(x, q, b, orders, opts, F, dig, C):
                        best_s, executed = score, x
            if executed >= 0:
                for s in range(q):
                    d = dig[executed, s]
                    if d > 0:
                        i = orders[b, s]
                        C[i * F + opts[i, d - 1]] -= 1
                revenue += rew[executed]
        if mode == 2:
            tt = t + 1
            if nxt < len(boundaries) and boundaries[nxt] == tt:
                for r in range(len(C)):
                    base[r] = max(0.0, base[r] - rate[r] * clock)
                    rate[r] = C[r] / (T - tt)
                clock = 0.0
                ell = tt
                nxt += 1
            clock = _advance(a, q, b, orders, opts, F, dig, base, rate, clock, step_constant / (tt - ell + 1))
        else:
            clock = _advance(a, q, b, orders, opts, F, dig, base, rate, clock, eta_dmd)
    return revenue, C

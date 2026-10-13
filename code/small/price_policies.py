"""The price-based policies of Section 6.2.

For an arriving request of type j, each policy selects the action with the largest positive r_ja - w_ja^T lambda at
the current prices (rejection if none) and updates the prices with the selected action. With fallback = True, a
selected action that does not fit is replaced by the action that fits with the largest positive r_ja - w_ja^T lambda
(rejection if none); with fallback = False the request is rejected, as in the original algorithms.

DMD and Ma et al. (2025) store the prices lazily: the price of resource i is max(0, base_i - rate_i * clock), and the
projected update lambda <- [lambda + step (w - rate)]_+ only touches the resources of the selected action.
"""
import numpy as np
from numba import njit


@njit(cache=True)
def fits(a, ap, resources, C):
    for k in range(ap[a], ap[a + 1]):
        if C[resources[k]] < 1:
            return False
    return True


@njit(cache=True)
def take(a, ap, resources, C):
    for k in range(ap[a], ap[a + 1]):
        C[resources[k]] -= 1


@njit(cache=True)
def select(j, tp, ap, resources, reward, base, rate, clock, C):
    """The selected action, and the action that fits with the largest positive score (-1 = rejection)."""
    best, action, best_s, action_s = 0.0, -1, 0.0, -1
    for a in range(tp[j], tp[j + 1]):
        score = reward[a]
        for k in range(ap[a], ap[a + 1]):
            i = resources[k]
            score -= max(0.0, base[i] - rate[i] * clock)
        if score > best:
            best, action = score, a
        if score > best_s and fits(a, ap, resources, C):
            best_s, action_s = score, a
    return action, action_s


@njit(cache=True)
def advance(action, ap, resources, base, rate, clock, step):
    """lambda <- [lambda + step (w_action - rate)]_+ in the lazy representation."""
    next_clock = clock + step
    if action >= 0:
        for k in range(ap[action], ap[action + 1]):
            i = resources[k]
            price = max(0.0, base[i] - rate[i] * clock)
            price = max(0.0, price + step * (1.0 - rate[i]))
            base[i] = price + rate[i] * next_clock
    return next_clock


@njit(cache=True)
def execute(a, a_s, ap, resources, reward, C, fallback):
    """Executed action and its reward."""
    if a < 0:
        return -1, 0.0
    if fits(a, ap, resources, C):
        take(a, ap, resources, C)
        return a, reward[a]
    if fallback and a_s >= 0:
        take(a_s, ap, resources, C)
        return a_s, reward[a_s]
    return -1, 0.0


@njit(cache=True)
def dmd(tp, ap, resources, reward, arrivals, capacity, step, fallback):
    """Balseiro, Lu and Mirrokni (2023), Algorithm 1, Euclidean mirror map: prices start at zero and
    lambda <- [lambda + step (w_ja - B_T / T)]_+; the paper uses step = T^(-1/2)."""
    T = len(arrivals)
    rate = capacity / T
    C = capacity.copy()
    q = np.zeros(len(C))
    clock = 0.0
    decisions = np.full(T, -1, dtype=np.int32)
    revenue = 0.0
    for t in range(T):
        a, a_s = select(arrivals[t], tp, ap, resources, reward, q, rate, clock, C)
        decisions[t], val = execute(a, a_s, ap, resources, reward, C, fallback)
        revenue += val
        clock = advance(a, ap, resources, q, rate, clock, step)
    return revenue, C, decisions


@njit(cache=True)
def ma2025(tp, ap, resources, reward, arrivals, capacity, boundary, step_constant, fallback):
    """Ma et al. (2025), Algorithm 5 without the regularizer: one price update per period with step
    step_constant / (t - l + 1); at an epoch boundary (boundary[t]) the target becomes C_t / (T - t) and l = t."""
    T = len(arrivals)
    C = capacity.copy()
    rate = capacity / T
    base = np.zeros(len(C))
    clock = 0.0
    ell = 0
    decisions = np.full(T, -1, dtype=np.int32)
    revenue = 0.0
    for tt in range(T):
        t = tt + 1
        a, a_s = select(arrivals[tt], tp, ap, resources, reward, base, rate, clock, C)
        decisions[tt], val = execute(a, a_s, ap, resources, reward, C, fallback)
        revenue += val
        if boundary[t]:
            for i in range(len(C)):
                base[i] = max(0.0, base[i] - rate[i] * clock)
                rate[i] = C[i] / (T - t)
            clock = 0.0
            ell = t
        clock = advance(a, ap, resources, base, rate, clock, step_constant / (t - ell + 1))
    return revenue, C, decisions


@njit(cache=True)
def he(tp, ap, resources, reward, arrivals, capacity, initial, rate, V, fallback):
    """He et al. (2025), Algorithm 1: prices start at the fluid-LP dual and lambda <- [lambda + (w_ja - B_T / T) / V]_+.

    The term -rate / V of every period is applied lazily from the period of the last change of each price.
    """
    C, price = capacity.copy(), initial.copy()
    last = np.zeros(len(C), dtype=np.int64)
    decisions = np.full(len(arrivals), -1, dtype=np.int32)
    revenue, clock = 0.0, 0
    for t in range(len(arrivals)):
        j = arrivals[t]
        selected, best, servable, best_s = -1, 0.0, -1, 0.0
        for a in range(tp[j], tp[j + 1]):
            surplus = reward[a]
            for k in range(ap[a], ap[a + 1]):
                i = resources[k]
                surplus -= max(price[i] - (clock - last[i]) * rate[i] / V, 0.0)
            if surplus > best:
                selected, best = a, surplus
            if surplus > best_s and fits(a, ap, resources, C):
                servable, best_s = a, surplus
        if selected >= 0:
            for k in range(ap[selected], ap[selected + 1]):
                i = resources[k]
                current = max(price[i] - (clock - last[i]) * rate[i] / V, 0.0)
                price[i] = max(current + (1 - rate[i]) / V, 0.0)
                last[i] = clock + 1
        clock += 1
        if selected >= 0:
            if fits(selected, ap, resources, C):
                take(selected, ap, resources, C)
                revenue += reward[selected]
                decisions[t] = selected
            elif fallback and servable >= 0:
                take(servable, ap, resources, C)
                revenue += reward[servable]
                decisions[t] = servable
    return revenue, C, decisions

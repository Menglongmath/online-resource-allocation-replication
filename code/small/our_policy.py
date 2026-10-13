"""Our policy as run in Section 6: Algorithm 1 with I_t = J_t, kappa_t = min{t, T - t + 1}^(-gamma), the constants
L_j of (3.6), the starting solution of Section 6.2 (i) or the LP start, and the fallback rule of Section 6.2 (ii).

State. The fractions q_j = (q_j0, q_j1, ..., q_jK_j) of type j (rejection first) are stored in q at the offsets
type_ptr[j] + j, ..., type_ptr[j + 1] + j. The vector Wy = sum_j eta_j W_j q_j is kept as aggregate * scale:
  known p (forecast eta_t = p):      aggregate = sum_j p_j W_j q_j,           scale = 1;
  unknown p (forecast pi_t (2.1)):   aggregate = sum_j (H_j + 1) W_j q_j,     scale = 1 / (t - 1 + m),
where H_j counts the past arrivals of type j. The target c_t = C_t / s_t enters as min{c_t, 1}; this does not change
the update because every coordinate of Wy is at most one.
"""
import numpy as np
from numba import njit


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
def block_update(type_id, q, aggregate, scale, weight, remaining, periods, kappa,
                 ell, type_ptr, action_ptr, resources, reward):
    """The projected gradient step of Algorithm 1 on the block of type_id, in fractions; weight = eta_j.

    y_j <- Proj(y_j + (kappa / L_j) grad_j f), with grad_j f = r_j - W_j^T (Wy - c)_+ / kappa, written for q_j = y_j / eta_j.
    Returns the new fractions, rejection first.
    """
    start, stop = type_ptr[type_id], type_ptr[type_id + 1]
    accepting = q[start + type_id + 1:stop + type_id + 1].copy()
    for action in range(start, stop):
        gradient = -kappa * reward[action]
        for pos in range(action_ptr[action], action_ptr[action + 1]):
            resource = resources[pos]
            gradient += max(aggregate[resource] * scale -
                            min(remaining[resource] / periods, 1.0), 0.0)
        accepting[action - start] -= gradient / (weight * ell[type_id])
    accepting = capped_simplex(accepting)
    values = np.empty(stop - start + 1)
    values[0] = max(0.0, 1.0 - accepting.sum())
    values[1:] = accepting
    return values


@njit(cache=True)
def commit(type_id, values, q, aggregate, unnormalized_weight, type_ptr, action_ptr, resources):
    """Store the new fractions of type_id and update aggregate."""
    start, stop = type_ptr[type_id], type_ptr[type_id + 1]
    for action in range(start, stop):
        difference = values[action - start + 1] - q[action + type_id + 1]
        change = unnormalized_weight * difference
        for pos in range(action_ptr[action], action_ptr[action + 1]):
            aggregate[resources[pos]] += change
    q[start + type_id:stop + type_id + 1] = values


@njit(cache=True)
def recompute_aggregate(q, weights, type_ptr, action_ptr, resources, resource_count):
    """sum_j weights_j W_j q_j."""
    value = np.zeros(resource_count)
    for type_id in range(len(weights)):
        for action in range(type_ptr[type_id], type_ptr[type_id + 1]):
            for pos in range(action_ptr[action], action_ptr[action + 1]):
                value[resources[pos]] += weights[type_id] * q[action + type_id + 1]
    return value


@njit(cache=True)
def fits(action, action_ptr, resources, remaining):
    for pos in range(action_ptr[action], action_ptr[action + 1]):
        if remaining[resources[pos]] < 1:
            return False
    return True


@njit(cache=True)
def heuristic_start(type_ptr, action_ptr, resources, reward, p, capacity, T, known):
    """Starting solution of Section 6.2 (i).

    Each type puts the fraction min{1, min_i (B_T,i / T) / Lambda_i} on its highest-reward action and the rest on
    rejection; Lambda is the per-period consumption when every type uses that action, computed with p if known and
    with equal probabilities otherwise. Returns q and aggregate.
    """
    n, A = len(p), len(reward)
    q = np.zeros(A + n)
    aggregate = np.zeros(len(capacity))
    selected = np.full(n, -1, dtype=np.int64)
    denominator = 1. if known else float(n)
    for j in range(n):
        q[type_ptr[j] + j] = 1.
        best = 0.
        for a in range(type_ptr[j], type_ptr[j + 1]):
            ok = True
            for k in range(action_ptr[a], action_ptr[a + 1]):
                if capacity[resources[k]] < 1:
                    ok = False
            if ok and reward[a] > best:
                selected[j], best = a, reward[a]
        if selected[j] >= 0:
            weight = p[j] if known else 1.
            a = selected[j]
            for k in range(action_ptr[a], action_ptr[a + 1]):
                aggregate[resources[k]] += weight
    load = aggregate / denominator
    aggregate[:] = 0.
    for j in range(n):
        a = selected[j]
        if a < 0:
            continue
        fraction = 1.
        for k in range(action_ptr[a], action_ptr[a + 1]):
            i = resources[k]
            fraction = min(fraction, capacity[i] / T / load[i])
        q[type_ptr[j] + j] = 1 - fraction
        q[a + j + 1] = fraction
        weight = p[j] if known else 1.
        for k in range(action_ptr[a], action_ptr[a + 1]):
            aggregate[resources[k]] += weight * fraction
    return q, aggregate


def kappa_schedule(T, gamma):
    """kappa_t = min{t, T - t + 1}^(-gamma) for t = 1, ..., T (0-based array)."""
    t = np.arange(1, T + 1, dtype=np.float64)
    return np.minimum(t, T - t + 1.0) ** (-gamma)


@njit(cache=True)
def run(type_ptr, action_ptr, resources, reward, p, ell, arrivals, capacity, kappas, initial_q, initial_aggregate,
        known):
    """One demand sequence. Returns the total reward, the remaining inventory and the executed actions (-1 = reject).

    Fallback rule: if the selected action does not fit, form lambda_t = (Wy - c_t)_+ / kappa_t at the updated
    allocation and execute the action that fits with the largest positive r_ja - w_ja^T lambda_t; reject if none.
    """
    horizon, type_count = len(arrivals), len(p)
    q, aggregate = initial_q.copy(), initial_aggregate.copy()
    counts = np.zeros(type_count, dtype=np.int64)
    remaining = capacity.copy()
    decisions = np.full(horizon, -1, dtype=np.int32)
    revenue = 0.0
    for t in range(horizon):
        periods = horizon - t
        if known:
            scale = 1.0
        else:
            scale = 1.0 / (t + type_count)
        type_id = arrivals[t]
        if known:
            weight = p[type_id]
        else:
            weight = (counts[type_id] + 1.0) * scale
        values = block_update(type_id, q, aggregate, scale, weight, remaining, periods, kappas[t], ell,
                              type_ptr, action_ptr, resources, reward)
        if known:
            commit(type_id, values, q, aggregate, p[type_id], type_ptr, action_ptr, resources)
        else:
            commit(type_id, values, q, aggregate, counts[type_id] + 1.0, type_ptr, action_ptr, resources)
        selected = np.argmax(values) - 1                  # smallest index among ties; -1 = rejection
        if selected >= 0 and not fits(type_ptr[type_id] + selected, action_ptr, resources, remaining):
            start = type_ptr[type_id]
            prices = np.maximum(aggregate * scale - np.minimum(remaining / periods, 1.0), 0.0) / kappas[t]
            best, selected = 0.0, -1
            for k in range(type_ptr[type_id + 1] - start):
                action = start + k
                if not fits(action, action_ptr, resources, remaining):
                    continue
                score = reward[action]
                for pos in range(action_ptr[action], action_ptr[action + 1]):
                    score -= prices[resources[pos]]
                if score > best:
                    best, selected = score, k
        if selected >= 0:
            action = type_ptr[type_id] + selected
            for pos in range(action_ptr[action], action_ptr[action + 1]):
                remaining[resources[pos]] -= 1
            revenue += reward[action]
            decisions[t] = action
        if not known:                                     # H_j <- H_j + 1 after the decision
            counts[type_id] += 1
            start, stop = type_ptr[type_id], type_ptr[type_id + 1]
            for action in range(start, stop):
                value = q[action + type_id + 1]
                for pos in range(action_ptr[action], action_ptr[action + 1]):
                    aggregate[resources[pos]] += value
    return revenue, remaining, decisions

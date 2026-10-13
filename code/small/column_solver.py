"""The LP max {r^T x : W x <= capacity, sum_a x_ja <= demand_j, x >= 0} by column generation with HiGHS.

Used for the hindsight LP Phi(B_T, N) (demand = arrival counts N) and for the fluid LP (demand = T p). Each solve
prices all actions of all menus and stops only when the dual bound certifies optimality over the full menus.
"""
import numpy as np
import highspy
from numba import njit


@njit(cache=True)
def pricing(tp, ap, rr, reward, price):
    """For each type, the largest positive reduced reward r_ja - w_ja^T price and its action (-1 if none)."""
    value = np.zeros(len(tp)-1)
    best = np.full(len(tp)-1, -1, dtype=np.int64)
    for j in range(len(best)):
        for a in range(tp[j], tp[j+1]):
            score = reward[a]
            for k in range(ap[a], ap[a+1]):
                score -= price[rr[k]]
            if score > value[j]:
                best[j], value[j] = a, score
    return value, best


@njit(cache=True)
def columns(acts, action_types, ap, rr, d):
    lengths = np.empty(len(acts), dtype=np.int64)
    for k, a in enumerate(acts):
        lengths[k] = ap[a+1]-ap[a]+1
    start = np.empty(len(acts)+1, dtype=np.int64)
    start[0] = 0
    for k in range(len(acts)):
        start[k+1] = start[k]+lengths[k]
    rows = np.empty(start[-1], dtype=np.int32)
    for k, a in enumerate(acts):
        n = ap[a+1]-ap[a]
        rows[start[k]:start[k]+n] = rr[ap[a]:ap[a+1]]
        rows[start[k]+n] = d+action_types[a]
    return start, rows


class ColumnSolver:
    def __init__(self, model, d):
        self.model = model
        self.d = d
        self.types = np.repeat(np.arange(model.n, dtype=np.int32), np.diff(model.type_ptr))
        self.included = np.zeros(model.actions, dtype=bool)
        self.acts = np.empty(0, dtype=np.int64)
        self.h = highspy.Highs()
        for key, value in [('output_flag', False), ('log_to_console', False),
                           ('threads', 1), ('parallel', 'off'), ('solver', 'simplex'),
                           ('primal_feasibility_tolerance', 1e-8),
                           ('dual_feasibility_tolerance', 1e-8)]:
            assert self.h.setOptionValue(key, value) == highspy.HighsStatus.kOk
        lp = highspy.HighsLp()
        lp.num_row_ = self.d+model.n
        lp.num_col_ = 0
        lp.sense_ = highspy.ObjSense.kMaximize
        lp.col_cost_, lp.col_lower_, lp.col_upper_ = [], [], []
        lp.row_lower_ = np.full(lp.num_row_, -highspy.kHighsInf)
        lp.row_upper_ = np.zeros(lp.num_row_)
        lp.a_matrix_.format_ = highspy.MatrixFormat.kColwise
        lp.a_matrix_.start_ = [0]
        assert self.h.passModel(lp) == highspy.HighsStatus.kOk

    def add(self, acts):
        if len(acts) == 0:
            return
        acts = np.unique(acts)
        acts = acts[~self.included[acts]]
        if len(acts) == 0:
            return
        m = self.model
        start, rows = columns(acts, self.types, m.action_ptr, m.resources, self.d)
        status = self.h.addCols(len(acts), m.reward[acts], np.zeros(len(acts)),
                               np.full(len(acts), highspy.kHighsInf), len(rows),
                               start, rows, np.ones(len(rows)))
        assert status == highspy.HighsStatus.kOk
        self.acts = np.r_[self.acts, acts]
        self.included[acts] = True

    def solve(self, capacity, demand):
        """Returns x, resource prices, type prices and a certificate (objective, dual bound, gap, violation)."""
        m = self.model
        rhs = np.r_[capacity, demand].astype(float)
        assert np.isfinite(rhs).all() and rhs.min() >= 0
        self.h.changeRowsBounds(len(rhs), np.arange(len(rhs), dtype=np.int32),
                                np.full(len(rhs), -highspy.kHighsInf), rhs)
        if not len(self.acts):
            _, best = pricing(m.type_ptr, m.action_ptr, m.resources, m.reward, np.zeros(self.d))
            self.add(best[(demand > 0) & (best >= 0)])
        if not len(self.acts):
            price = np.zeros(self.d)
            type_price, _ = pricing(m.type_ptr, m.action_ptr, m.resources, m.reward, price)
            assert float(demand @ type_price) == 0.0
            info = dict(status='optimal', objective=0.0, dual_bound=0.0, duality_gap=0.0, max_violation=0.0)
            return np.zeros(m.actions), price, type_price, info
        while True:
            self.h.run()
            assert self.h.getModelStatus() == highspy.HighsModelStatus.kOptimal
            sol = self.h.getSolution()
            small_x = np.asarray(sol.col_value)
            dual = np.maximum(0.0, np.asarray(sol.row_dual))
            price = dual[:self.d].copy()
            # a resource without inventory may carry any nonnegative price; this also prices out its actions
            price[capacity == 0] = np.maximum(price[capacity == 0], m.reward.max())
            type_price, best = pricing(m.type_ptr, m.action_ptr, m.resources, m.reward, price)
            objective = float(m.reward[self.acts] @ small_x)
            upper = float(capacity @ price + demand @ type_price)
            gap = upper-objective
            if -1e-6 <= gap <= 1e-8*max(1.0, objective):
                break
            improve = (demand > 0) & (best >= 0) & (type_price > dual[self.d:]+1e-8)
            new = best[improve]
            new = new[~self.included[new]]
            if not len(new):
                raise RuntimeError(f'column generation stalled: gap={gap}, objective={objective}')
            self.add(new)
        x = np.zeros(m.actions)
        x[self.acts] = small_x
        usage = np.bincount(m.resources, weights=np.repeat(x, np.diff(m.action_ptr)), minlength=self.d)
        type_use = np.bincount(self.types, weights=x, minlength=m.n)
        violation = max(0.0, float(np.max(usage-capacity)), float(np.max(type_use-demand)), float(-x.min()))
        assert violation <= 2e-6
        info = dict(status='optimal', objective=objective, dual_bound=upper, duality_gap=gap, max_violation=violation)
        return x, price, type_price, info

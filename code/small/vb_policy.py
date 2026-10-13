"""Vera and Banerjee (2021), Algorithm 4, with known arrival probabilities (Section 6.2).

In each period t the policy solves the fluid LP with the remaining inventory C_t and demand s_t p, s_t = T - t + 1,
and takes the action of the arriving type with the largest LP allocation, including rejection, whose allocation is
s_t p_j minus the allocations of the type's actions; ties favor rejection. HiGHS dual simplex solves one LP per period,
starting from the basis of the previous period.
"""
import numpy as np
import highspy
from scipy.sparse import csc_matrix, vstack


class EveryPeriodLP:
    def __init__(self, model, d):
        capacity = csc_matrix((np.ones(len(model.resources)), model.resources, model.action_ptr),
                              shape=(d, model.actions))
        types = np.repeat(np.arange(model.n, dtype=np.int32), np.diff(model.type_ptr))
        demand = csc_matrix((np.ones(model.actions), types, np.arange(model.actions + 1)),
                            shape=(model.n, model.actions))
        matrix = vstack([capacity, demand], format='csc').tocsc()
        self.rows = d + model.n
        self.ids = np.arange(self.rows, dtype=np.int32)
        self.lower = np.full(self.rows, -highspy.kHighsInf)
        self.h = highspy.Highs()
        for name, value in [
            ('output_flag', False), ('threads', 1), ('parallel', 'off'),
            ('solver', 'simplex'), ('simplex_strategy', 1), ('presolve', 'off'),
            ('primal_feasibility_tolerance', 1e-7),
            ('dual_feasibility_tolerance', 1e-7), ('random_seed', 0),
        ]:
            assert self.h.setOptionValue(name, value) == highspy.HighsStatus.kOk
        lp = highspy.HighsLp()
        lp.num_col_, lp.num_row_ = model.actions, self.rows
        lp.sense_ = highspy.ObjSense.kMaximize
        lp.col_cost_ = np.asarray(model.reward, dtype=float)
        lp.col_lower_ = np.zeros(model.actions)
        lp.col_upper_ = np.full(model.actions, highspy.kHighsInf)
        lp.row_lower_ = self.lower
        lp.row_upper_ = np.zeros(self.rows)
        lp.a_matrix_.format_ = highspy.MatrixFormat.kColwise
        lp.a_matrix_.start_ = matrix.indptr
        lp.a_matrix_.index_ = matrix.indices
        lp.a_matrix_.value_ = matrix.data
        assert self.h.passModel(lp) == highspy.HighsStatus.kOk
        self.solves = 0

    def solve(self, capacity, demand):
        rhs = np.concatenate((capacity, demand)).astype(float, copy=False)
        assert self.h.changeRowsBounds(self.rows, self.ids, self.lower, rhs) == highspy.HighsStatus.kOk
        status = self.h.run()                     # the basis of the previous solve is kept as a warm start
        self.solves += 1
        assert status == highspy.HighsStatus.kOk
        assert self.h.getModelStatus() == highspy.HighsModelStatus.kOptimal
        return np.asarray(self.h.getSolution().col_value, dtype=float)


def run_vb(model, arrivals, capacity):
    """One demand sequence. Returns the total reward, the remaining inventory and the executed actions (-1 = reject)."""
    T = len(arrivals)
    C = np.asarray(capacity, dtype=float).copy()
    solver = EveryPeriodLP(model, len(C))
    decisions = np.full(T, -1, dtype=np.int32)
    revenue = 0.
    for t in range(T):
        demand = (T - t) * model.p
        x = solver.solve(C, demand)
        j = int(arrivals[t])
        lo, hi = (int(v) for v in model.type_ptr[j:j + 2])
        if hi > lo:
            block = x[lo:hi]
            residual = max(0., float(demand[j] - np.sum(block)))     # allocation of rejection
            local = int(np.argmax(block))
            if float(block[local]) > residual:
                action = lo + local
                resources = model.resources[model.action_ptr[action]:model.action_ptr[action + 1]]
                if np.all(C[resources] >= 1):
                    C[resources] -= 1
                    revenue += float(model.reward[action])
                    decisions[t] = action
    assert solver.solves == T
    return revenue, C, decisions

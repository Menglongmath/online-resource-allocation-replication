"""An instance: request types, their action menus, rewards, arrival probabilities and the constants L_j.

Type j (a basket-region pair) has the actions type_ptr[j], ..., type_ptr[j + 1] - 1; rejection is not stored.
Action a uses one unit of each resource (item-center pair) in resources[action_ptr[a]:action_ptr[a + 1]].
Rewards are divided by the largest reward of the instance. ell[j] is L_j of (3.6).
"""
from dataclasses import dataclass, fields
import numpy as np


@dataclass
class Instance:
    type_ptr: np.ndarray
    action_ptr: np.ndarray
    resources: np.ndarray
    reward: np.ndarray
    p: np.ndarray
    ell: np.ndarray

    @property
    def n(self):
        return len(self.p)

    @property
    def actions(self):
        return len(self.reward)

    def save(self, file):
        np.savez_compressed(file, **{f.name: getattr(self, f.name) for f in fields(self)})

    @classmethod
    def load(cls, file):
        with np.load(file) as z:
            return cls(**{f.name: z[f.name].copy() for f in fields(cls)})

"""Finger contact forces read from the MuJoCo contacts.

Every contact touching a hand body adds its force (world frame, acting on the
hand) to that body. Bodies are grouped per digit (th, ff, mf, rf, lf) + palm.
"""

import mujoco
import numpy as np

from .scene import HAND_PREFIX

DIGITS = ("th", "ff", "mf", "rf", "lf", "palm")


class FingerForces:
    def __init__(self, model: mujoco.MjModel, object_body_id: int | None = None):
        self.model = model
        self.object_body_id = object_body_id
        self.body_ids = np.array([b for b in range(model.nbody) if model.body(b).name.startswith(HAND_PREFIX)])
        self.body_names = [model.body(b).name for b in self.body_ids]
        # lfmetacarpal sits inside the palm, so it counts as palm
        self.digit = np.array([next((dg for dg in DIGITS[:5] if n.startswith(HAND_PREFIX + dg)
                                     and "metacarpal" not in n), "palm") for n in self.body_names])
        self._row = -np.ones(model.nbody, int)
        self._row[self.body_ids] = np.arange(len(self.body_ids))
        self.force = np.zeros((len(self.body_ids), 3))   # net contact force per body (N, world)
        self.normal = np.zeros(len(self.body_ids))       # summed normal force per body (N)
        self.object_normal = np.zeros(len(self.body_ids))

    def update(self, data: mujoco.MjData):
        """Recompute per-body contact forces from the current contacts."""
        m = self.model
        self.force[:] = 0
        self.normal[:] = 0
        self.object_normal[:] = 0
        f6 = np.zeros(6)
        for i in range(data.ncon):
            c = data.contact[i]
            body1, body2 = m.geom_bodyid[c.geom1], m.geom_bodyid[c.geom2]
            rows = self._row[body1], self._row[body2]
            if rows[0] < 0 and rows[1] < 0:
                continue
            mujoco.mj_contactForce(m, data, i, f6)
            f_world = c.frame.reshape(3, 3).T @ f6[:3]    # force of geom1 on geom2
            for r, sign in zip(rows, (-1.0, 1.0)):
                if r >= 0:
                    self.force[r] += sign * f_world
                    self.normal[r] += f6[0]
            if self.object_body_id is not None:
                other_body = (
                    body2 if body1 == self.object_body_id
                    else body1 if body2 == self.object_body_id
                    else -1
                )
                if other_body >= 0:
                    row = self._row[other_body]
                    if row >= 0:
                        self.object_normal[row] += f6[0]
        return self.force

    def per_digit(self) -> dict[str, float]:
        """Summed normal force per digit / palm (N)."""
        return {dg: float(self.normal[self.digit == dg].sum()) for dg in DIGITS}

    def object_per_digit(self) -> dict[str, float]:
        """Summed mug-to-hand normal force per digit / palm (N)."""
        if self.object_body_id is None:
            raise RuntimeError("No object body was configured for force tracking")
        return {dg: float(self.object_normal[self.digit == dg].sum()) for dg in DIGITS}

    def fingertips(self) -> np.ndarray:
        """Net contact force on each distal phalanx (5, 3), order th, ff, mf, rf, lf."""
        return np.stack([self.force[self.body_names.index(f"{HAND_PREFIX}{dg}distal")] for dg in DIGITS[:5]])

"""Hand + object environment (gymnasium-style API, no hard dependency).

Per physics step: mj_step -> finger contact forces.
Observation = hand joint state + contact forces. Reward is a placeholder.
"""

import mujoco
import numpy as np

from .forces import FingerForces
from .scene import (ObjectSpec, build_scene, hand_joint_dof_adr, hand_joint_qpos_adr,
                    load_keyframes, qpos_to_ctrl)


class HandTactileEnv:
    def __init__(self, obj: ObjectSpec | None = None, control_dt=0.02):
        self.model = build_scene(obj)
        self.data = mujoco.MjData(self.model)
        self.kf = load_keyframes(self.model)
        self.qadr = hand_joint_qpos_adr(self.model)
        self.dadr = hand_joint_dof_adr(self.model)
        self.holder = self.model.equality("holder").id
        self.n_substeps = int(round(control_dt / self.model.opt.timestep))
        self.ctrl_lo, self.ctrl_hi = self.model.actuator_ctrlrange.T
        self.forces = FingerForces(self.model)

    # ---- API ----
    def reset(self, seed=None):
        mujoco.mj_resetData(self.model, self.data)
        self.data.qpos[self.qadr] = self.kf["open hand"]
        self.data.ctrl[:] = qpos_to_ctrl(self.model, self.kf["open hand"])
        self.data.eq_active[self.holder] = 1
        mujoco.mj_forward(self.model, self.data)
        self.forces.update(self.data)
        return self._obs(), {}

    def step(self, action):
        """action in [-1, 1]^20 -> actuator position targets."""
        a = np.clip(action, -1, 1)
        self.data.ctrl[:] = self.ctrl_lo + (a + 1) / 2 * (self.ctrl_hi - self.ctrl_lo)
        for _ in range(self.n_substeps):
            self.physics_step()
        reward, info = self._reward()
        return self._obs(), reward, False, False, info

    def physics_step(self, ctrl=None):
        """One physics step; returns the per-body contact forces (n_bodies, 3)."""
        if ctrl is not None:
            self.data.ctrl[:] = ctrl
        mujoco.mj_step(self.model, self.data)
        return self.forces.update(self.data)

    def release_object(self):
        self.data.eq_active[self.holder] = 0

    def grip_ctrl(self, alpha):
        """Grip synergy: blend open -> closed hand (alpha~0.6-0.7 = moderate grip)."""
        q = self.kf["open hand"] + alpha * (self.kf["close hand"] - self.kf["open hand"])
        return qpos_to_ctrl(self.model, q)

    # ---- internals ----
    def _obs(self):
        return {"qpos": self.data.qpos[self.qadr].copy(), "qvel": self.data.qvel[self.dadr].copy(),
                "fingertip_force": self.forces.fingertips(), "contact_normal": self.forces.normal.copy()}

    def _reward(self):
        obj = self.model.body("object").id
        drop = max(0.0, 0.045 - self.data.xpos[obj][2])
        force = self.forces.normal.sum()
        # placeholder: keep object, use minimal grip force
        return -10 * drop - 0.01 * force, {"total_force": force, "object_pos": self.data.xpos[obj].copy()}

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
        self.holder = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_EQUALITY, "holder"
        )
        self.lift_actuator = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, "hand_lift_act"
        )
        self.approach_actuator = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, "hand_approach_act"
        )
        self.lift_joint = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_JOINT, "hand_lift"
        )
        self.approach_joint = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_JOINT, "hand_approach"
        )
        self.n_substeps = int(round(control_dt / self.model.opt.timestep))
        self.ctrl_lo, self.ctrl_hi = self.model.actuator_ctrlrange.T
        self.forces = FingerForces(self.model, object_body_id=self.model.body("object").id)

    # ---- API ----
    def reset(self, seed=None, hand_pose="open hand"):
        if hand_pose not in self.kf:
            raise ValueError(f"Unknown hand keyframe: {hand_pose}")
        mujoco.mj_resetData(self.model, self.data)
        self.data.qpos[self.qadr] = self.kf[hand_pose]
        self.data.ctrl[:] = qpos_to_ctrl(self.model, self.kf[hand_pose])
        if self.holder >= 0:
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
        if self.holder >= 0:
            self.data.eq_active[self.holder] = 0

    @property
    def lift_height(self):
        if self.lift_joint < 0:
            return 0.0
        qpos_address = self.model.jnt_qposadr[self.lift_joint]
        return float(self.data.qpos[qpos_address])

    @property
    def approach_position(self):
        if self.approach_joint < 0:
            return 0.0
        qpos_address = self.model.jnt_qposadr[self.approach_joint]
        return float(self.data.qpos[qpos_address])

    def set_lift_target(self, height):
        if self.lift_actuator < 0:
            raise RuntimeError("This scene does not have a hand lift actuator")
        lower, upper = self.model.actuator_ctrlrange[self.lift_actuator]
        self.data.ctrl[self.lift_actuator] = np.clip(height, lower, upper)
        return float(self.data.ctrl[self.lift_actuator])

    def set_approach_target(self, distance):
        if self.approach_actuator < 0:
            raise RuntimeError("This scene does not have a hand approach actuator")
        lower, upper = self.model.actuator_ctrlrange[self.approach_actuator]
        self.data.ctrl[self.approach_actuator] = np.clip(distance, lower, upper)
        return float(self.data.ctrl[self.approach_actuator])

    def grip_ctrl(self, alpha, start_pose="open hand"):
        """Blend a starting hand pose toward closed while keeping the wrist fixed."""
        if start_pose not in self.kf:
            raise ValueError(f"Unknown hand keyframe: {start_pose}")
        q_start = self.kf[start_pose]
        q_end = self.kf["close hand"].copy()
        q_end[:2] = q_start[:2]
        q = q_start + alpha * (q_end - q_start)
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

"""Hand + tactile pathway environment (gymnasium-style API, no hard dependency).

Per physics step: mj_step -> skin -> afferent spikes -> CNS (cuneate..area 2).
Observation = cortical percept (+ proprioception). Reward is a placeholder.
"""

import mujoco
import numpy as np
import torch

from .afferents import AfferentParams, AfferentPopulation
from .cns import SomatosensoryCortex, encode_proprio
from .scene import (ObjectSpec, build_scene, hand_joint_dof_adr, hand_joint_qpos_adr,
                    load_keyframes, qpos_to_ctrl)
from .skin import Skin, SkinParams


class HandTactileEnv:
    def __init__(self, obj: ObjectSpec | None = None, control_dt=0.02, use_cns=True,
                 afferent_params: AfferentParams | None = None, seed=0):
        self.model = build_scene(obj)
        self.data = mujoco.MjData(self.model)
        self.kf = load_keyframes(self.model)
        self.qadr = hand_joint_qpos_adr(self.model)
        self.dadr = hand_joint_dof_adr(self.model)
        jids = [j for j in range(self.model.njnt) if self.model.joint(j).name.startswith("rh_")]
        self.q_lo, self.q_hi = self.model.jnt_range[jids].T
        self.holder = self.model.equality("holder").id
        self.n_substeps = int(round(control_dt / self.model.opt.timestep))
        self.ctrl_lo, self.ctrl_hi = self.model.actuator_ctrlrange.T

        self.skin = Skin(self.model, SkinParams(seed=seed))
        self.aff = AfferentPopulation(self.skin, afferent_params or AfferentParams(seed=seed))
        self.use_cns = use_cns
        n_prop = len(self.qadr) * 6 + 2 * len(self.dadr)
        self.cns = SomatosensoryCortex(self.aff, n_prop, dt=self.model.opt.timestep, seed=seed) if use_cns else None
        self.last_spikes = {}

    # ---- API ----
    def reset(self, seed=None):
        mujoco.mj_resetData(self.model, self.data)
        self.data.qpos[self.qadr] = self.kf["open hand"]
        self.data.ctrl[:] = qpos_to_ctrl(self.model, self.kf["open hand"])
        self.data.eq_active[self.holder] = 1
        mujoco.mj_forward(self.model, self.data)
        self.aff.reset()
        self.cns_state = self.cns.init_state() if self.use_cns else None
        self.percept = np.zeros(self.cns.sizes["out"] if self.use_cns else 0, np.float32)
        return self._obs(np.zeros(self.aff.n)), {}

    def step(self, action):
        """action in [-1, 1]^20 -> actuator position targets."""
        a = np.clip(action, -1, 1)
        self.data.ctrl[:] = self.ctrl_lo + (a + 1) / 2 * (self.ctrl_hi - self.ctrl_lo)
        counts = np.zeros(self.aff.n)
        for _ in range(self.n_substeps):
            counts += self.physics_step()
        obs = self._obs(counts)
        reward, info = self._reward()
        return obs, reward, False, False, info

    def physics_step(self, ctrl=None):
        """One physics + neural step; returns afferent spike counts for this step."""
        if ctrl is not None:
            self.data.ctrl[:] = ctrl
        mujoco.mj_step(self.model, self.data)
        self.skin.update(self.data)
        spikes = self.aff.step(self.model.opt.timestep)
        counts = spikes.sum(0).astype(np.float32)
        self.last_spikes = {"aff": counts}
        if self.use_cns:
            with torch.no_grad():
                out, self.cns_state, sp = self.cns(torch.from_numpy(counts)[None],
                                                   torch.from_numpy(self._proprio())[None], self.cns_state)
            self.percept = out[0].numpy()
            self.last_spikes.update({k: v[0].numpy() for k, v in sp.items()})
        return counts

    def release_object(self):
        self.data.eq_active[self.holder] = 0

    def grip_ctrl(self, alpha):
        """Grip synergy: blend open -> closed hand (alpha~0.6-0.7 = moderate grip)."""
        q = self.kf["open hand"] + alpha * (self.kf["close hand"] - self.kf["open hand"])
        return qpos_to_ctrl(self.model, q)

    # ---- internals ----
    def _proprio(self):
        return encode_proprio(self.data.qpos[self.qadr], self.data.qvel[self.dadr], self.q_lo, self.q_hi)

    def _obs(self, counts):
        return {"percept": self.percept.copy(), "proprio": self._proprio(), "afferent_counts": counts}

    def _reward(self):
        obj = self.model.body("object").id
        drop = max(0.0, 0.045 - self.data.xpos[obj][2])
        force = self.skin.pressure @ self.skin.area
        # placeholder: keep object, use minimal grip force
        return -10 * drop - 0.01 * force, {"total_force": force, "object_pos": self.data.xpos[obj].copy()}

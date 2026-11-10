"""Train a policy to regulate fingertip force on a fixture-held sphere.

This controlled experiment isolates grip-force regulation and tactile feedback.
It does not simulate picking up, slipping, or crushing an object.

Install optional packages: ``python -m pip install gymnasium stable-baselines3``
Run one encoding: ``python -m tactile_rl.rl_feedback --condition sqrt``
Change the condition to try different encoding strategies.
"""

import argparse
from pathlib import Path

import gymnasium as gym
import mujoco
import numpy as np

from .env import HandTactileEnv
from .feedback import EncodingConfig, FeedbackPipeline, SensorConfig
from .scene import ObjectSpec

TIPS = ("th", "ff", "mf", "rf")
CONTROL_DT = 0.02
SPHERE_RADIUS_RANGE_M = (0.029, 0.030)
TARGET_FORCE_RANGE_N = (22.0, 26.0)
FORCE_SENSOR_MAX_N = 15.0
MAX_FORCE_N = 30.0  # Experimental cutoff only; not a material or safety limit
FORCE_TOLERANCE_N = 1.0


class GraspFeedbackEnv(gym.Env):
    """Regulate target fingertip force using none, raw, or encoded feedback."""

    def __init__(
        self,
        feedback="encoded",
        encoding=None,
        levels=6,
        noise=0.05,
        radius_range=SPHERE_RADIUS_RANGE_M,
        target_force_range=TARGET_FORCE_RANGE_N,
        max_safe_force_n=MAX_FORCE_N,
        episode_s=2.0,
        hold_s=0.3,
        seed=None,
    ):
        super().__init__()
        if feedback not in ("none", "raw", "encoded"):
            raise ValueError("feedback must be 'none', 'raw', or 'encoded'")
        if levels < 2:
            raise ValueError("levels must be at least 2")

        self.feedback = feedback
        self.levels = levels
        self.noise = noise
        self.radius_range = radius_range
        self.target_force_range = target_force_range
        self.max_safe_force_n = max_safe_force_n
        self.n_steps = round(episode_s / CONTROL_DT)
        self.success_hold_steps = max(1, round(hold_s / CONTROL_DT))

        self.sim = HandTactileEnv(
            obj=ObjectSpec(
                shape="sphere",
                size=(0.030,),
                mass=0.04,
                offset=(0.025, -0.010, -0.010),
            ),
            control_dt=CONTROL_DT,
        )

        self.obj = self.sim.model.body("object").id
        self.sphere_geom = self.sim.model.geom("object_geom").id
        self.pipe = FeedbackPipeline(
            len(TIPS),
            SensorConfig(max_force_n=FORCE_SENSOR_MAX_N),
            encoding or EncodingConfig(),
        )

        # The single action adjusts squeeze; force target and time are observations.
        self.action_space = gym.spaces.Box(-1.0, 1.0, shape=(1,), dtype=np.float32)
        self.observation_space = gym.spaces.Box(
            0.0, 1.0, shape=(len(TIPS) + 3,), dtype=np.float32
        )
        super().reset(seed=seed)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.sim.reset(hand_pose="open hand")
        radius = float(self.np_random.uniform(*self.radius_range))
        model = self.sim.model
        model.geom_size[self.sphere_geom, 0] = radius
        qpos = self.sim.data.qpos.copy()
        qvel = self.sim.data.qvel.copy()
        ctrl = self.sim.data.ctrl.copy()
        mujoco.mj_setConst(model, self.sim.data)
        self.sim.data.qpos[:] = qpos
        self.sim.data.qvel[:] = qvel
        self.sim.data.ctrl[:] = ctrl
        mujoco.mj_forward(model, self.sim.data)
        self.sim.forces.update(self.sim.data)

        self.pipe.reset()
        self.grip = 0.5        # start just before contact, not fully open
        self.target_force = float(self.np_random.uniform(*self.target_force_range))
        self.t = 0
        self.ever_succeeded = False
        self.target_steps = 0
        self.forces = self._tip_forces()
        _, intensities = self.pipe.update(self.forces.tolist())
        self.intensities = np.asarray(intensities)
        return self._obs(), {
            "sphere_radius_m": radius,
            "target_force_n": self.target_force,
            "object_geometry": "rigid sphere held in fixture",
        }

    def step(self, action):
        # 1. ACT: change the grip a little (max 1% per step, so it can brake before overshooting)
        action = np.asarray(action, dtype=np.float32)
        self.grip = float(np.clip(self.grip + 0.01 * action[0], 0.0, 1.0))

        # 2. PHYSICS: simulate 20 ms
        ctrl = self.sim.grip_ctrl(self.grip, start_pose="open hand")
        for _ in range(self.sim.n_substeps):
            self.sim.physics_step(ctrl)
        self.t += 1

        # 3. FEEL: fingertip forces -> our encoding -> vibration intensities
        self.forces = self._tip_forces()
        _, intensities = self.pipe.update(self.forces.tolist())
        self.intensities = np.asarray(intensities)

        # 4. CHECK: how far from the target, and is it over the limit?
        fingertip_force = float(self.forces.sum())
        force_error = fingertip_force - self.target_force
        over_force = fingertip_force > self.max_safe_force_n

        # Success bookkeeping (for evaluation only; it no longer ends the trial)
        if abs(force_error) <= FORCE_TOLERANCE_N:
            self.target_steps += 1
        else:
            self.target_steps = 0
        if self.target_steps >= self.success_hold_steps:
            self.ever_succeeded = True
        success = self.ever_succeeded

        terminated = False                      # nothing ends the trial early
        truncated = self.t >= self.n_steps      # only the time limit ends it

        # 5. SCORE: positive when close to the target, negative while over the limit
        error_frac = min(abs(force_error) / self.target_force, 1.0)   # 0 = on target, 1 = far off
        reward = 1.0 - error_frac                                    # 0..1 each step: closer = better
        if abs(force_error) <= FORCE_TOLERANCE_N:
            reward += 1.0                                            # bonus for each step on target
        if over_force:
            reward -= 2.0                                            # over the limit hurts every step
                                        
        info = {
            "fingertip_force_n": fingertip_force,
            "target_force_n": self.target_force,
            "force_error_n": force_error,
            "over_force": bool(over_force),
            "success": bool(success),
            "force_limit_is_validated": False,
            "force_limit_note": "experimental fingertip-force cutoff, not a safety limit",
        }
        return self._obs(), float(reward), terminated, truncated, info


    def _tip_forces(self):
        object_forces = self.sim.forces.object_per_digit()
        return np.array([object_forces[tip] for tip in TIPS], dtype=float)

    def _obs(self):
        if self.feedback == "none":
            felt = np.zeros(len(TIPS))
        elif self.feedback == "raw":
            felt = np.clip(self.forces / FORCE_SENSOR_MAX_N, 0.0, 1.0)
        else:
            felt = np.round(self.intensities * (self.levels - 1)) / (self.levels - 1)
            felt = np.clip(felt + self.noise * self.np_random.standard_normal(felt.shape), 0, 1)
        target_normalized = (
            (self.target_force - self.target_force_range[0])
            / (self.target_force_range[1] - self.target_force_range[0])
        )
        return np.append(
            felt,
            [self.grip, target_normalized, min(self.t / self.n_steps, 1.0)],
        ).astype(np.float32)

CONDITIONS = {
    "none": {"feedback": "none"},
    "raw": {"feedback": "raw"},
    "linear": {"feedback": "encoded", "encoding": EncodingConfig(amplitude_exponent=1.0)},
    "sqrt": {"feedback": "encoded", "encoding": EncodingConfig(amplitude_exponent=0.5)},
    "binary": {"feedback": "encoded", "levels": 2},
}

def main():
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import DummyVecEnv

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--condition", choices=CONDITIONS, default="sqrt")
    parser.add_argument("--steps", type=int, default=20_000)
    parser.add_argument("--max-force-n", type=float, default=MAX_FORCE_N)
    parser.add_argument("--output-dir", default="models")
    args = parser.parse_args()

    condition = CONDITIONS[args.condition]
    env = DummyVecEnv([
        lambda: GraspFeedbackEnv(seed=0, max_safe_force_n=args.max_force_n, **condition)
    ])
    try:
        model = PPO("MlpPolicy", env, seed=0, verbose=0)
        model.learn(total_timesteps=args.steps)
        model_path = Path(args.output_dir) / f"ppo_{args.condition}"
        model_path.parent.mkdir(parents=True, exist_ok=True)
        model.save(str(model_path))
    finally:
        env.close()

    test_env = GraspFeedbackEnv(
        seed=1000, max_safe_force_n=args.max_force_n, **condition
    )
    successes = 0
    errors = []
    try:
        for episode in range(10):
            obs, _ = test_env.reset(seed=1000 + episode)
            done = False
            while not done:
                action, _ = model.predict(obs, deterministic=True)
                obs, _, terminated, truncated, info = test_env.step(action)
                done = terminated or truncated
                errors.append(abs(info["force_error_n"]))
            successes += info["success"]
    finally:
        test_env.close()
    print(
        f"{args.condition}: target-force success {successes}/10; "
        f"mean absolute force error {np.mean(errors):.2f} N"
    )
    print(f"Saved model to {model_path}.zip")


if __name__ == "__main__":
    main()

"""Watch tactile grip-force regulation trials in the browser viewer.

    .venv/bin/python -m scripts.watch_rl_trial --grip 0.6
    #to watch a model that you have trained, run the rl_feedback.py for a certain condition first then watch that model
    .venv/bin/python -m scripts.watch_rl_trial --model models/ppo_sqrt.zip --condition sqrt 
    .venv/bin/python -m scripts.watch_rl_trial --condition none --grip 0.8
Then open the URL printed by the viewer.

The terminal prints target force, measured fingertip force, grip, and encoded
feedback. Fixed-grip mode is an open-loop baseline; trials have randomized
targets and sphere sizes.
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
from tactile_rl.rl_feedback import CONDITIONS, CONTROL_DT, MAX_FORCE_N, GraspFeedbackEnv  # noqa: E402
from mjviser import Viewer  # noqa: E402

PAUSE_CALLS = 50  # how long to show the end of a trial before starting the next

ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("--condition", choices=CONDITIONS, default="raw")
ap.add_argument("--grip", type=float, default=0.6, help="fixed grip for the open-loop baseline")
ap.add_argument("--model", help="trained force-regulation agent, e.g. models/ppo_sqrt.zip")
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--max-force-n", type=float, default=MAX_FORCE_N)
ap.add_argument("--delay", type=float, default=CONTROL_DT,
                help="seconds to wait per 20 ms step; raise to slow down, 0 = as fast as possible")
args = ap.parse_args()
if not 0.0 <= args.grip <= 1.0:
    ap.error("--grip must be between 0 and 1")

env = GraspFeedbackEnv(max_safe_force_n=args.max_force_n, **CONDITIONS[args.condition])
model = None
if args.model:
    from stable_baselines3 import PPO
    model = PPO.load(args.model)
    if model.observation_space.shape != env.observation_space.shape:
        raise ValueError(
            f"Model observation shape {model.observation_space.shape} does not match "
            f"the current force-regulation environment {env.observation_space.shape}. "
            "Load a model trained with the current task."
        )
    if model.action_space.shape != env.action_space.shape:
        raise ValueError(
            f"Model action shape {model.action_space.shape} does not match "
            f"the current force-regulation environment {env.action_space.shape}."
        )

state = {"obs": None, "trial": 0, "done": False, "pause": 0}


def new_trial():
    obs, info = env.reset(seed=args.seed + state["trial"])
    state.update(obs=obs, done=False, pause=0)
    who = f"agent {args.model}" if model else f"fixed grip {args.grip:.2f}"
    print(f"\n=== trial {state['trial']}  ·  sphere radius {1000 * info['sphere_radius_m']:.0f} mm"
          f"  ·  target {info['target_force_n']:.1f} N"
          f"  ·  {args.condition}  ·  {who} ===")
    state["trial"] += 1


def choose_action(obs):
    if model is not None:
        action, _ = model.predict(obs, deterministic=True)
        return action
    return np.array([np.clip((args.grip - env.grip) / 0.05, -1.0, 1.0)], dtype=np.float32)


def step_fn(m, d):
    if state["done"]:                       # trial over: hold the last frame for a moment
        state["pause"] += 1
        if state["pause"] >= PAUSE_CALLS:
            new_trial()
        return

    obs, _, terminated, truncated, info = env.step(choose_action(state["obs"]))
    state["obs"] = obs
    t = env.t * CONTROL_DT
    if env.t % 5 == 0 or terminated or truncated:
        felt = " ".join(f"{x:.2f}" for x in obs[:4])
        print(
            f"  t={t:4.2f}s  grip={env.grip:.2f}"
            f"  force={info['fingertip_force_n']:5.1f}/{info['target_force_n']:4.1f} N"
            f"  error={info['force_error_n']:+5.1f} N  felt=[{felt}]"
        )
    if terminated or truncated:
        outcome = (
            "TARGET FORCE HELD" if info["success"]
            else "OVER EXPERIMENTAL FORCE LIMIT" if info["over_force"]
            else "TIME LIMIT"
        )
        print(f"  -> {outcome}")
        state["done"] = True

    if args.delay > 0:
        time.sleep(args.delay)


new_trial()
Viewer(env.sim.model, env.sim.data, step_fn=step_fn,
       reset_fn=lambda m, d: new_trial()).run()

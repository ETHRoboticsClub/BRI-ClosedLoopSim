"""Interactive web viewer (mjviser): scripted grasp with live finger forces.

    python scripts/view.py            # then open the printed URL (default http://localhost:8080)
    python scripts/view.py --grip 0.7 # stronger grip

Protocol (loops): close hand -> release object -> hold -> open hand.
Prints the summed normal contact force per digit once per second.
"""

import argparse
import sys
from pathlib import Path

import numpy as np
from mjviser import Viewer

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from tactile_rl.env import HandTactileEnv  # noqa: E402
from tactile_rl.forces import DIGITS  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--grip", type=float, default=0.65)
args = ap.parse_args()

env = HandTactileEnv()
env.reset()
dt = env.model.opt.timestep
open_c, grip_c = env.grip_ctrl(0.0), env.grip_ctrl(args.grip)
CYCLE = 4.0
t_print = 0.0


def step_fn(m, d):
    global t_print
    t = d.time % CYCLE
    if t < dt:
        env.reset()
        t_print = 0.0
    s = np.clip((t - 0.3) / 0.7, 0, 1) if t < 2.8 else 1 - np.clip((t - 2.8) / 0.5, 0, 1)
    if t > 1.3:
        env.release_object()
    env.physics_step(open_c + s * (grip_c - open_c))
    if d.time - t_print >= 1.0:
        f = env.forces.per_digit()
        print(f"t={d.time:5.1f} F={sum(f.values()):5.1f}N  " + " ".join(f"{dg}:{f[dg]:5.2f}" for dg in DIGITS))
        t_print = d.time


Viewer(env.model, env.data, step_fn=step_fn, reset_fn=lambda m, d: env.reset()).run()

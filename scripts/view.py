"""Interactive viewer: scripted grasp with the full tactile pathway running live.

    python scripts/view.py            # with CNS
    python scripts/view.py --no-cns   # skin + afferents only (faster)

Protocol (loops): close hand -> release object -> hold -> open hand.
Prints firing rates per afferent type and per cortical area once per second.
"""

import argparse
import sys
import time
from pathlib import Path

import mujoco
import mujoco.viewer
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from tactile_rl.afferents import TYPES  # noqa: E402
from tactile_rl.env import HandTactileEnv  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--no-cns", action="store_true")
ap.add_argument("--grip", type=float, default=0.65)
args = ap.parse_args()

env = HandTactileEnv(use_cns=not args.no_cns)
print(f"taxels={env.skin.n}  afferents={env.aff.n} "
      + " ".join(f"{t}={len(env.aff.type_idx[t])}" for t in TYPES)
      + (f"  CNS={env.cns.sizes}" if env.cns else ""))
env.reset()
m, d, dt = env.model, env.data, env.model.opt.timestep
open_c, grip_c = env.grip_ctrl(0.0), env.grip_ctrl(args.grip)
CYCLE = 4.0

with mujoco.viewer.launch_passive(m, d) as viewer:
    acc, n_acc, t_print = {}, 0, 0.0
    while viewer.is_running():
        t0 = time.time()
        t = d.time % CYCLE
        if t < dt:
            env.reset()
        s = np.clip((t - 0.3) / 0.7, 0, 1) if t < 2.8 else 1 - np.clip((t - 2.8) / 0.5, 0, 1)
        if t > 1.3:
            env.release_object()
        env.physics_step(open_c + s * (grip_c - open_c))
        for k, v in env.last_spikes.items():
            acc[k] = acc.get(k, 0) + v
        n_acc += 1
        if d.time - t_print >= 1.0:
            T = n_acc * dt
            a = acc["aff"] / T
            msg = " ".join(f"{ty}:{a[env.aff.type_idx[ty]].mean():5.1f}Hz" for ty in TYPES)
            msg += "".join(f" | {k}:{(acc[k] / T).mean():5.1f}Hz" for k in ("cn", "3b", "1", "2") if k in acc)
            print(f"t={d.time:5.1f} F={env.skin.pressure @ env.skin.area:5.1f}N  {msg}")
            acc, n_acc, t_print = {}, 0, d.time
        viewer.sync()
        time.sleep(max(0.0, dt - (time.time() - t0)))

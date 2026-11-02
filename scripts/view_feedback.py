"""Shadow Hand viewer with four-channel force-to-vibration encoding.

Prints raw distal-finger contact forces (N) and commanded vibration
intensities (0..1), not simulated or measured motor vibration.
"""

import argparse
import sys
from pathlib import Path

import numpy as np
from mjviser import Viewer

# Allow running as `python scripts/view_feedback.py` from the simulation repo.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from tactile_rl.env import HandTactileEnv  # noqa: E402
from tactile_rl.feedback import (
    FeedbackPipeline,
    SensorConfig,
    EncodingConfig,
)

FINGERS = ("th", "ff", "mf", "rf")  # thumb, index, middle, ring
CYCLE_S = 4.0
FEEDBACK_INTERVAL_S = 0.02  # 50 Hz processing; MuJoCo physics may be faster

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--grip", type=float, default=0.65)
args = parser.parse_args()

env = HandTactileEnv()
open_ctrl = env.grip_ctrl(0.0)
grip_ctrl = env.grip_ctrl(args.grip)
physics_dt = float(env.model.opt.timestep)
physics_steps_per_feedback = max(1, round(FEEDBACK_INTERVAL_S / physics_dt))

# This measures normal forces on entire distal phalanx bodies, NOT a modeled
# FSR patch, and includes contacts with anything touching those bodies.
tip_rows = [env.forces.body_names.index(f"rh_{finger}distal") for finger in FINGERS]



pipeline = FeedbackPipeline(
    n_sensors=4,

    sensor_config=SensorConfig(
        min_force_n=0.0,
        max_force_n=5.0,
        smoothing_alpha=0.25,
    ),

    encoding_config=EncodingConfig(
        contact_on=0.12,
        contact_off=0.06,
        amplitude_exponent=0.5,
    ),
)



def reset_simulation():
    global physics_steps, last_print_s
    env.reset()
    pipeline.reset()
    physics_steps = 0
    last_print_s = 0.0


def step_fn(model, data):
    global physics_steps, last_print_s

    t = data.time % CYCLE_S
    if t < physics_dt:
        reset_simulation()
        t = 0.0

    # Follow the original scripted grasp protocol from scripts/view.py.
    close_fraction = (
        np.clip((t - 0.3) / 0.7, 0, 1)
        if t < 2.8
        else 1 - np.clip((t - 2.8) / 0.5, 0, 1)
    )
    if t > 1.3:
        env.release_object()

    env.physics_step(open_ctrl + close_fraction * (grip_ctrl - open_ctrl))
    physics_steps += 1

    # Read force and run filter/encoder at ~50 Hz, not at every physics step.
    if physics_steps % physics_steps_per_feedback != 0:
        return

    forces_n = env.forces.normal[tip_rows].tolist()
    readings, intensities = pipeline.update(forces_n)

    if data.time - last_print_s >= 0.5:
        print(f"t={data.time:.2f}s")
        for finger, reading, intensity in zip(FINGERS, readings, intensities):
            print(
                f"  {finger}: force={reading.force_n:.3f} N  "
                f"normalized={reading.normalized:.3f}  "
                f"intensity={intensity:.3f}"
            )
        last_print_s = data.time


reset_simulation()
Viewer(env.model, env.data, step_fn=step_fn,
       reset_fn=lambda model, data: reset_simulation()).run()

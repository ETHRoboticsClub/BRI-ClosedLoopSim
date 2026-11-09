# closed_loop_rl_feedback

Shadow Hand E3M5 (MuJoCo) with finger contact-force readout, as a testbed for
comparing feedback strategies with RL.

```
MuJoCo contacts -> per-body contact forces (world frame) -> per-digit / fingertip forces -> RL policy
```

## Layout
- `assets/shadow_hand/` – MuJoCo Menagerie Shadow Hand E3M5 (Apache-2.0), commit 0059d43
- `tactile_rl/scene.py` – hand + object scene (MjSpec): sphere/box/cylinder/ellipsoid objects held by a "holder" weld, or an optional mug on a table with forearm approach/lift joints; keyframe loading and joint-angle → actuator mapping
- `tactile_rl/forces.py` – `FingerForces`: contact forces per hand body, per digit, per fingertip
- `tactile_rl/env.py` – `HandTactileEnv` (gymnasium-style `reset`/`step`; reward is a placeholder)
- `tactile_rl/feedback/` – feedback encoding logic
- `tactile_rl/rl_feedback.py` – RL grip-force experiment (see below)
- `scripts/view.py` – interactive web viewer (mjviser) with live finger forces
- `scripts/view_feedback.py` – viewer with live feedback values
- `scripts/watch_rl_trial.py` – watch an RL trial

## Run
```bash
pip install -r requirements.txt
python scripts/view.py            # open http://localhost:8080
python scripts/view.py --grip 0.7 # stronger grip
```
Grasp loop: hand closes (0.3–1.0 s), experimenter releases object (1.3 s),
hold, hand opens (2.8 s). Per-digit normal forces are printed once per second.

## Run with feedback values in terminal
```bash
python scripts/view_feedback.py --grip 0.7   # open http://localhost:8080
```

## Vibrotactile Feedback Encoding

Converts MuJoCo fingertip contact forces into normalized vibration intensity
commands for four fingers (thumb, index, middle, ring).

**Pipeline:** Force (N) → Smoothing & Normalization → Encoding → Vibration Intensity (0–1)

**Files:**
- `tactile_rl/feedback/sensor.py` — Smooths and normalizes forces
- `tactile_rl/feedback/encoding.py` — Maps pressure to vibration intensity
- `tactile_rl/feedback/pipeline.py` — Processes all four sensors
- `scripts/view_feedback.py` — Connects the pipeline to MuJoCo

**Experimentation:**

Adjust parameters in `view_feedback.py`:
- `smoothing_alpha` — Controls signal smoothing; higher alpha means less smoothing with the previous force value
- `max_force_n` — Sets normalization range
- `contact_on` / `contact_off` — Set contact detection thresholds
- `amplitude_exponent` — Controls encoding strategy (`0.5` = square root, `1.0` = linear, `2.0` = squared)

Outputs are simulated vibration intensity commands, not physical motor vibrations.

## RL feedback experiment

`tactile_rl/rl_feedback.py` trains a policy to regulate fingertip force with
the Shadow Hand. A rigid sphere is held in the existing fixture while the
policy adjusts grip to match a force target. The target varies from 22 to 26 N,
and the sphere radius varies from 2.9 to 3.0 cm between episodes. The sphere is
positioned so thumb, index, middle, and ring fingers all contact it during the
force-control range.

**Observation (7 values):** four feedback channels (one per finger), current
grip, normalized target force, and episode progress.
**Action:** a single value that increases or decreases grip (at most 1% per step).
**Episodes:** always run for 2 s (100 control steps at 50 Hz); nothing ends them early.

**Reward per step:** higher the closer summed fingertip force is to the target,
+1 bonus while within 1 N of the target, −2 while summed force exceeds 30 N,
and a small penalty for grip strength. The 30 N limit is only an experimental
setting; it is not a real-world calibration, crush threshold, or safety limit.

**Feedback conditions:**
- `none` — feedback channels are all zero
- `raw` — each finger's instantaneous force normalized against 15 N; no smoothing, quantization or noise
- `linear` / `sqrt` — force smoothed (EMA, alpha 0.7) and normalized against 15 N, encoded (exponent 1.0 / 0.5), quantized to 6 levels, plus 5% Gaussian noise
- `binary` — same smoothing and normalization, encoded intensity quantized to 2 levels (on/off), plus 5% Gaussian noise

Note that `raw` differs from the encoded conditions in smoothing, quantization
and noise, not just in the force-to-intensity mapping. Because the pipeline
normalizes against 15 N, the 0.12 contact-on threshold corresponds to about
1.8 N of smoothed force at one fingertip.

Install Gymnasium and Stable-Baselines3 with
`python -m pip install gymnasium stable-baselines3`, then train one condition:

```bash
python -m tactile_rl.rl_feedback --condition sqrt --steps 20000
```

Options:
- `--condition` — `none`, `raw`, `linear`, `sqrt` (default), or `binary`
- `--steps` — training timesteps (default 20000)
- `--max-force-n` — over-force penalty threshold (default 30)
- `--output-dir` — where to save the model (default `models/`)

Each command trains a new PPO policy from scratch and saves it as
`<output-dir>/ppo_<condition>.zip`. Trained models are not tracked in git.
After training, the policy is evaluated on 10 held-out episodes and the script
prints the success count and mean absolute force error. Success means keeping
summed fingertip force within 1 N of the target for 0.3 s at some point in the
episode. Policies trained with the previous three-channel environment are
incompatible and must be retrained.

This is deliberately a controlled grip-force experiment, not a mug pickup:
the fixture prevents the sphere from slipping or dropping. It tests whether
feedback encoding helps the policy adjust grip across different contact
responses, not whether the simulated hand can grasp objects in daily life.

## Third-party assets

The mug's visual mesh and texture come from [BiGym](https://github.com/NeuracoreAI/bigym)
at commit `14beb30318ad14c5d6723175c2ee2281129792af` (Apache-2.0; see
`assets/third_party/bigym_mug/LICENSE`). The optional procedural mug scene uses
simple box-and-capsule collision shapes; the copied BiGym collision meshes are
retained as source assets but are not used in the simulation.
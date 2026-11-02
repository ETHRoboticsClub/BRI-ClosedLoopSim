# closed_loop_rl_feedback

Shadow Hand E3M5 (MuJoCo) with finger contact-force readout, as a testbed for
comparing feedback strategies with RL.

```
MuJoCo contacts -> per-body contact forces (world frame) -> per-digit / fingertip forces -> RL policy
```

## Layout
- `assets/shadow_hand/` – MuJoCo Menagerie Shadow Hand E3M5 (Apache-2.0), commit 0059d43
- `tactile_rl/scene.py` – hand + object scene (MjSpec), keyframes, grip helpers, "holder" weld
- `tactile_rl/forces.py` – `FingerForces`: contact forces per hand body, per digit, per fingertip
- `tactile_rl/env.py` – `HandTactileEnv` (gymnasium-style `reset`/`step`; reward is a placeholder)
- `tactile_rl/feedback/` -Folder for the feedback encoding logic 
- `scripts/view.py` – interactive web viewer (mjviser) with live finger forces

## Run
```bash
pip install -r requirements.txt
python scripts/view.py            # open http://localhost:8080
python scripts/view.py --grip 0.7 # stronger grip
```
Grasp loop: hand closes (0.3–1.0 s), experimenter releases object (1.3 s),
hold, hand opens (2.8 s). Per-digit normal forces are printed once per second.

## Run with feedback values in terminal
```bash          # open http://localhost:8080
python scripts/view_feedback.py --grip 0.7 #  # open http://localhost:8080
```

## Vibrotactile Feedback Encoding

Converts MuJoCo fingertip contact forces into normalized vibration intensity commands for four fingers (thumb, index, middle, ring).

**Pipeline:** Force (N) → Smoothing & Normalization → Encoding → Vibration Intensity (0–1)

**Files:**
- `tactile_rl/feedback/sensor.py` — Smooths and normalizes forces
- `tactile_rl/feedback/encoding.py` — Maps pressure to vibration intensity
- `tactile_rl/feedback/pipeline.py` — Processes all four sensors
- `scripts/view_feedback.py` — Connects the pipeline to MuJoCo.


**Experimentation:**

Adjust parameters in `view_feedback.py`:
- `smoothing_alpha` — Controls signal smoothing, higher alpha correlates to less smoothing with previous force value
- `max_force_n` — Sets normalization range
- `contact_on` / `contact_off` — Set contact detection thresholds.
- `amplitude_exponent` — Controls encoding strategy (`0.5` = square root, `1.0` = linear, `2.0` = squared).

Outputs are simulated vibration intensity commands, not physical motor vibrations.

## Notes / TODO
- Next: define the RL task + reward, then compare different feedback
  encodings built on top of the finger forces.
- Next: translate to arduino

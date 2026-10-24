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
- `scripts/view.py` – interactive web viewer (mjviser) with live finger forces

## Run
```bash
pip install -r requirements.txt
python scripts/view.py            # open http://localhost:8080
python scripts/view.py --grip 0.7 # stronger grip
```
Grasp loop: hand closes (0.3–1.0 s), experimenter releases object (1.3 s),
hold, hand opens (2.8 s). Per-digit normal forces are printed once per second.

## Notes / TODO
- Next: define the RL task + reward, then compare different feedback
  encodings built on top of the finger forces.

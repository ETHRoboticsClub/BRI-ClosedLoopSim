# closed_loop_rl_feedback

Shadow Hand E3M5 (MuJoCo) with a bio-inspired tactile pathway, as a testbed for
comparing feedback strategies with RL.

```
MuJoCo contacts -> skin (taxels, Hertz pressure/shear)
               -> afferents (SA1 / RA1 / SA2 / PC, adaptive LIF spikes)
               -> cuneate nucleus (fixed, centre-surround)
               -> S1 area 3b (somatotopic) -> area 1 -> area 2 (+ proprioception)
               -> percept vector  ->  RL policy
```

## Layout
- `assets/shadow_hand/` – MuJoCo Menagerie Shadow Hand E3M5 (Apache-2.0), commit 0059d43
- `tactile_rl/scene.py` – hand + object scene (MjSpec), keyframes, grip helpers, "holder" weld
- `tactile_rl/skin.py` – taxels on collision surfaces, contact → pressure/shear maps
- `tactile_rl/afferents.py` – peripheral encoder (4 afferent classes, spiking)
- `tactile_rl/cns.py` – spiking cuneate + S1 model (PyTorch, surrogate gradients)
- `tactile_rl/env.py` – `HandTactileEnv` (gymnasium-style `reset`/`step`; reward is a placeholder)
- `scripts/view.py` – interactive MuJoCo viewer running the full pathway live

## Run
```bash
pip install -r requirements.txt
python scripts/view.py            # full pathway (slower than real time)
python scripts/view.py --no-cns   # skin + afferents only
python scripts/view.py --grip 0.7 # stronger grip
```
Grasp loop: hand closes (0.3–1.0 s), experimenter releases object (1.3 s),
hold, hand opens (2.8 s). Firing rates are printed once per second.

## Notes / TODO
- Parameters (densities, RF sizes, transduction gains) are literature-inspired
  placeholders; not yet calibrated against microneurography data.
- `AfferentParams.density_scale=0.3` gives ~15k afferents (real hand ≈17k).
- Next: define the RL task + reward, then swap in different feedback encodings
  (the thing to compare) between afferents and the CNS.

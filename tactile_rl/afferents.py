"""Peripheral encoder: four classes of tactile afferents with adaptive LIF neurons.

    SA1 (Merkel)   - sustained pressure + its onset, small RF
    RA1 (Meissner) - rate of pressure change (on and off), small RF
    SA2 (Ruffini)  - skin stretch / directional shear, large RF, spontaneous
    PC  (Pacinian) - transients / vibration (2nd derivative of force), huge RF

Densities are per cm^2 of glabrous skin (approx. Johansson & Vallbo 1979).
"""

from dataclasses import dataclass, field

import numpy as np
from scipy import sparse

from .skin import Skin

TYPES = ("SA1", "RA1", "SA2", "PC")

DENSITY = {  # afferents / cm^2
    "fingertip": {"SA1": 70, "RA1": 140, "SA2": 9, "PC": 21},
    "finger": {"SA1": 30, "RA1": 37, "SA2": 14, "PC": 10},
    "palm": {"SA1": 8, "RA1": 25, "SA2": 16, "PC": 9},
}


@dataclass
class TypeParams:
    rf_sigma: float      # m
    x50: float           # half-saturation of transduction
    i_max: float         # max drive (threshold = 1)
    tau_m: float = 0.010
    tau_w: float = 0.1
    b: float = 0.0       # spike-triggered adaptation
    bias: float = 0.0    # spontaneous drive
    noise: float = 0.05


DEFAULT_TYPES = {
    "SA1": TypeParams(rf_sigma=1.0e-3, x50=20e3, i_max=6.0, b=0.05, tau_w=0.3),
    "RA1": TypeParams(rf_sigma=1.5e-3, x50=300e3, i_max=8.0, b=0.3, tau_w=0.05),
    "SA2": TypeParams(rf_sigma=4.0e-3, x50=10e3, i_max=4.0, b=0.02, bias=0.95, tau_w=0.5),
    "PC": TypeParams(rf_sigma=15e-3, x50=2e3, i_max=8.0, b=0.5, tau_w=0.03),
}


@dataclass
class AfferentParams:
    density_scale: float = 0.3   # ~15k afferents (human hand ~17k)
    dt_neuron: float = 0.0005
    t_ref: float = 0.001
    tau_skin: float = 0.004      # viscoelastic low-pass on taxel signals (s)
    types: dict = field(default_factory=lambda: dict(DEFAULT_TYPES))
    seed: int = 0


class AfferentPopulation:
    def __init__(self, skin: Skin, params: AfferentParams | None = None):
        self.skin, self.p = skin, params or AfferentParams()
        rng = np.random.default_rng(self.p.seed)
        rows, cols, vals = [], [], []
        types, pos, body, region, digit, pref = [], [], [], [], [], []
        k = 0
        for b in skin.body_ids:
            idx = np.flatnonzero((skin.body == b) & skin.glabrous)
            if len(idx) == 0:
                continue
            reg = skin.region[idx[0]]
            area_cm2 = skin.area[idx].sum() * 1e4
            for t in TYPES:
                tp = self.p.types[t]
                n = rng.poisson(DENSITY[reg][t] * area_cm2 * self.p.density_scale)
                if n == 0:
                    continue
                centers = skin.pos[rng.choice(idx, n)]
                all_idx = np.flatnonzero(skin.body == b)
                d2 = ((centers[:, None] - skin.pos[all_idx][None]) ** 2).sum(-1)
                w = np.where(d2 < (3 * tp.rf_sigma) ** 2, np.exp(-d2 / (2 * tp.rf_sigma ** 2)), 0.0)
                if t == "PC":   # PC sums force over its RF
                    w = w * skin.area[all_idx][None]
                else:           # others: local weighted-average pressure
                    w = w / np.maximum(w.sum(1, keepdims=True), 1e-12)
                r, c = np.nonzero(w)
                rows.append(r + k); cols.append(all_idx[c]); vals.append(w[r, c])
                types += [t] * n; pos.append(centers); body += [b] * n; region += [reg] * n
                digit += [skin.digit[idx[0]]] * n
                v = rng.normal(size=(n, 3)); pref.append(v / np.linalg.norm(v, axis=1, keepdims=True))
                k += n
        self.n = k
        rows, cols, vals = map(np.concatenate, (rows, cols, vals))
        self.W = sparse.csr_matrix((vals, (rows, cols)), shape=(k, skin.n))
        self.type = np.array(types); self.pos = np.concatenate(pos); self.body = np.array(body)
        self.region = np.array(region); self.digit = np.array(digit)
        self.pref = np.concatenate(pref)  # SA2 preferred stretch direction (body frame)
        self.type_idx = {t: np.flatnonzero(self.type == t) for t in TYPES}

        def per(attr):
            out = np.zeros(k)
            for t in TYPES:
                out[self.type_idx[t]] = getattr(self.p.types[t], attr)
            return out
        gain_jitter = rng.lognormal(0, 0.3, k)
        self.x50 = per("x50") * gain_jitter
        self.i_max, self.tau_m, self.tau_w = per("i_max"), per("tau_m"), per("tau_w")
        self.b, self.bias, self.noise = per("b"), per("bias"), per("noise")
        self.reset()

    def reset(self):
        n, ns = self.n, self.skin.n
        self.v = np.zeros(n); self.w = np.zeros(n); self.ref = np.zeros(n)
        self.p_f = np.zeros(ns); self.s_f = np.zeros((ns, 3))
        self.P_prev = np.zeros(n); self.dP_prev = np.zeros(n)
        self.I_prev = np.zeros(n)

    def _drive(self, dt):
        a = dt / (self.p.tau_skin + dt)
        self.p_f += a * (self.skin.pressure - self.p_f)
        self.s_f += a * (self.skin.shear - self.s_f)
        P = self.W @ self.p_f
        dP = (P - self.P_prev) / dt
        ddP = (dP - self.dP_prev) / dt
        self.P_prev, self.dP_prev = P, dP
        x = np.zeros(self.n)
        i = self.type_idx
        x[i["SA1"]] = P[i["SA1"]] + 0.05 * np.maximum(dP[i["SA1"]], 0)
        x[i["RA1"]] = np.abs(dP[i["RA1"]])
        S = self.W[i["SA2"]] @ self.s_f
        x[i["SA2"]] = (np.maximum((S * self.pref[i["SA2"]]).sum(1), 0)
                       + 0.3 * np.linalg.norm(S, axis=1) + 0.2 * P[i["SA2"]])
        x[i["PC"]] = np.abs(ddP[i["PC"]]) * 1e-3
        return self.bias + self.i_max * x / (x + self.x50)

    def step(self, dt_phys: float) -> np.ndarray:
        """Advance one physics step. Returns spikes (n_sub, n_afferents) bool."""
        I_new = self._drive(dt_phys)
        n_sub = max(1, int(round(dt_phys / self.p.dt_neuron)))
        h = dt_phys / n_sub
        spikes = np.zeros((n_sub, self.n), dtype=bool)
        for s in range(n_sub):
            I = self.I_prev + (I_new - self.I_prev) * (s + 1) / n_sub
            active = self.ref <= 0
            dv = (-self.v + I - self.w) / self.tau_m * h
            self.v += np.where(active, dv, 0) + self.noise * np.sqrt(h / self.tau_m) * np.random.randn(self.n)
            self.w -= self.w / self.tau_w * h
            self.ref -= h
            fired = self.v >= 1.0
            spikes[s] = fired
            self.v[fired] = 0.0; self.w[fired] += self.b[fired]; self.ref[fired] = self.p.t_ref
            self.v = np.maximum(self.v, -1.0)
        self.I_prev = I_new
        return spikes

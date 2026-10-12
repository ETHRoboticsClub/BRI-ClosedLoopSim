"""Central somatosensory pathway (spiking, PyTorch, trainable via surrogate gradients).

    afferents -> cuneate nucleus -> S1 area 3b -> area 1 -> area 2 (+ proprioception) -> percept

* Cuneate (CN): fixed. Local convergence of mixed afferent types with a few
  dominant synapses, plus wider feedforward inhibition (centre-surround).
* Area 3b: somatotopic - each neuron only sees CN cells of one digit. Weights
  initialised as centre + offset inhibitory flank (orientation-tuned, DiCarlo 1998).
* Area 1: per-digit recurrent layer with neighbouring-digit input (larger RFs, temporal).
* Area 2: whole-hand recurrent layer integrating touch with proprioception.
* Readout: leaky non-spiking units -> "percept" vector for the RL policy.
"""

import numpy as np
import torch
from torch import nn

from .afferents import AfferentPopulation

DIGIT_ORDER = ("th", "ff", "mf", "rf", "lf", "palm")


class SpikeFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        ctx.save_for_backward(x)
        return (x > 0).float()

    @staticmethod
    def backward(ctx, g):
        (x,) = ctx.saved_tensors
        return g / (1 + 10 * x.abs()) ** 2


spike = SpikeFn.apply


class LIF(nn.Module):
    def __init__(self, n, tau=0.02, dt=0.002):
        super().__init__()
        self.n, self.beta = n, float(np.exp(-dt / tau))

    def forward(self, I, v):
        v = self.beta * v + (1 - self.beta) * I
        s = spike(v - 1.0)
        return s, v - s.detach() * v   # reset to 0


class SomatosensoryCortex(nn.Module):
    def __init__(self, aff: AfferentPopulation, n_proprio: int, cn_ratio=0.3,
                 n_3b=128, n_1=64, n_2=256, n_out=64, dt=0.002, seed=0):
        super().__init__()
        g = np.random.default_rng(seed)
        self.dt, self.n_aff = dt, aff.n
        self.syn_beta = float(np.exp(-dt / 0.005))

        # ---- cuneate nucleus (fixed, sparse) ----
        cn_pos, cn_digit, er, ec, ev, ir, ic, iv = [], [], [], [], [], [], [], []
        j = 0
        for b in np.unique(aff.body):
            ids = np.flatnonzero(aff.body == b)
            sig = 2e-3 if aff.region[ids[0]] == "fingertip" else 5e-3
            c = aff.pos[g.choice(ids, max(1, int(len(ids) * cn_ratio)), replace=False)]
            nc = len(c)
            d2 = ((c[:, None] - aff.pos[ids][None]) ** 2).sum(-1)
            we = np.exp(-d2 / (2 * sig ** 2)) * g.lognormal(0, 1.0, d2.shape)
            kk = min(30, len(ids))
            top = np.argpartition(we, -kk, axis=1)[:, -kk:]
            wt = np.take_along_axis(we, top, 1); wt /= wt.sum(1, keepdims=True)
            er.append(np.repeat(np.arange(nc) + j, kk)); ec.append(ids[top].ravel()); ev.append(wt.ravel())
            wi = np.where(d2 < (7.5 * sig) ** 2, np.exp(-d2 / (2 * (2.5 * sig) ** 2)), 0)
            wi /= wi.sum(1, keepdims=True)
            r, cc = np.nonzero(wi)
            ir.append(r + j); ic.append(ids[cc]); iv.append(wi[r, cc])
            cn_pos.append(c); cn_digit += [aff.digit[ids[0]]] * nc; j += nc
        er, ec, ev, ir, ic, iv = map(np.concatenate, (er, ec, ev, ir, ic, iv))
        cn_pos = np.concatenate(cn_pos)
        self.n_cn = j
        mk = lambda r, c, v: torch.sparse_coo_tensor(np.stack([r, c]), torch.tensor(v, dtype=torch.float32),
                                                     (j, aff.n), check_invariants=False).coalesce()
        self.register_buffer("W_cn_e", mk(er, ec, ev))
        self.register_buffer("W_cn_i", mk(ir, ic, iv))
        self.cn_gain, self.cn_inh = 60.0, 0.6
        self.cn = LIF(j, tau=0.01, dt=dt)
        cn_digit = np.array(cn_digit)

        # ---- area 3b: digit-masked, centre/flank init ----
        nd = len(DIGIT_ORDER)
        W3 = np.zeros((n_3b * nd, j)); M3 = np.zeros_like(W3)
        for di, dg in enumerate(DIGIT_ORDER):
            cols = np.flatnonzero(cn_digit == dg)
            if len(cols) == 0:
                continue
            M3[di * n_3b:(di + 1) * n_3b, cols] = 1
            c = cn_pos[g.choice(cols, n_3b)]
            theta = g.normal(size=(n_3b, 3)); theta /= np.linalg.norm(theta, axis=1, keepdims=True)
            d = cn_pos[cols][None] - c[:, None]
            exc = np.exp(-(d ** 2).sum(-1) / (2 * 1.5e-3 ** 2))
            flank = np.exp(-((d - 2.5e-3 * theta[:, None]) ** 2).sum(-1) / (2 * 2e-3 ** 2))
            W3[di * n_3b:(di + 1) * n_3b, cols] = 3.0 * exc - 1.5 * flank
        self.W_3b = nn.Parameter(torch.tensor(W3, dtype=torch.float32))
        self.register_buffer("M_3b", torch.tensor(M3, dtype=torch.float32))
        self.a3b = LIF(n_3b * nd, dt=dt)

        # ---- area 1: own digit + neighbours, recurrent ----
        adj = np.eye(nd) + 0.3 * (np.abs(np.subtract.outer(np.arange(nd), np.arange(nd))) == 1)
        adj[:, -1] = adj[-1, :] = 0.3; adj[-1, -1] = 1  # palm touches all
        M1 = np.kron(adj, np.ones((n_1, n_3b)))
        self.W_1 = nn.Parameter(torch.randn(n_1 * nd, n_3b * nd) * 0.3 / np.sqrt(n_3b))
        self.register_buffer("M_1", torch.tensor(M1, dtype=torch.float32))
        self.R_1 = nn.Parameter(torch.randn(n_1 * nd, n_1 * nd) * 0.1 / np.sqrt(n_1))
        self.a1 = LIF(n_1 * nd, tau=0.05, dt=dt)

        # ---- area 2: whole hand + proprioception ----
        self.W_2 = nn.Linear(n_1 * nd + n_proprio, n_2)
        self.R_2 = nn.Parameter(torch.randn(n_2, n_2) * 0.1 / np.sqrt(n_2))
        self.a2 = LIF(n_2, tau=0.1, dt=dt)

        self.readout = nn.Linear(n_2, n_out)
        self.out_beta = float(np.exp(-dt / 0.05))
        self.sizes = {"cn": j, "3b": n_3b * nd, "1": n_1 * nd, "2": n_2, "out": n_out}

    def init_state(self, batch=1):
        z = lambda n: torch.zeros(batch, n)
        return {"syn": z(self.n_aff), "v_cn": z(self.n_cn), "v_3b": z(self.sizes["3b"]),
                "v_1": z(self.sizes["1"]), "s_1": z(self.sizes["1"]), "v_2": z(self.sizes["2"]),
                "s_2": z(self.sizes["2"]), "out": z(self.sizes["out"])}

    def forward(self, aff_counts, proprio, st):
        """One dt step. aff_counts (B, n_aff), proprio (B, n_proprio)."""
        syn = self.syn_beta * st["syn"] + aff_counts
        e = torch.sparse.mm(self.W_cn_e, syn.T).T
        i = torch.sparse.mm(self.W_cn_i, syn.T).T
        s_cn, v_cn = self.cn(self.cn_gain * torch.relu(e - self.cn_inh * i), st["v_cn"])
        s_3b, v_3b = self.a3b(20.0 * s_cn @ (self.W_3b * self.M_3b).T, st["v_3b"])
        s_1, v_1 = self.a1(20.0 * s_3b @ (self.W_1 * self.M_1).T + s_1_rec(st, self.R_1), st["v_1"])
        x2 = torch.cat([s_1, proprio], dim=1)
        s_2, v_2 = self.a2(5.0 * self.W_2(x2) + st["s_2"] @ self.R_2.T, st["v_2"])
        out = self.out_beta * st["out"] + (1 - self.out_beta) * self.readout(s_2)
        new = {"syn": syn, "v_cn": v_cn, "v_3b": v_3b, "v_1": v_1, "s_1": s_1,
               "v_2": v_2, "s_2": s_2, "out": out}
        spikes = {"cn": s_cn, "3b": s_3b, "1": s_1, "2": s_2}
        return out, new, spikes


def s_1_rec(st, R):
    return st["s_1"] @ R.T


def encode_proprio(q, qd, q_lo, q_hi, n_pref=6, width=0.25):
    """Muscle-spindle-like population code: Gaussian tuning over joint range + velocity."""
    u = (q - q_lo) / np.maximum(q_hi - q_lo, 1e-6)
    centers = np.linspace(0, 1, n_pref)
    pos = np.exp(-(u[:, None] - centers) ** 2 / (2 * width ** 2)).ravel()
    vel = np.concatenate([np.maximum(qd, 0), np.maximum(-qd, 0)]) * 0.1
    return np.concatenate([pos, vel]).astype(np.float32)

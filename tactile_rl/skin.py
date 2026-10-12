"""Skin model: taxels on the hand's collision surfaces + Hertzian contact pressure.

MuJoCo gives point contacts. Each contact force is spread over nearby taxels with
a Hertz pressure profile p(r) = p0 sqrt(1 - r^2/a^2), whose radius a grows with
force: a = (3 F R / 4 E*)^(1/3). Force is conserved over the taxels.
"""

from dataclasses import dataclass, field

import mujoco
import numpy as np

# body-name fragment -> (region, taxel spacing m, effective skin radius R m)
REGIONS = {
    "distal": ("fingertip", 0.0012, 0.008),
    "middle": ("finger", 0.002, 0.009),
    "proximal": ("finger", 0.002, 0.009),
    "palm": ("palm", 0.003, 0.02),
    "metacarpal": ("palm", 0.003, 0.02),
}
DIGITS = ("th", "ff", "mf", "rf", "lf")


@dataclass
class SkinParams:
    E_eff: float = 2e5          # effective skin modulus (Pa)
    a_min: float = 0.5e-3       # contact radius bounds (m)
    a_max: float = 8e-3
    glabrous_cos: float = -0.2  # taxel normal · palmar dir above this = glabrous
    seed: int = 0


def _palmar_dir(body_name: str) -> np.ndarray:
    # Derived from the grasp pose (direction from segment to object, body frame).
    if body_name in ("rh_thproximal", "rh_thmiddle"):
        return np.array([-1.0, 0.0, 0.0])
    return np.array([0.0, -1.0, 0.0])


def _sample_geom_surface(model, g, n, rng):
    """Uniform-ish surface samples (points, normals) in the geom frame."""
    t, s = model.geom_type[g], model.geom_size[g]
    G = mujoco.mjtGeom
    if t == G.mjGEOM_SPHERE:
        nrm = rng.normal(size=(n, 3)); nrm /= np.linalg.norm(nrm, axis=1, keepdims=True)
        return nrm * s[0], nrm
    if t == G.mjGEOM_CAPSULE:
        r, h = s[0], s[1]
        nrm = rng.normal(size=(n, 3)); nrm /= np.linalg.norm(nrm, axis=1, keepdims=True)
        z = rng.uniform(-h - r, h + r, n)
        side = np.abs(z) <= h
        nrm[side, 2] = 0; nrm[side] /= np.linalg.norm(nrm[side], axis=1, keepdims=True)
        pts = nrm * r
        pts[:, 2] += np.where(side, z, np.sign(nrm[:, 2]) * h)
        return pts, nrm
    if t == G.mjGEOM_CYLINDER:
        r, h = s[0], s[1]
        phi = rng.uniform(0, 2 * np.pi, n); z = rng.uniform(-h, h, n)
        nrm = np.stack([np.cos(phi), np.sin(phi), np.zeros(n)], 1)
        return nrm * r + np.stack([0 * z, 0 * z, z], 1), nrm
    if t == G.mjGEOM_BOX:
        areas = np.array([s[1] * s[2], s[0] * s[2], s[0] * s[1]]).repeat(2)
        face = rng.choice(6, n, p=areas / areas.sum())
        ax, sign = face // 2, np.where(face % 2, 1.0, -1.0)
        pts = rng.uniform(-1, 1, (n, 3)) * s
        pts[np.arange(n), ax] = sign * s[ax]
        nrm = np.zeros((n, 3)); nrm[np.arange(n), ax] = sign
        return pts, nrm
    if t == G.mjGEOM_MESH:
        mid = model.geom_dataid[g]
        v = model.mesh_vert[model.mesh_vertadr[mid]: model.mesh_vertadr[mid] + model.mesh_vertnum[mid]]
        f = model.mesh_face[model.mesh_faceadr[mid]: model.mesh_faceadr[mid] + model.mesh_facenum[mid]]
        a, b, c = v[f[:, 0]], v[f[:, 1]], v[f[:, 2]]
        cr = np.cross(b - a, c - a); area = np.linalg.norm(cr, axis=1)
        idx = rng.choice(len(f), n, p=area / area.sum())
        u, w = rng.random(n), rng.random(n)
        flip = u + w > 1; u[flip], w[flip] = 1 - u[flip], 1 - w[flip]
        pts = a[idx] + u[:, None] * (b - a)[idx] + w[:, None] * (c - a)[idx]
        nrm = cr[idx] / np.maximum(area[idx, None], 1e-12)
        return pts, nrm
    raise ValueError(f"unsupported geom type {t}")


@dataclass
class Skin:
    model: mujoco.MjModel
    params: SkinParams = field(default_factory=SkinParams)

    def __post_init__(self):
        m, rng = self.model, np.random.default_rng(self.params.seed)
        pos, nrm, body, region, R, area = [], [], [], [], [], []
        self.body_ids = []
        for b in range(m.nbody):
            name = m.body(b).name
            key = next((k for k in REGIONS if k in name), None)
            if not name.startswith("rh_") or key is None:
                continue
            reg, spacing, Rb = REGIONS[key]
            geoms = [g for g in range(m.ngeom) if m.geom_bodyid[g] == b and m.geom_contype[g]]
            P, N = [], []
            for g in geoms:
                p, n = _sample_geom_surface(m, g, 20000, rng)
                Rg = np.zeros(9); mujoco.mju_quat2Mat(Rg, m.geom_quat[g]); Rg = Rg.reshape(3, 3)
                P.append(p @ Rg.T + m.geom_pos[g]); N.append(n @ Rg.T)
            P, N = np.concatenate(P), np.concatenate(N)
            # voxel downsample to the region's spacing
            _, keep = np.unique(np.floor(P / spacing).astype(np.int64), axis=0, return_index=True)
            P, N = P[keep], N[keep]
            self.body_ids.append(b)
            pos.append(P); nrm.append(N); body += [b] * len(P)
            region += [reg] * len(P); R += [Rb] * len(P); area += [spacing ** 2] * len(P)
        self.pos = np.concatenate(pos)          # body frame (m)
        self.normal = np.concatenate(nrm)
        self.body = np.array(body)
        self.region = np.array(region)
        self.R = np.array(R)
        self.area = np.array(area)
        self.n = len(self.pos)
        names = [m.body(b).name for b in self.body]
        self.digit = np.array([next((dg for dg in DIGITS if n.startswith("rh_" + dg)), "palm") for n in names])
        palmar = np.array([_palmar_dir(n) for n in names])
        self.glabrous = (self.normal * palmar).sum(1) > self.params.glabrous_cos
        self._idx = {b: np.flatnonzero(self.body == b) for b in self.body_ids}
        self.pressure = np.zeros(self.n)        # Pa
        self.shear = np.zeros((self.n, 3))      # Pa, body frame

    def update(self, data: mujoco.MjData):
        """Recompute taxel pressure/shear from the current contacts."""
        m, prm = self.model, self.params
        self.pressure[:] = 0; self.shear[:] = 0
        f6 = np.zeros(6)
        for i in range(data.ncon):
            c = data.contact[i]
            b1, b2 = m.geom_bodyid[c.geom1], m.geom_bodyid[c.geom2]
            for b, sign in ((b1, -1.0), (b2, 1.0)):
                if b not in self._idx:
                    continue
                mujoco.mj_contactForce(m, data, i, f6)
                if f6[0] <= 0:
                    continue
                frame = c.frame.reshape(3, 3)
                Rb = data.xmat[b].reshape(3, 3)
                p_loc = Rb.T @ (c.pos - data.xpos[b])
                n_loc = Rb.T @ (-sign * frame[0])     # outward skin normal at contact
                ft_loc = Rb.T @ (sign * (f6[1] * frame[1] + f6[2] * frame[2]))
                idx = self._idx[b]
                F = f6[0]
                a = np.clip((3 * F * self.R[idx[0]] / (4 * prm.E_eff)) ** (1 / 3), prm.a_min, prm.a_max)
                r2 = ((self.pos[idx] - p_loc) ** 2).sum(1)
                facing = self.normal[idx] @ n_loc > 0
                w = np.where((r2 < a * a) & facing, np.sqrt(np.clip(1 - r2 / (a * a), 0, 1)), 0.0)
                if w.sum() == 0:
                    w[np.argmin(r2)] = 1.0
                w = w / (w * self.area[idx]).sum()
                self.pressure[idx] += F * w
                self.shear[idx] += w[:, None] * ft_loc[None]
        return self.pressure, self.shear

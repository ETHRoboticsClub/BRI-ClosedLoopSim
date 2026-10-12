"""MuJoCo scene: Shadow Hand E3M5 (right) with a graspable object resting on the palm.

The menagerie mount puts the palm facing world +z, so an object dropped above
``grasp_site`` settles in the palm and the fingers can close around it.
"""

from dataclasses import dataclass
from pathlib import Path

import mujoco
import numpy as np

ASSETS = Path(__file__).resolve().parent.parent / "assets" / "shadow_hand"
HAND_XML = ASSETS / "right_hand.xml"
KEYFRAMES_XML = ASSETS / "keyframes.xml"

HAND_PREFIX = "rh_"


@dataclass
class ObjectSpec:
    shape: str = "sphere"           # sphere | box | cylinder | ellipsoid
    size: tuple = (0.03,)           # MuJoCo geom size (m)
    mass: float = 0.08              # kg
    friction: tuple = (1.0, 0.01, 0.001)
    rgba: tuple = (0.75, 0.45, 0.3, 1.0)
    offset: tuple = (0.0, 0.0, 0.0)  # object centre relative to grasp_site (m)


def build_scene(obj: ObjectSpec | None = None) -> mujoco.MjModel:
    obj = obj or ObjectSpec()
    spec = mujoco.MjSpec.from_file(str(HAND_XML))
    spec.meshdir = str(ASSETS / "assets")

    spec.visual.global_.azimuth = 200
    spec.visual.global_.elevation = -25
    spec.stat.extent = 0.3
    spec.stat.center = [0.33, 0.0, 0.05]

    world = spec.worldbody
    world.add_light(pos=[0.3, 0, 1.0], dir=[0, 0, -1], diffuse=[0.8, 0.8, 0.8])
    world.add_geom(type=mujoco.mjtGeom.mjGEOM_PLANE, size=[0, 0, 0.05], pos=[0, 0, -0.15],
                   rgba=[0.25, 0.3, 0.35, 1])

    # grasp_site is at palm-local (0, -0.035, 0.09); palm frame is fixed at
    # the default (zero) wrist angles, so this world position is stable.
    grasp_site = np.array([0.33701, 0.0, 0.045])
    obj_pos = (grasp_site + np.asarray(obj.offset)).tolist()
    body = world.add_body(name="object", pos=obj_pos)
    body.add_freejoint(name="object_free")
    shape = {
        "sphere": mujoco.mjtGeom.mjGEOM_SPHERE,
        "box": mujoco.mjtGeom.mjGEOM_BOX,
        "cylinder": mujoco.mjtGeom.mjGEOM_CYLINDER,
        "ellipsoid": mujoco.mjtGeom.mjGEOM_ELLIPSOID,
    }[obj.shape]
    size = list(obj.size) + [0.0] * (3 - len(obj.size))
    body.add_geom(name="object_geom", type=shape, size=size, mass=obj.mass,
                  friction=list(obj.friction), rgba=list(obj.rgba),
                  condim=6, priority=1)

    # "Virtual experimenter": holds the object in place until released
    # (toggle with data.eq_active[holder_eq_id]).
    spec.add_equality(name="holder", type=mujoco.mjtEq.mjEQ_WELD,
                      objtype=mujoco.mjtObj.mjOBJ_BODY, name1="object", name2="world",
                      solref=[0.02, 1.0])

    model = spec.compile()
    return model


def load_keyframes(model: mujoco.MjModel) -> dict[str, np.ndarray]:
    """Hand-joint keyframes from the menagerie file, as {name: qpos(24,)}."""
    import xml.etree.ElementTree as ET

    root = ET.parse(KEYFRAMES_XML).getroot()
    return {k.get("name"): np.array(k.get("qpos").split(), dtype=float) for k in root.iter("key")}


def hand_joint_qpos_adr(model: mujoco.MjModel) -> np.ndarray:
    """qpos addresses of the 24 hand joints, in model (= keyframe) order."""
    ids = [j for j in range(model.njnt) if model.joint(j).name.startswith(HAND_PREFIX)]
    return model.jnt_qposadr[ids]


def hand_joint_dof_adr(model: mujoco.MjModel) -> np.ndarray:
    ids = [j for j in range(model.njnt) if model.joint(j).name.startswith(HAND_PREFIX)]
    return model.jnt_dofadr[ids]


def qpos_to_ctrl(model: mujoco.MjModel, hand_qpos: np.ndarray) -> np.ndarray:
    """Map 24 hand-joint angles to the 20 position-actuator targets.

    Coupled J1/J2 finger joints are driven through fixed tendons (J0 = J1 + J2).
    """
    names = [model.joint(j).name for j in range(model.njnt) if model.joint(j).name.startswith(HAND_PREFIX)]
    q = dict(zip(names, hand_qpos))
    ctrl = np.zeros(model.nu)
    for a in range(model.nu):
        target = model.actuator(a).name.replace("rh_A_", "rh_")
        if target.endswith("J0"):
            f = target[:-2]
            ctrl[a] = q[f + "J1"] + q[f + "J2"]
        else:
            ctrl[a] = q[target]
    return np.clip(ctrl, model.actuator_ctrlrange[:, 0], model.actuator_ctrlrange[:, 1])

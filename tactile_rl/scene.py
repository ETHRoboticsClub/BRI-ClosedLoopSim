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
BIGYM_MUG_ASSETS = ASSETS.parent / "third_party" / "bigym_mug" / "assets"

HAND_PREFIX = "rh_"
MUG_TABLE_TOP_Z = -0.08
MUG_LIFT_MAX_M = 0.12


@dataclass
class ObjectSpec:
    shape: str = "sphere"           # sphere | box | cylinder | ellipsoid | mug
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

    # grasp_site is at palm-local (0, -0.035, 0.09); the hand's fixed starting
    # pose puts it in front of this location.
    grasp_site = np.array([0.33701, 0.0, 0.045])
    if obj.shape == "mug":
        radius, height = obj.size
        table_top_z = MUG_TABLE_TOP_Z + obj.offset[2]
        obj_pos = [
            grasp_site[0] + obj.offset[0],
            grasp_site[1] + obj.offset[1],
            table_top_z + height / 2,
        ]
        world.add_geom(
            name="mug_table",
            type=mujoco.mjtGeom.mjGEOM_BOX,
            size=[0.14, 0.14, 0.01],
            pos=[obj_pos[0], obj_pos[1], table_top_z - 0.01],
            rgba=[0.35, 0.28, 0.22, 1.0],
            contype=4,
            conaffinity=2,
        )
        mesh_scale = [radius / 0.04531, height / 0.11375, radius / 0.04531]
        visual_z_offset = 0.004006 * mesh_scale[1]
        spec.add_mesh(
            name="bigym_mug_visual",
            file=str(BIGYM_MUG_ASSETS / "mug.obj"),
            scale=mesh_scale,
        )
        spec.add_texture(
            name="bigym_mug_texture",
            type=mujoco.mjtTexture.mjTEXTURE_2D,
            file=str(BIGYM_MUG_ASSETS / "mug.png"),
        )
        spec.add_material(
            name="bigym_mug_material",
            textures=["bigym_mug_texture"],
            rgba=[1.0, 1.0, 1.0, 1.0],
        )
    else:
        obj_pos = (grasp_site + np.asarray(obj.offset)).tolist()

    body = world.add_body(name="object", pos=obj_pos)
    body.add_freejoint(name="object_free")
    if obj.shape == "mug":
        body.add_geom(
            name="mug_visual",
            type=mujoco.mjtGeom.mjGEOM_MESH,
            meshname="bigym_mug_visual",
            pos=[0.0, 0.0, visual_z_offset],
            quat=[0.5, -0.5, -0.5, 0.5],
            mass=0.0,
            contype=0,
            conaffinity=0,
            group=2,
            material="bigym_mug_material",
        )
        wall = 0.006
        color = list(obj.rgba)
        cup_geoms = (
            ("mug_bottom", [radius, radius, wall / 2],
             [0, 0, -height / 2 + wall / 2], 0.20),
            ("mug_wall_xp", [wall / 2, radius, height / 2],
             [radius - wall / 2, 0, 0], 0.16),
            ("mug_wall_xn", [wall / 2, radius, height / 2],
             [-radius + wall / 2, 0, 0], 0.16),
            ("mug_wall_yp", [radius - wall, wall / 2, height / 2],
             [0, radius - wall / 2, 0], 0.16),
            ("mug_wall_yn", [radius - wall, wall / 2, height / 2],
             [0, -radius + wall / 2, 0], 0.16),
        )
        for name, size, pos, mass_fraction in cup_geoms:
            body.add_geom(
                name=name,
                type=mujoco.mjtGeom.mjGEOM_BOX,
                size=size,
                pos=pos,
                mass=obj.mass * mass_fraction,
                friction=list(obj.friction),
                rgba=color,
                contype=2,
                conaffinity=1,
                condim=6,
                priority=1,
            )

        # A rounded, rigid handle extends toward the palm.
        handle_points = (
            (-radius + wall, 0, 0.025),
            (-radius - 0.03, 0, 0.025),
            (-radius - 0.03, 0, -0.025),
            (-radius + wall, 0, -0.025),
            (-radius + wall, 0, 0.025),
        )
        for index, (start, end) in enumerate(zip(handle_points, handle_points[1:])):
            body.add_geom(
                name=f"mug_handle_{index}",
                type=mujoco.mjtGeom.mjGEOM_CAPSULE,
                size=[0.004],
                fromto=[*start, *end],
                mass=obj.mass * 0.04,
                friction=list(obj.friction),
                rgba=color,
                contype=2,
                conaffinity=1,
                condim=6,
                priority=1,
            )
    else:
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

        # The virtual experimenter holds ordinary test objects until release.
        spec.add_equality(name="holder", type=mujoco.mjtEq.mjEQ_WELD,
                          objtype=mujoco.mjtObj.mjOBJ_BODY, name1="object", name2="world",
                          solref=[0.02, 1.0])

    if obj.shape == "mug":
        # Add simple forearm translations for the mug setup and vertical lift.
        forearm = spec.body("rh_forearm")
        forearm.add_joint(
            name="hand_approach",
            type=mujoco.mjtJoint.mjJNT_SLIDE,
            axis=[0, 0, 1],
            limited=True,
            range=[0.0, 0.16],
            damping=[2.0],
        )
        forearm.add_joint(
            name="hand_lift",
            type=mujoco.mjtJoint.mjJNT_SLIDE,
            axis=[1, 0, 0],
            limited=True,
            range=[0.0, MUG_LIFT_MAX_M],
            damping=[2.0],
        )
        position_gain = [2000.0] + [0.0] * 9
        position_bias = [0.0, -2000.0, -50.0] + [0.0] * 7
        spec.add_actuator(
            name="hand_approach_act",
            target="hand_approach",
            trntype=mujoco.mjtTrn.mjTRN_JOINT,
            gaintype=mujoco.mjtGain.mjGAIN_FIXED,
            gainprm=position_gain,
            biastype=mujoco.mjtBias.mjBIAS_AFFINE,
            biasprm=position_bias,
            ctrllimited=True,
            ctrlrange=[0.0, 0.16],
            forcelimited=True,
            forcerange=[-500.0, 500.0],
        )
        spec.add_actuator(
            name="hand_lift_act",
            target="hand_lift",
            trntype=mujoco.mjtTrn.mjTRN_JOINT,
            gaintype=mujoco.mjtGain.mjGAIN_FIXED,
            gainprm=position_gain,
            biastype=mujoco.mjtBias.mjBIAS_AFFINE,
            biasprm=position_bias,
            ctrllimited=True,
            ctrlrange=[0.0, MUG_LIFT_MAX_M],
            forcelimited=True,
            forcerange=[-500.0, 500.0],
        )

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
        actuator_name = model.actuator(a).name
        if not actuator_name.startswith("rh_A_"):
            continue
        target = actuator_name.replace("rh_A_", "rh_")
        if target.endswith("J0"):
            f = target[:-2]
            ctrl[a] = q[f + "J1"] + q[f + "J2"]
        else:
            ctrl[a] = q[target]
    return np.clip(ctrl, model.actuator_ctrlrange[:, 0], model.actuator_ctrlrange[:, 1])

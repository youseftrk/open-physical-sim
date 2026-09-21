"""Load Env Commons-style scene packages or local MJCF fixtures.

Env Commons (`env.json`) and Reconstruct (`scene.json`) both declare a
**collision** layer (mesh / mesh_urdf) separate from **visual** (3DGS).

For MuJoCo v0 we only consume collision geometry (or a bundled empty-room
MJCF fixture). Visual splat paths are recorded but not rendered.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

# Package root: .../open-physical-sim
_PKG_ROOT = Path(__file__).resolve().parents[3]
_DEFAULT_EMPTY_MJCF = _PKG_ROOT / "assets" / "scenes" / "empty_room" / "scene.xml"
_DEFAULT_ROBOT_MJCF = _PKG_ROOT / "assets" / "robots" / "simple_arm" / "robot.xml"


@dataclass
class ScenePackage:
    """Resolved scene for the MuJoCo loader."""

    root: Path
    manifest: dict[str, Any] = field(default_factory=dict)
    collision_mjcf: Optional[Path] = None
    collision_mesh: Optional[Path] = None
    visual_path: Optional[Path] = None
    robot_mjcf: Optional[Path] = None
    units: str = "meters"
    up_axis: str = "z"
    source: str = "fixture"  # fixture | env_commons | reconstruct

    @property
    def mjcf_path(self) -> Path:
        if self.collision_mjcf is not None and self.collision_mjcf.exists():
            return self.collision_mjcf
        raise FileNotFoundError(
            f"No collision MJCF resolved under {self.root} "
            "(expected assets/collision/*.xml or bundled empty_room fixture)"
        )


def _read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return data


def _find_first(directory: Path, patterns: tuple[str, ...]) -> Optional[Path]:
    if not directory.is_dir():
        return None
    for pat in patterns:
        hits = sorted(directory.glob(pat))
        if hits:
            return hits[0]
    return None


def load_scene_package(
    path: str | Path | None = None,
    *,
    robot_path: str | Path | None = None,
) -> ScenePackage:
    """Load an Env Commons package, Reconstruct scene.json, or default fixture.

    Parameters
    ----------
    path:
        Directory containing ``env.json`` / ``scene.json``, a path to those
        manifests, an MJCF/XML file, or ``None`` for the bundled empty room.
    robot_path:
        Optional robot MJCF/URDF. Defaults to the bundled simple 2-link arm.
    """
    robot = Path(robot_path) if robot_path else _DEFAULT_ROBOT_MJCF

    if path is None:
        return ScenePackage(
            root=_DEFAULT_EMPTY_MJCF.parent,
            collision_mjcf=_DEFAULT_EMPTY_MJCF,
            robot_mjcf=robot if robot.exists() else None,
            source="fixture",
        )

    p = Path(path).expanduser().resolve()

    # Direct MJCF / XML
    if p.is_file() and p.suffix.lower() in {".xml", ".mjcf"}:
        return ScenePackage(
            root=p.parent,
            collision_mjcf=p,
            robot_mjcf=robot if robot.exists() else None,
            source="fixture",
        )

    # Manifest file
    if p.is_file() and p.name in {"env.json", "scene.json"}:
        root = p.parent
        manifest = _read_json(p)
        source = "env_commons" if p.name == "env.json" else "reconstruct"
    elif p.is_dir():
        root = p
        if (p / "env.json").is_file():
            manifest = _read_json(p / "env.json")
            source = "env_commons"
        elif (p / "scene.json").is_file():
            manifest = _read_json(p / "scene.json")
            source = "reconstruct"
        else:
            mjcf = _find_first(p, ("*.xml", "**/*.xml"))
            return ScenePackage(
                root=root,
                collision_mjcf=mjcf or _DEFAULT_EMPTY_MJCF,
                robot_mjcf=robot if robot.exists() else None,
                source="fixture",
            )
    else:
        raise FileNotFoundError(f"Scene path not found: {p}")

    units = "meters"
    up_axis = "z"
    collision_mjcf: Optional[Path] = None
    collision_mesh: Optional[Path] = None
    visual_path: Optional[Path] = None

    if source == "env_commons":
        assets = manifest.get("assets", {})
        sim = manifest.get("sim", {})
        units = str(sim.get("units", units))
        up_axis = str(sim.get("up_axis", up_axis))
        col = assets.get("collision", {})
        col_rel = col.get("path", "assets/collision/")
        col_dir = (root / col_rel).resolve() if col_rel else root / "assets" / "collision"
        collision_mjcf = _find_first(col_dir, ("*.xml", "**/*.xml", "*.mjcf"))
        collision_mesh = _find_first(
            col_dir, ("*.obj", "*.stl", "*.urdf", "**/*.obj", "**/*.stl")
        )
        vis = assets.get("visual", {})
        vis_rel = vis.get("path")
        if vis_rel:
            visual_path = (root / vis_rel).resolve()
    else:  # reconstruct scene.json
        wf = manifest.get("world_frame", {})
        up_axis = str(wf.get("up_axis", up_axis))
        units = str(manifest.get("units", "m"))
        if units in {"m", "meter"}:
            units = "meters"
        col = manifest.get("collision", {})
        mesh_rel = col.get("mesh_path") or col.get("path")
        if mesh_rel:
            collision_mesh = (root / mesh_rel).resolve()
            sibling = collision_mesh.with_suffix(".xml")
            if sibling.is_file():
                collision_mjcf = sibling
            elif collision_mesh.is_file() and collision_mesh.suffix.lower() in {".obj", ".stl"}:
                collision_mjcf = _mjcf_for_mesh(collision_mesh)
        vis = manifest.get("visual", {})
        splat = vis.get("splat_path")
        if splat:
            visual_path = (root / splat).resolve()
        # Prefer textured/proxy mesh path for discovery; points.ply ok as discovery only
        proxy = vis.get("proxy_mesh_path") or vis.get("proxy_path") or vis.get("proxy_mesh")
        if proxy and visual_path is None:
            visual_path = (root / proxy).resolve()

    if collision_mjcf is None:
        # Last resort: any collision mesh under package, else empty room fixture
        if collision_mesh is not None and collision_mesh.is_file():
            collision_mjcf = _mjcf_for_mesh(collision_mesh)
        else:
            collision_mjcf = _DEFAULT_EMPTY_MJCF

    return ScenePackage(
        root=root,
        manifest=manifest,
        collision_mjcf=collision_mjcf,
        collision_mesh=collision_mesh,
        visual_path=visual_path,
        robot_mjcf=robot if robot.exists() else None,
        units=units,
        up_axis=up_axis,
        source=source,
    )


def _inner_xml(path: Path, tag: str) -> str:
    """Extract concatenated children of all <tag>...</tag> blocks."""
    raw = path.read_text(encoding="utf-8")
    raw = re.sub(r"<!--.*?-->", "", raw, flags=re.DOTALL)
    parts: list[str] = []
    for m in re.finditer(
        rf"<{tag}\b[^>]*>(.*?)</{tag}>", raw, flags=re.DOTALL | re.IGNORECASE
    ):
        parts.append(m.group(1).strip())
    return "\n".join(parts)


def _mjcf_for_mesh(mesh_path: Path, out_dir: Path | None = None) -> Path:
    """Write a tiny MJCF that loads a collision mesh (OBJ/STL) as a static geom."""
    mesh_path = mesh_path.resolve()
    dest = out_dir or (mesh_path.parent / ".physical_sim_build")
    dest.mkdir(parents=True, exist_ok=True)
    out = dest / f"{mesh_path.stem}_collision.xml"
    # meshdir = parent of mesh so <mesh file="name.obj"/> resolves
    rel = mesh_path.name
    out.write_text(
        f"""<?xml version="1.0" ?>
<mujoco model="{mesh_path.stem}_collision">
  <compiler angle="radian" meshdir="{mesh_path.parent.as_posix()}" autolimits="true"/>
  <option timestep="0.002" gravity="0 0 -9.81"/>
  <asset>
    <mesh name="collision_mesh" file="{rel}"/>
  </asset>
  <worldbody>
    <light pos="0 0 4" dir="0 0 -1" diffuse="0.8 0.8 0.8"/>
    <geom name="collision_shell" type="mesh" mesh="collision_mesh"
          rgba="0.7 0.7 0.75 0.35" contype="1" conaffinity="1"
          friction="1 0.005 0.0001" condim="3"/>
  </worldbody>
</mujoco>
""",
        encoding="utf-8",
    )
    return out



def resolve_mjcf(
    scene: ScenePackage,
    *,
    include_robot: bool = True,
    out_dir: Path | None = None,
) -> Path:
    """Return an MJCF path ready for ``mujoco.MjModel.from_xml_path``.

    Writes a single self-contained model merging room + robot worldbody /
    actuator content (avoids nested multi-root ``<include>``).

    If ``scene`` and ``robot`` live in the same asset package and the scene is
    a menagerie-style ``scene.xml`` (already ``<include>``s the robot), return
    that scene path unchanged so mesh ``meshdir`` resolution keeps working.
    """
    scene_xml = scene.mjcf_path
    if not include_robot or scene.robot_mjcf is None:
        return scene_xml

    robot_xml = scene.robot_mjcf.resolve()
    if robot_xml.suffix.lower() == ".urdf":
        robot_xml = _DEFAULT_ROBOT_MJCF

    # Self-contained robot package (e.g. unitree_h1/scene.xml + h1.xml).
    try:
        if (
            scene_xml.resolve().parent == robot_xml.parent
            and scene_xml.name == "scene.xml"
            and robot_xml.suffix.lower() in {".xml", ".mjcf"}
        ):
            return scene_xml.resolve()
    except OSError:
        pass

    room_src = scene_xml
    for cand in (
        scene_xml.with_name("room_fragment.xml"),
        scene_xml.with_name(f"{scene_xml.stem}_fragment.xml"),
    ):
        if cand.is_file():
            room_src = cand
            break

    robot_src = robot_xml
    for cand in (
        robot_xml.with_name("robot_fragment.xml"),
        robot_xml.with_name(f"{robot_xml.stem}_fragment.xml"),
    ):
        if cand.is_file():
            robot_src = cand
            break

    room_wb = _inner_xml(room_src, "worldbody")
    robot_wb = _inner_xml(robot_src, "worldbody")
    robot_act = _inner_xml(robot_src, "actuator")
    robot_default = _inner_xml(robot_src, "default")
    room_asset = _inner_xml(room_src, "asset")
    robot_asset = _inner_xml(robot_src, "asset")

    if not room_wb:
        return scene_xml

    # Prefer absolute meshdir of collision mesh / room MJCF so OBJ/STL resolve.
    if scene.collision_mesh is not None and scene.collision_mesh.is_file():
        meshdir = scene.collision_mesh.resolve().parent.as_posix()
    else:
        meshdir = room_src.resolve().parent.as_posix()

    dest_dir = out_dir or (scene.root / ".physical_sim_build")
    dest_dir.mkdir(parents=True, exist_ok=True)
    wrapper = dest_dir / "composed_scene.xml"
    default_block = (
        f"  <default>\n{robot_default}\n  </default>\n" if robot_default else ""
    )
    actuator_block = (
        f"  <actuator>\n{robot_act}\n  </actuator>\n" if robot_act else ""
    )
    asset_inner = "\n".join(x for x in (room_asset, robot_asset) if x)
    asset_block = f"  <asset>\n{asset_inner}\n  </asset>\n" if asset_inner else ""
    wrapper.write_text(
        f"""<?xml version="1.0" ?>
<mujoco model="physical_sim_composed">
  <compiler angle="radian" meshdir="{meshdir}" autolimits="true"/>
  <option timestep="0.002" gravity="0 0 -9.81"/>
  <visual>
    <global offwidth="64" offheight="64"/>
  </visual>
{default_block}{asset_block}  <worldbody>
{room_wb}
{robot_wb}
  </worldbody>
{actuator_block}</mujoco>
""",
        encoding="utf-8",
    )
    return wrapper


__all__ = [
    "ScenePackage",
    "load_scene_package",
    "resolve_mjcf",
    "_PKG_ROOT",
    "_DEFAULT_EMPTY_MJCF",
    "_DEFAULT_ROBOT_MJCF",
]

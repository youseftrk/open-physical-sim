# Scene packages → MuJoCo loader

`open_physical_sim.scene.loader` maps **Env Commons** and **Reconstruct** packages
into a MuJoCo collision MJCF. Visual 3DGS assets are recorded but **not** rendered
in v0 (collision + empty room only).

## Sources

| Input | Detected by | Collision used |
|-------|-------------|----------------|
| Bundled fixture | `path=None` or `assets/scenes/empty_room/` | `scene.xml` / `room_fragment.xml` |
| Env Commons package | `env.json` | `assets.collision.path` → first `*.xml` / mesh |
| Reconstruct package | `scene.json` | `collision.mesh_path` (+ sibling `.xml` if present) |
| Raw MJCF | `*.xml` file | that file |

Manifest schemas live in:

- `/workspace/open-env-commons/ENV_MANIFEST_v0.md` — `env.json`, `assets.visual` (3DGS) vs `assets.collision` (mesh_urdf)
- `/workspace/real2sim-reconstruct/docs/PACKAGE_CONTRACT.md` — `scene.json`, visual=3DGS, collision=coarse mesh

## Load path (v0)

```text
config YAML
  ├─ scene.path / env.path  →  load_scene_package()
  │                              ├─ parse env.json | scene.json | MJCF
  │                              └─ ScenePackage(collision_mjcf, visual_path, …)
  └─ robot.path             →  simple_arm MJCF (empty-v0)
                                 or H1/G1 stub URDF (future configs only)

resolve_mjcf(scene)
  └─ composed_scene.xml = room_fragment + robot_fragment
       └─ mujoco.MjModel.from_xml_path(...)   # headless, no EGL
```

## What MuJoCo gets

- **Collision / room shell**: plane floor + optional walls, or a coarse mesh when
  Reconstruct/Env Commons ship one under `assets/collision/`.
- **Robot**: `assets/robots/simple_arm/` for `PhysicalSim/empty-v0`.
- **Not loaded in v0**: 3DGS / NeRF under `assets/visual/` (path kept on
  `ScenePackage.visual_path` for later novel-view RGB).

## Gym API

```python
import open_physical_sim  # registers PhysicalSim/empty-v0
import gymnasium as gym

env = gym.make("PhysicalSim/empty-v0", config_path="configs/empty_placeholder.yaml")
obs, info = env.reset()
obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
```

Observation keys: `proprio` (qpos‖qvel) plus configured camera keys as zero
arrays (no offscreen renderer in v0).

## Future configs (not registered)

- `configs/h1_hello_twin.yaml` → `assets/robots/unitree_h1/h1.urdf` (stub)
- `configs/g1_livingroom_sit.yaml` → `assets/robots/unitree_g1/g1.urdf` (stub)

Do not register `PhysicalSim/g1-livingroom-sit-v0` / H1 ids until empty-v0 works
and real models are licensed in-tree.

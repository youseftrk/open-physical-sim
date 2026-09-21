> **Status:** draft — do not publish until the GitHub org (`youseftrk`) is confirmed.

# Schemas — open-physical-sim

## Gymnasium

- Env id (**exact**): `PhysicalSim/empty-v0`
- Install: `pip install -e /workspace/open-physical-sim`
- `gymnasium.make("PhysicalSim/empty-v0", config_path=...)`
- `reset() -> (obs, info)`, `step(action) -> (obs, reward, terminated, truncated, info)`

Configs in-tree: empty, H1 hello-twin walk, G1 sit (URDFs placeholder until licensed).

```python
import gymnasium as gym

env = gym.make("PhysicalSim/empty-v0", config_path="configs/empty.yaml")
obs, info = env.reset()
obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
```

## Scene package (v0 loader)

Aligned with Reconstruct `PACKAGE_CONTRACT.md` + Sim Runtime lock:

| Asset | Path | Notes |
|-------|------|-------|
| Hero visual | `visual/splat/gaussians.ply` | **No in-sim 3DGS rasterize in v0** |
| Textured proxy | `visual/proxy/` | **Always ship** |
| Collision | `collision/room_shell.obj` | **Always ship** |
| Metadata | `scene.json` | units m; world RH Z-up; floor≈0; RDF→world 4×4 |

Domain randomization may only touch declared visual/sensor layers — never remesh captured collision.

### Folder layout (Reconstruct export)

```
scene_pkg/
  scene.json
  visual/
    splat/
      gaussians.ply
      train_config.yaml
      render_meta.json
    proxy/                       # required textured / decimated mesh or points
    nerf/                        # optional
  collision/
    room_shell.obj               # meters; after world align
  materials/
  transforms/
    world_frame.json
    capture_rdf_to_world.json    # RDF optical → export world (4x4)
  sensors_synth/
    default_rgbd_camera.json
  provenance/
  qa/
```

### `scene.json` world frame (locked proposal → runtime)

```json
{
  "package_version": "0.1.0-draft",
  "scene_id": "string",
  "source_session_id": "uuid-from-capture-manifest",
  "units": "m",
  "world_frame": {
    "convention": "right_handed_z_up",
    "up_axis": "z",
    "floor_z": 0.0,
    "gravity_dir": [0, 0, -1]
  },
  "visual": {
    "primary": "3dgs",
    "splat_path": "visual/splat/gaussians.ply",
    "proxy_path": "visual/proxy/"
  },
  "collision": {
    "mesh_path": "collision/room_shell.obj",
    "wireframe_source": "collision_mesh"
  }
}
```

Capture cameras remain RDF optical; Reconstruct writes the RDF→world 4×4 into `scene.json` / `transforms/capture_rdf_to_world.json`.

## Loader detection

| Input | Detected by | Collision used |
|-------|-------------|----------------|
| Bundled fixture | empty-room assets | bundled MJCF |
| Env Commons package | `env.json` | `assets.collision.path` |
| Reconstruct package | `scene.json` | `collision.mesh_path` → `room_shell.obj` |
| Raw MJCF | `*.xml` | that file |

## Demo Zarr (Policy Eng)

```
runs/<run_id>/
  meta.json
  episodes/<ep_id>/
    timestamps, actions, rewards?, terminated, truncated
    obs/proprio, obs/head_rgb?, obs/head_depth?
    info/success
```

## pyproject

- Package: `open_physical_sim`
- Entry: `"PhysicalSim/empty-v0" = "open_physical_sim.envs.empty:EmptyRoomEnv"`

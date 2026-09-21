> **Status:** draft — do not publish until the GitHub org (`youseftrk`) is confirmed.

# Architecture — open-physical-sim

Physical Sim is the **runtime** that loads reconstructed deployment-site twins into gymnasium-compatible environments for policy training and eval.

Python package: `open_physical_sim` (Apache-2.0). Gymnasium id (exact): **`PhysicalSim/empty-v0`**.

## Install (v0 stub)

```bash
pip install -e /workspace/open-physical-sim   # local editable; youseftrk TBD
```

Then:

```python
import gymnasium as gym

env = gym.make("PhysicalSim/empty-v0", config_path="configs/empty.yaml")
obs, info = env.reset()
```

MuJoCo stub is headless-first. H1 hello-twin and G1 sit configs may exist in-tree with placeholder URDFs; do not register additional Gym ids until models are licensed.

## Role in Open Real2Sim

```
open-real2sim-capture → open-real2sim-reconstruct → open-physical-sim → policies
                                              ↕
                                        open-env-commons
```

Locked public repos: capture, reconstruct, physical-sim, env-commons.

- **Upstream:** Reconstruct exports a scene package (`scene.json` + hero splat + textured proxy + collision). Env Commons packages (`env.json` + `assets/`) are an alternate load root once published locally.
- **This repo:** scene loader, backend adapters, robot assets, Gymnasium entry points, optional Zarr logging for demos.
- **Downstream:** Policy Eng trains BC/RL against `PhysicalSim/<slug>-v0` with geometry-aware domain randomization that must not remesh captured collision.

## Scene package contract (loader — locked)

| Asset | Path | v0 rule |
|-------|------|---------|
| Hero visual | `visual/splat/gaussians.ply` | Ship the file; **no in-sim 3DGS rasterize** in v0 (offline / future backends only) |
| Textured proxy | `visual/proxy/` | **Always ship** (decimated mesh and/or points for viewers that cannot load splats) |
| Collision | `collision/room_shell.obj` | **Always ship** (physics + wireframe) |
| Metadata | `scene.json` | Units **meters**; world **right-handed Z-up**; floor ≈ 0; **RDF→world 4×4** matrix recorded (also under `transforms/capture_rdf_to_world.json`) |

v0 MuJoCo path loads **collision** (and optional proxy for debug). The splat path is retained on the package object; it is not rasterized inside the sim loop.

## Components

| Component | Responsibility |
|-----------|----------------|
| Gymnasium registry | Entry point `PhysicalSim/empty-v0` → `open_physical_sim.envs.empty:EmptyRoomEnv` |
| Scene loader | Detect Env Commons (`env.json`), Reconstruct (`scene.json`), or raw MJCF; resolve collision MJCF |
| Backends | **MuJoCo stub first** (headless CI); Isaac Lab smoke later; Genesis planned |
| Robot assets | Placeholder / simple arm for `empty-v0`; Unitree **H1** then **G1** as demo reference robots after smoke |
| Config YAML | `schema_version`, `env_id`, `scene`, `robot`, `task`, `domain_randomization`, `logging` |
| Zarr logging | Optional (`zarr` extra): episodes under `runs/<run_id>/episodes/<ep_id>/` |

## Data flow

1. `pip install -e` registers `PhysicalSim/empty-v0`.
2. Caller `gym.make("PhysicalSim/empty-v0", config_path=...)`.
3. Loader reads scene path: Env Commons or Reconstruct layout (or bundled empty room).
4. Requires (for real twins): `collision/room_shell.obj` + `visual/proxy/`; records `visual/splat/gaussians.ply` without rasterizing it.
5. Collision MJCF is composed with the robot fragment; MuJoCo loads headless.
6. `reset` / `step` expose proprio (+ zero camera arrays until an offscreen renderer lands).
7. Optional Zarr writer records demos for Policy Eng BC.

## Robot sequence (locked with Policy Eng)

1. `PhysicalSim/empty-v0` — minimal placeholder robot; installable first.
2. Unitree **H1** — hello-twin walk/squat smoke when URDF is licensed in-tree.
3. Unitree **G1** — first BC/eval task (living-room sit) after H1 smoke.

## Domain randomization (contract)

DR may randomize lighting, textures, and sensor noise. It must **keep** collision mesh / floor plan from the twin. Never silently remesh or rewrite captured wall layout.

## Ownership

| Surface | Owner |
|---------|-------|
| Gym env ids, MuJoCo/Isaac loaders, robot asset packaging | Sim Runtime |
| Scene package export (`scene.json`, splat, proxy, collision) | Reconstruct Eng (+ Sim Runtime on loader expectations) |
| Training loop, Zarr demo schema, BC/eval harness | Policy Eng |
| Env registry / `env.json` | Env Commons |

## Non-goals (v0)

- Training on generic empty rooms as the **final** sim2real target (stub is for wiring only).
- In-sim 3DGS / Gaussian rasterization inside MuJoCo.
- Closed cloud-only runtime.
- Marketplace or incentive programs (out of scope for v0 public docs).

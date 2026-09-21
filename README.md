# open-physical-sim

Physical sim runtime for Open Real2Sim. v0 ships a **headless MuJoCo** Gymnasium
env: empty room + minimal placeholder robot.

> Clean-room open source. Not affiliated with Inverted Lambda / InvLambda.

## Install

```bash
cd /workspace/open-physical-sim
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

Optional Zarr trajectory writer:

```bash
pip install -e ".[zarr,dev]"
```

## Hello twin

```bash
python scripts/hello_twin.py -n 20
# H1 gym env (PhysicalSim/h1-hello-twin-v0):
python scripts/smoke_h1.py --env
```

Or one-liners:

```bash
python -c "import open_physical_sim, gymnasium as gym; e=gym.make('PhysicalSim/empty-v0'); o,i=e.reset(); print(o['proprio'].shape); e.step(e.action_space.sample()); e.close()"
python -c "import open_physical_sim, gymnasium as gym; e=gym.make('PhysicalSim/h1-hello-twin-v0'); o,i=e.reset(); print(o['proprio'].shape, i.get('mjcf')); e.step(e.action_space.sample()); e.close()"
```

## Gym ids

| Id | Status |
|----|--------|
| `PhysicalSim/empty-v0` | **Registered** — empty room + `simple_arm` |
| `PhysicalSim/h1-hello-twin-v0` | **Registered** — Unitree H1 menagerie `scene.xml` (`configs/h1_hello_twin.yaml`) |
| `PhysicalSim/g1-livingroom-sit-v0` | Config stub only (`configs/g1_livingroom_sit.yaml`) |

```python
import open_physical_sim
import gymnasium as gym

env = gym.make(
    "PhysicalSim/empty-v0",
    config_path="configs/empty_placeholder.yaml",
)
h1 = gym.make("PhysicalSim/h1-hello-twin-v0")
```

## Layout

```
src/open_physical_sim/
  envs/empty.py          # EmptyRoomEnv (empty-v0)
  envs/h1_hello_twin.py  # H1HelloTwinEnv (h1-hello-twin-v0)
  scene/loader.py        # Env Commons / Reconstruct → MJCF
  logging_zarr.py        # TrajectoryWriter (no-op without zarr)
assets/
  scenes/empty_room/     # floor + walls MJCF
  robots/simple_arm/     # placeholder 2-link arm (used by empty-v0)
  robots/unitree_h1/     # MuJoCo Menagerie H1 (BSD-3-Clause MJCF+STL)
  robots/unitree_g1/     # stub URDF only
configs/
  empty_placeholder.yaml
  h1_hello_twin.yaml
  g1_livingroom_sit.yaml
```

## Docs

- [`docs/SCENE_PACKAGE.md`](docs/SCENE_PACKAGE.md) — how Env Commons packages map into the MuJoCo loader
- Env Commons contract: `/workspace/open-env-commons/ENV_MANIFEST_v0.md`
- Reconstruct package contract: `/workspace/real2sim-reconstruct/docs/PACKAGE_CONTRACT.md`

## Tests

```bash
pytest -q
```

## License

Apache-2.0 for code. Placeholder `simple_arm` / empty room are original.
Unitree H1 under `assets/robots/unitree_h1/` is **BSD-3-Clause** (MuJoCo Menagerie /
Unitree); see that directory's LICENSE + NOTICE. G1 remains a stub.

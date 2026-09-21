# Unitree H1 (MuJoCo Menagerie)

Usable public H1 humanoid for hello-twin / H1 env wiring.

## Files

| Path | Role |
|------|------|
| `h1.xml` | Robot MJCF (visual STL meshes + primitive collision) |
| `scene.xml` | Robot + floor + home keyframe (preferred smoke entry) |
| `assets/*.stl` | Visual meshes referenced by `h1.xml` (`meshdir="assets"`) |
| `LICENSE` | **BSD-3-Clause** (Unitree Robotics) |
| `CHANGELOG.md` | Upstream menagerie changelog |

`PhysicalSim/empty-v0` does **not** use this directory (it uses `simple_arm`).

## License

- **Model / meshes:** BSD-3-Clause — copyright HangZhou YuShu TECHNOLOGY CO.,LTD.
  ("Unitree Robotics"). Full text in [`LICENSE`](LICENSE).
- **Source package:** [google-deepmind/mujoco_menagerie `unitree_h1`](https://github.com/google-deepmind/mujoco_menagerie/tree/main/unitree_h1)
- Clean-room friendly for Apache-2.0 projects: retain copyright + LICENSE on
  redistribution; do not imply Unitree endorsement.

Do **not** mix in proprietary InvLambda / closed SDK assets.

## Install / fetch

Meshes are vendored here (~14 MB total STL). No git-LFS required for this tree.

To refresh from upstream:

```bash
git clone --depth 1 --filter=blob:none --sparse \
  https://github.com/google-deepmind/mujoco_menagerie.git /tmp/mujoco_menagerie
cd /tmp/mujoco_menagerie && git sparse-checkout set unitree_h1
cp -a unitree_h1/h1.xml unitree_h1/scene.xml unitree_h1/LICENSE \
      unitree_h1/CHANGELOG.md \
      /workspace/open-physical-sim/assets/robots/unitree_h1/
cp -a unitree_h1/assets/*.stl \
      /workspace/open-physical-sim/assets/robots/unitree_h1/assets/
```

Alternate official tree (also BSD-3-Clause):  
[`unitreerobotics/unitree_mujoco` → `unitree_robots/h1`](https://github.com/unitreerobotics/unitree_mujoco).

## Smoke load (headless)

```bash
cd /workspace/open-physical-sim && source .venv/bin/activate
python scripts/smoke_h1.py
# or
pytest -q tests/test_h1_load.py
```

Direct MuJoCo:

```python
import mujoco
m = mujoco.MjModel.from_xml_path("assets/robots/unitree_h1/scene.xml")
```

## Config

[`configs/h1_hello_twin.yaml`](../../../configs/h1_hello_twin.yaml) points at
`scene.xml` + `h1.xml`. Gym id for H1 is still **not** registered in v0; empty-v0
remains the only registered id.

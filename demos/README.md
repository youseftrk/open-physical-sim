# Demos

## `h1_walk_nofall_stand_v0.zarr`

Gym-faithful Unitree H1 upright-stand demos for BC on
`PhysicalSim/h1-hello-twin-v0` + `eval.task: walk_nofall`.

- Recorded via `gym.make(...).step` with `sim.frame_skip: 10`
- Horizon 40 gym steps
- **Pre-step** observations (vanilla closed-loop BC)
- Reset uses env **keyframe home** (same as `gym.reset`)

```bash
.venv/bin/python scripts/record_h1_stand_demo.py --steps 40 --episodes 5
```

## `h1_walk_nofall_walk_v0.zarr`

Cyclic H1 **walking gait** (sinusoidal hip/knee/ankle) under the same gym-faithful
rules as the stand demo. Survives 40 steps upright on `walk_nofall`.

```bash
.venv/bin/python scripts/record_h1_walk_demo.py --steps 40 --episodes 5
```

# Demos

## `h1_walk_nofall_stand_v0.zarr`

Gym-faithful Unitree H1 **upright stand** demos for BC on
`PhysicalSim/h1-hello-twin-v0` + `eval.task: walk_nofall`.

- Recorded via `gym.make(...).step` with **`sim.frame_skip: 10`**
- Horizon **40** gym steps (⇒ `info["success"]=True` on upright truncate)
- Generator: `scripts/record_h1_stand_demo.py`

```bash
.venv/bin/python scripts/record_h1_stand_demo.py --steps 40 --episodes 5
```

Open-loop replay of this zarr into the same gym env must stay upright.

# Demos

## `h1_walk_nofall_stand_v0.zarr`

Scripted Unitree H1 **upright stand** demos for BC against
`PhysicalSim/h1-hello-twin-v0` + `eval.task: walk_nofall`.

- Format: `open_physical_sim.trajectory.v1` (episodic Zarr)
- Generator: `scripts/record_h1_stand_demo.py`
- Success: survive horizon upright (no fall) → `info/success`

```bash
.venv/bin/python scripts/record_h1_stand_demo.py --steps 300 --episodes 5
```

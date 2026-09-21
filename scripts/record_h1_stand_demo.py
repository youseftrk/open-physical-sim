#!/usr/bin/env python3
"""Record H1 upright-stand demos for BC (walk_nofall).

PD-tracks Menagerie ``home`` keyframe with light free-root damping so the
torso stays up for ``--steps``. Writes TrajectoryWriter v1 Zarr for Policy Eng.
"""
from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import mujoco
import numpy as np
import yaml

from open_physical_sim.logging_zarr import TrajectoryWriter

_ROOT = Path(__file__).resolve().parents[1]


def _walk_nofall(data: mujoco.MjData, model: mujoco.MjModel, cfg: dict) -> dict:
    torso = str(cfg.get("torso_body", "torso_link"))
    pelvis = str(cfg.get("pelvis_body", "pelvis"))
    min_z = float(cfg.get("min_torso_z", 0.7))
    max_tilt = float(cfg.get("max_tilt_deg", 45.0))
    tid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, torso)
    if tid < 0:
        tid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, pelvis)
    z = float(data.xpos[tid, 2])
    up_z = float(data.xmat[tid].reshape(3, 3)[2, 2])
    tilt = float(np.degrees(np.arccos(np.clip(up_z, -1.0, 1.0))))
    fallen = (z < min_z) or (tilt > max_tilt)
    return {"torso_z": z, "tilt_deg": tilt, "fallen": fallen, "upright": not fallen}


def _obs(data: mujoco.MjData) -> dict[str, np.ndarray]:
    proprio = np.concatenate(
        [data.qpos.astype(np.float32), data.qvel.astype(np.float32)]
    )
    return {
        "proprio": proprio,
        "rgb": np.zeros((3, 64, 64), dtype=np.float32),
        "depth": np.zeros((3, 64, 64), dtype=np.float32),
        "ego": np.zeros((3, 64, 64), dtype=np.float32),
    }


def _pd(
    data: mujoco.MjData, model: mujoco.MjModel, q_des: np.ndarray, kp: float, kd: float
) -> np.ndarray:
    act = np.zeros(model.nu, dtype=np.float32)
    for i in range(model.nu):
        jid = int(model.actuator_trnid[i, 0])
        qadr = int(model.jnt_qposadr[jid])
        vadr = int(model.jnt_dofadr[jid])
        tau = kp * (q_des[qadr] - data.qpos[qadr]) - kd * data.qvel[vadr]
        lo, hi = model.actuator_ctrlrange[i]
        act[i] = np.clip(tau, lo, hi)
    return act


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=300)
    ap.add_argument("--episodes", type=int, default=5)
    ap.add_argument("--kp", type=float, default=400.0)
    ap.add_argument("--kd", type=float, default=10.0)
    ap.add_argument("--root-damp", type=float, default=0.99)
    ap.add_argument(
        "--out",
        type=Path,
        default=_ROOT / "demos" / "h1_walk_nofall_stand_v0.zarr",
    )
    args = ap.parse_args()

    full_cfg = yaml.safe_load((_ROOT / "configs" / "h1_hello_twin.yaml").read_text()) or {}
    eval_cfg = full_cfg.get("eval") or {}

    model = mujoco.MjModel.from_xml_path(str(_ROOT / "assets" / "robots" / "unitree_h1" / "scene.xml"))
    data = mujoco.MjData(model)
    q_home = np.array(model.key_qpos[0], dtype=np.float64)
    dt = float(model.opt.timestep)

    if args.out.exists():
        shutil.rmtree(args.out)
    writer = TrajectoryWriter(args.out)

    successes = 0
    for ep in range(args.episodes):
        mujoco.mj_resetDataKeyframe(model, data, 0)
        mujoco.mj_forward(model, data)
        writer.start_episode(f"stand_{ep:02d}")
        info: dict = {"success": False}
        for t in range(args.steps):
            action = _pd(data, model, q_home, args.kp, args.kd)
            data.ctrl[:] = action
            data.qvel[:6] *= args.root_damp
            mujoco.mj_step(model, data)
            wn = _walk_nofall(data, model, eval_cfg)
            terminated = bool(wn["fallen"])
            truncated = (t == args.steps - 1) and not terminated
            success = bool(truncated and not terminated)
            info = {**wn, "step": t + 1, "success": success}
            writer.append(
                obs=_obs(data),
                action=action,
                reward=(-1.0 if terminated else 0.01),
                terminated=terminated,
                truncated=truncated,
                info=info,
                timestamp=float(t) * dt,
            )
            if terminated:
                break
        successes += int(bool(info.get("success")))
        print(
            f"ep={ep} success={info.get('success')} fallen={info.get('fallen')} "
            f"z={float(info.get('torso_z', 0)):.3f} steps={info.get('step')}"
        )

    writer.close()
    print(f"wrote {args.out} successes={successes}/{args.episodes}")
    if successes < args.episodes:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

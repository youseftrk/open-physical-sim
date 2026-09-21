#!/usr/bin/env python3
"""Record H1 upright-stand demos via the Gym API (walk_nofall).

Matches ``PhysicalSim/h1-hello-twin-v0`` (``sim.frame_skip: 10``). Logs
**pre-step** observations for closed-loop BC. Relies on env reset keyframe-home.
"""
from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import gymnasium as gym
import numpy as np

import open_physical_sim  # noqa: F401
from open_physical_sim.logging_zarr import TrajectoryWriter

_ROOT = Path(__file__).resolve().parents[1]


def _pd(env: gym.Env, q_des: np.ndarray, kp: float, kd: float) -> np.ndarray:
    m = env.unwrapped.model
    d = env.unwrapped.data
    act = np.zeros(m.nu, dtype=np.float32)
    for i in range(m.nu):
        jid = int(m.actuator_trnid[i, 0])
        qadr = int(m.jnt_qposadr[jid])
        vadr = int(m.jnt_dofadr[jid])
        tau = kp * (q_des[qadr] - d.qpos[qadr]) - kd * d.qvel[vadr]
        lo, hi = m.actuator_ctrlrange[i]
        act[i] = np.clip(tau, lo, hi)
    return act


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=40)
    ap.add_argument("--episodes", type=int, default=5)
    ap.add_argument("--kp", type=float, default=400.0)
    ap.add_argument("--kd", type=float, default=10.0)
    ap.add_argument(
        "--out",
        type=Path,
        default=_ROOT / "demos" / "h1_walk_nofall_stand_v0.zarr",
    )
    args = ap.parse_args()

    env = gym.make("PhysicalSim/h1-hello-twin-v0")
    env.unwrapped.max_episode_steps = int(args.steps)
    m = env.unwrapped.model
    assert m.nkey >= 1, "expected home keyframe"
    q_home = np.array(m.key_qpos[0], dtype=np.float64)
    fs = int(getattr(env.unwrapped, "_frame_skip", 10))
    print(f"gym record: frame_skip={fs} horizon={args.steps} pre-step obs + keyframe reset")

    if args.out.exists():
        shutil.rmtree(args.out)
    writer = TrajectoryWriter(args.out)

    successes = 0
    for ep in range(args.episodes):
        obs, info = env.reset()  # env snaps to home keyframe
        writer.start_episode(f"stand_{ep:02d}")
        last = info
        for t in range(args.steps):
            action = _pd(env, q_home, args.kp, args.kd)
            # Log observation *before* the transition (closed-loop BC).
            pre_obs = {k: np.asarray(v).copy() for k, v in obs.items()}
            obs, reward, terminated, truncated, last = env.step(action)
            writer.append(
                obs=pre_obs,
                action=action,
                reward=float(reward),
                terminated=bool(terminated),
                truncated=bool(truncated),
                info=last,
                timestamp=float(t),
            )
            if terminated or truncated:
                break
        ok = bool(last.get("success", False))
        successes += int(ok)
        print(
            f"ep={ep} success={ok} fallen={last.get('fallen')} "
            f"z={last.get('torso_z')} steps={last.get('step')}"
        )

    writer.close()
    env.close()
    print(f"wrote {args.out} successes={successes}/{args.episodes}")
    if successes < args.episodes:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

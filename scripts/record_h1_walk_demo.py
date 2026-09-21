#!/usr/bin/env python3
"""Record H1 cyclic walking-gait demos via Gym (walk_nofall).

Same rules as stand demos: frame_skip=10, pre-step obs, keyframe-home reset.
PD-tracks a sinusoid gait around the Menagerie home pose.
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
    m, d = env.unwrapped.model, env.unwrapped.data
    act = np.zeros(m.nu, dtype=np.float32)
    for i in range(m.nu):
        jid = int(m.actuator_trnid[i, 0])
        qadr = int(m.jnt_qposadr[jid])
        vadr = int(m.jnt_dofadr[jid])
        tau = kp * (q_des[qadr] - d.qpos[qadr]) - kd * d.qvel[vadr]
        lo, hi = m.actuator_ctrlrange[i]
        act[i] = np.clip(tau, lo, hi)
    return act


def _gait_qdes(
    q_home: np.ndarray,
    t: int,
    *,
    period: float,
    amp_hip: float,
    amp_knee: float,
    amp_ankle: float,
    lean: float,
) -> np.ndarray:
    q_des = q_home.copy()
    phase = 2.0 * np.pi * (t / period)
    # Left (9,10,11) / right (14,15,16) hip pitch, knee, ankle — out of phase.
    for hip, knee, ankle, off in ((9, 10, 11, 0.0), (14, 15, 16, np.pi)):
        s = np.sin(phase + off)
        q_des[hip] = q_home[hip] + amp_hip * s + lean
        q_des[knee] = q_home[knee] + amp_knee * max(0.0, s)
        q_des[ankle] = q_home[ankle] - amp_ankle * s
    q_des[18] = q_home[18] + 0.12 * np.sin(phase + np.pi)
    q_des[22] = q_home[22] + 0.12 * np.sin(phase)
    return q_des


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=40)
    ap.add_argument("--episodes", type=int, default=5)
    ap.add_argument("--kp", type=float, default=500.0)
    ap.add_argument("--kd", type=float, default=12.0)
    ap.add_argument("--period", type=float, default=26.0)
    ap.add_argument("--amp-hip", type=float, default=0.12)
    ap.add_argument("--amp-knee", type=float, default=0.22)
    ap.add_argument("--amp-ankle", type=float, default=0.10)
    ap.add_argument("--lean", type=float, default=0.04)
    ap.add_argument(
        "--out",
        type=Path,
        default=_ROOT / "demos" / "h1_walk_nofall_walk_v0.zarr",
    )
    args = ap.parse_args()

    env = gym.make("PhysicalSim/h1-hello-twin-v0")
    env.unwrapped.max_episode_steps = int(args.steps)
    m = env.unwrapped.model
    assert m.nkey >= 1
    q_home = np.array(m.key_qpos[0], dtype=np.float64)
    fs = int(getattr(env.unwrapped, "_frame_skip", 10))
    print(
        f"walk record: frame_skip={fs} horizon={args.steps} "
        f"period={args.period} pre-step obs + keyframe reset"
    )

    if args.out.exists():
        shutil.rmtree(args.out)
    writer = TrajectoryWriter(args.out)

    successes = 0
    for ep in range(args.episodes):
        obs, info = env.reset()
        writer.start_episode(f"walk_{ep:02d}")
        last = info
        x0 = float(env.unwrapped.data.qpos[0])
        for t in range(args.steps):
            q_des = _gait_qdes(
                q_home,
                t,
                period=args.period,
                amp_hip=args.amp_hip,
                amp_knee=args.amp_knee,
                amp_ankle=args.amp_ankle,
                lean=args.lean,
            )
            action = _pd(env, q_des, args.kp, args.kd)
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
        dx = float(env.unwrapped.data.qpos[0]) - x0
        ok = bool(last.get("success", False))
        successes += int(ok)
        print(
            f"ep={ep} success={ok} fallen={last.get('fallen')} "
            f"z={last.get('torso_z')} dx={dx:.3f} steps={last.get('step')}"
        )

    writer.close()
    env.close()
    print(f"wrote {args.out} successes={successes}/{args.episodes}")
    if successes < args.episodes:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

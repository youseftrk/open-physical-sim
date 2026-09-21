#!/usr/bin/env python3
"""Headless smoke: make PhysicalSim/empty-v0, reset, step N times."""

from __future__ import annotations

import argparse
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("-n", "--steps", type=int, default=20)
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to YAML config (default: package empty_placeholder.yaml)",
    )
    args = parser.parse_args()

    import open_physical_sim  # noqa: F401 — registers envs
    import gymnasium as gym

    pkg_root = Path(__file__).resolve().parents[1]
    config = args.config or str(pkg_root / "configs" / "empty_placeholder.yaml")

    env = gym.make(
        "PhysicalSim/empty-v0",
        config_path=config,
    )
    obs, info = env.reset(seed=0)
    print("reset ok", {k: getattr(v, "shape", None) for k, v in obs.items()})
    print("info", info)
    print("action_space", env.action_space)

    for i in range(args.steps):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        if terminated or truncated:
            obs, info = env.reset()
            print(f"episode end at step {i}, reset")
    print(f"stepped {args.steps} times; last proprio[:4]={obs['proprio'][:4]}")
    env.close()
    print("hello_twin: OK")


if __name__ == "__main__":
    main()

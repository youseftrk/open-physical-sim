#!/usr/bin/env python3
"""Headless smoke: load Unitree H1 MJCF via MuJoCo (and optional gym config path)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PKG_ROOT = Path(__file__).resolve().parents[1]
H1_DIR = PKG_ROOT / "assets" / "robots" / "unitree_h1"
H1_XML = H1_DIR / "h1.xml"
H1_SCENE = H1_DIR / "scene.xml"
H1_CONFIG = PKG_ROOT / "configs" / "h1_hello_twin.yaml"


def _require_assets() -> None:
    missing = [p for p in (H1_XML, H1_SCENE) if not p.is_file()]
    assets = H1_DIR / "assets"
    if not assets.is_dir() or not any(assets.glob("*.stl")):
        missing.append(assets)
    if missing:
        print(
            "H1 assets missing:",
            ", ".join(str(p) for p in missing),
            file=sys.stderr,
        )
        print("See assets/robots/unitree_h1/README.md for fetch steps.", file=sys.stderr)
        sys.exit(2)


def load_models() -> None:
    import mujoco

    for path in (H1_XML, H1_SCENE):
        model = mujoco.MjModel.from_xml_path(str(path))
        data = mujoco.MjData(model)
        mujoco.mj_forward(model, data)
        print(
            f"loaded {path.relative_to(PKG_ROOT)}: "
            f"nq={model.nq} nv={model.nv} nu={model.nu} nbody={model.nbody}"
        )


def load_via_env_config() -> None:
    """Exercise config path resolution + resolve_mjcf same-package shortcut."""
    import mujoco
    import open_physical_sim  # noqa: F401
    from open_physical_sim.envs.h1_hello_twin import H1HelloTwinEnv

    env = H1HelloTwinEnv(config_path=str(H1_CONFIG))
    obs, info = env.reset(seed=0)
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info2 = env.step(action)
    print(
        "H1HelloTwinEnv: "
        f"nu={env.action_space.shape[0]} proprio={obs['proprio'].shape} "
        f"mjcf={info.get('mjcf')}"
    )
    env.close()
    # Also confirm raw from_xml still works after env close
    assert mujoco.MjModel.from_xml_path(str(H1_SCENE)) is not None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--env",
        action="store_true",
        help="Also construct H1HelloTwinEnv / configs/h1_hello_twin.yaml",
    )
    args = parser.parse_args()
    _require_assets()
    load_models()
    if args.env:
        load_via_env_config()
    print("smoke_h1: OK")


if __name__ == "__main__":
    main()

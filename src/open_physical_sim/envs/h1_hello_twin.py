"""H1HelloTwinEnv — Unitree H1 menagerie scene (hello twin)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from open_physical_sim.envs.empty import EmptyRoomEnv
from open_physical_sim.scene.loader import _PKG_ROOT

_DEFAULT_H1_CONFIG = _PKG_ROOT / "configs" / "h1_hello_twin.yaml"


class H1HelloTwinEnv(EmptyRoomEnv):
    """Gymnasium env defaulting to ``configs/h1_hello_twin.yaml``.

    Loads the vendored MuJoCo Menagerie Unitree H1 ``scene.xml`` (same-package
    shortcut preserves meshdir). Reuses EmptyRoomEnv reset/step/obs API.
    """

    def __init__(
        self,
        config_path: str | Path | None = None,
        **kwargs: Any,
    ) -> None:
        if config_path is None:
            config_path = _DEFAULT_H1_CONFIG
        super().__init__(config_path=config_path, **kwargs)


def make_h1_hello_twin_env(
    config_path: str | Path | None = None,
    **kwargs: Any,
) -> H1HelloTwinEnv:
    """Factory for ``PhysicalSim/h1-hello-twin-v0``."""
    return H1HelloTwinEnv(config_path=config_path, **kwargs)


__all__ = ["H1HelloTwinEnv", "make_h1_hello_twin_env"]

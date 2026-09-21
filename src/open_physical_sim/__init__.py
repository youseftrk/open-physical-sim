"""open_physical_sim — MuJoCo gymnasium environments for Open Real2Sim."""

from __future__ import annotations

__version__ = "0.1.0"

_REGISTERED = False

_ENVS = (
    (
        "PhysicalSim/empty-v0",
        "open_physical_sim.envs.empty:EmptyRoomEnv",
    ),
    (
        "PhysicalSim/h1-hello-twin-v0",
        "open_physical_sim.envs.h1_hello_twin:H1HelloTwinEnv",
    ),
)


def register_envs() -> None:
    """Register Gymnasium ids (idempotent)."""
    global _REGISTERED
    if _REGISTERED:
        return
    import gymnasium as gym
    from gymnasium.envs.registration import registry

    for env_id, entry_point in _ENVS:
        if env_id not in registry:
            gym.register(
                id=env_id,
                entry_point=entry_point,
                max_episode_steps=1000,
            )
    _REGISTERED = True


# Import-side registration so `import open_physical_sim` is enough.
register_envs()

__all__ = ["__version__", "register_envs"]

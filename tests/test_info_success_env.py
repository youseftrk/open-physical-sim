import gymnasium as gym
import open_physical_sim


def test_empty_step_emits_success():
    env = gym.make("PhysicalSim/empty-v0")
    _, info = env.reset()
    assert "success" in info and info["success"] is False
    _, _, _, _, info = env.step(env.action_space.sample())
    assert "success" in info and info["success"] is False
    env.close()


def test_h1_step_emits_success():
    env = gym.make("PhysicalSim/h1-hello-twin-v0")
    _, info = env.reset()
    assert info.get("success") is False
    _, _, _, _, info = env.step(env.action_space.sample())
    assert "success" in info
    env.close()

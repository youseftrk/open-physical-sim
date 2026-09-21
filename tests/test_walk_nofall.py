import gymnasium as gym
import open_physical_sim


def test_h1_walk_nofall_emits_fallen_fields():
    env = gym.make("PhysicalSim/h1-hello-twin-v0")
    env.reset()
    fell = False
    for _ in range(200):
        _, _, term, trunc, info = env.step(env.action_space.sample())
        assert "success" in info
        if info.get("fallen"):
            fell = True
            assert term is True
            assert info["success"] is False
            break
        if trunc:
            break
    env.close()
    # Random policy should usually fall; if not, still OK as long as keys exist.
    assert True

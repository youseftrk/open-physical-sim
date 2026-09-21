import numpy as np
import gymnasium as gym
import open_physical_sim
from open_physical_sim.action_scale import ActionScaler, normalize_action, denormalize_action


def test_roundtrip_h1():
    env = gym.make("PhysicalSim/h1-hello-twin-v0")
    scaler = ActionScaler(env)
    a = env.action_space.sample()
    n = scaler.normalize(a)
    assert np.all(n >= -1.01) and np.all(n <= 1.01)
    back = scaler.denormalize(n)
    np.testing.assert_allclose(back, a, rtol=1e-5, atol=1e-5)
    env.close()

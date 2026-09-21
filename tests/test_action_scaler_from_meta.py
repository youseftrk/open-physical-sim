import gymnasium as gym
import numpy as np
import open_physical_sim
from open_physical_sim.action_scale import ActionScaler


def test_from_meta_roundtrip():
    env = gym.make("PhysicalSim/h1-hello-twin-v0")
    s = ActionScaler(env)
    s2 = ActionScaler.from_meta(s.meta())
    a = env.action_space.sample()
    np.testing.assert_allclose(s.normalize(a), s2.normalize(a), atol=1e-6)
    env.close()

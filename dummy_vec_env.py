from typing import Callable

import gymnasium as gym
import numpy as np
from stable_baselines3.common.vec_env import DummyVecEnv


class DummyVecEnvIntRewards(DummyVecEnv):
    """This is a copy of the sb3 vec env with better float precision for large rewards"""
    def __init__(self, env_fns: list[Callable[[], gym.Env]]):
        super().__init__(env_fns)
        self.buf_rews = np.zeros((self.num_envs,), dtype=np.float64)

import numpy as np
from gymnasium.spaces import Box
from minigrid.core.constants import OBJECT_TO_IDX
from minigrid.wrappers import ImgObsWrapper





class OneHotObsWrapper(ImgObsWrapper):
    def __init__(self, env):
        super().__init__(env)
        # CNN expects CxHxW
        self.observation_space: Box = env.observation_space.spaces["image"]
        self.observation_space = Box(0,
                                     1,
                                     (len(OBJECT_TO_IDX), self.observation_space.shape[0], self.observation_space.shape[1]),
                                     np.uint8)

        self.one_hot_encoding = np.eye(len(OBJECT_TO_IDX), dtype=np.uint8)


    def observation(self, obs):
        color_obs = obs["image"][..., 0]
        color_obs = self.one_hot_encoding[color_obs]
        color_obs = np.transpose(color_obs, (2, 0, 1))
        return color_obs

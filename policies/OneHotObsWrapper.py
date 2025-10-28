import numpy as np
from gymnasium import spaces
from gymnasium.core import ObservationWrapper
from gymnasium.spaces import Box
from minigrid.core.constants import OBJECT_TO_IDX

from minigrid_env.environment import LeaderActions


class OneHotObsWrapper(ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)

        self.only_leader = False
        self.only_follower = False

        # CNN expects CxHxW
        if isinstance(env.observation_space, spaces.Dict):
            self._observation_space = spaces.Dict({
                "image": Box(
                    0, 1,
                    (len(OBJECT_TO_IDX),
                     env.observation_space.spaces["image"].shape[0],
                     env.observation_space.spaces["image"].shape[1]),
                    np.uint8,
                ),
                "follower_image": Box(
                    0, 1,
                    (len(OBJECT_TO_IDX) + len(LeaderActions),
                     env.observation_space.spaces["follower_image"].shape[0],
                     env.observation_space.spaces["follower_image"].shape[1]),
                    np.uint8,
                )
            })
        elif isinstance(env.observation_space, spaces.Box):
            if self.observation_space.shape[0] == 3:
                self.only_leader = True
                self._observation_space = spaces.Box(
                    0, 1,
                    (
                        env.observation_space.shape[0],
                        env.observation_space.shape[1],
                        len(OBJECT_TO_IDX),
                    ),
                )
            else:
                self.only_follower = True
                self._observation_space = spaces.Box(
                    0, 1,
                    (
                        env.observation_space.shape[0],
                        env.observation_space.shape[1],
                        len(OBJECT_TO_IDX) + len(LeaderActions),
                    ),
                )
        else:
            raise ValueError(f"Unsupported observation space {type(env.observation_space)}")

        assert isinstance(env.action_space, spaces.MultiDiscrete)
        self._leader_action_space = env.action_space[0]
        # TODO: improve one hot encoding to only the tiles being used
        self._one_hot_encoding = np.eye(len(OBJECT_TO_IDX), dtype=np.uint8)

    def observation(self, obs):
        if self.only_leader or self.only_follower:
            # TODO: This probably won't get used anywhere, but should be tested
            return self.observation_space.sample()

        leader_obs = obs["image"][..., 0]
        leader_obs = self._one_hot_encoding[leader_obs]

        follower_obs_without_action_encoded = self._one_hot_encoding[obs["follower_image"][..., 0]]
        follower_obs_only_action = obs["follower_image"][..., -self._leader_action_space.n:]
        follower_obs = np.concatenate([
            follower_obs_without_action_encoded,
            follower_obs_only_action,
        ], axis=-1)

        return {
            "image": leader_obs,
            "follower_image": follower_obs,
        }

from __future__ import annotations

import math
from enum import IntEnum
from typing import SupportsFloat, Any

import numpy as np
from gymnasium import spaces
from gymnasium.core import ActType, ObsType
from minigrid.core.constants import OBJECT_TO_IDX
from minigrid.core.grid import Grid
from minigrid.core.mission import MissionSpace
from minigrid.core.world_object import Goal, WorldObj, Lava
from minigrid.manual_control import ManualControl
from minigrid.minigrid_env import MiniGridEnv


class Actions(IntEnum):
    # Turn left, turn right, move forward
    left = 0
    right = 1
    forward = 2
    pickup = 3  # represents the null action if the control says it is dangerous


class LeaderActions(IntEnum):
    left = 0
    right = 1
    forward = 2


class FollowerActions(IntEnum):
    obey = 0
    disobey = 1


class RewardEncoding(IntEnum):
    leader_positive = 1
    leader_neutral = 1 << 1
    leader_negative = 1 << 2
    follower_positive = 1 << 3
    follower_neutral = 1 << 4
    follower_negative = 1 << 5


class Agent(WorldObj):
    def __init__(self, color="grey"):
        super().__init__("agent", color)

    def render(self, img):
        pass


class LavaEnv(MiniGridEnv):
    reward_encoding_base = 1000

    def __init__(
            self,
            grid_size=7,
            agent_start_position=(1, 1),
            agent_start_direction=0,
            max_steps: int | None = None,
            lava_tile_percentage: float = 0.1,
            only_leader_env: bool = False,
            only_follower_env: bool = False,
            one_hot_encode_tiles: bool = True,
            channel_first_obs: bool = True,
            return_summed_reward: bool = False,
            **kwargs,
    ):
        self.agent_start_position = agent_start_position
        self.agent_start_direction = agent_start_direction
        self.only_leader_env = only_leader_env
        self.only_follower_env = only_follower_env
        self.channel_first_obs = channel_first_obs
        self.is_eval_env = return_summed_reward

        self.one_hot_encode_tiles = one_hot_encode_tiles
        self._one_hot_encoding = np.eye(len(OBJECT_TO_IDX), dtype=np.float32)

        mission_space = MissionSpace(mission_func=self._generate_mission)

        assert lava_tile_percentage < 1.
        self.lava_tile_percentage = lava_tile_percentage

        super().__init__(
            mission_space=mission_space,
            grid_size=grid_size + 2,  # borders are added around the grid
            agent_view_size=grid_size,
            max_steps=max_steps or 256,
            # highlight=False,
            **kwargs,
        )

        # max steps * 3 possible rewards (0, 1, 2); add one for safety
        LavaEnv.reward_encoding_base = 10 ** int(math.log10(self.max_steps * 2) + 1)

        assert isinstance(self.observation_space, spaces.Dict)
        self.observation_space = spaces.Dict({
            "image": self.observation_space.spaces["image"],
        })

        assert isinstance(self.observation_space, spaces.Dict)
        follower_observation_space = spaces.Box(
            0, 255,
            (grid_size, grid_size, len(LeaderActions) + self.observation_space.spaces["image"].shape[-1]),
            dtype=np.float32)
        self.observation_space.spaces["follower_image"] = follower_observation_space

        if self.one_hot_encode_tiles:
            self.observation_space["image"] = spaces.Box(
                0, 1,
                (
                    self.observation_space.spaces["image"].shape[0],
                    self.observation_space.spaces["image"].shape[1],
                    len(OBJECT_TO_IDX),
                )
            )

            self.observation_space["follower_image"] = spaces.Box(
                0, 1,
                (
                    self.observation_space.spaces["image"].shape[0],
                    self.observation_space.spaces["image"].shape[1],
                    len(LeaderActions) + len(OBJECT_TO_IDX),
                )
            )

        if self.channel_first_obs:
            self.observation_space["image"] = spaces.Box(
                0, 1 if self.one_hot_encode_tiles else 255,
                (
                    self.observation_space.spaces["image"].shape[-1],
                    grid_size,
                    grid_size,
                )
            )

            self.observation_space["follower_image"] = spaces.Box(
                0, 1 if self.one_hot_encode_tiles else 255,
                (
                    self.observation_space.spaces["follower_image"].shape[-1],
                    grid_size,
                    grid_size,
                )
            )



        self.leader_actions = LeaderActions
        self.leader_action_space = spaces.Discrete(len(self.leader_actions))
        self.follower_actions = FollowerActions
        self.follower_action_space = spaces.Discrete(len(self.follower_actions))
        self.action_space = spaces.MultiDiscrete([len(self.leader_actions), len(self.follower_actions)])

        if self.only_leader_env:
            self.observation_space = self.observation_space.spaces["image"]
            self.action_space = self.leader_action_space
        elif self.only_follower_env:
            self.observation_space = self.observation_space.spaces["follower_image"]
            self.action_space = self.follower_action_space

        self.lava_tile = Lava

    @staticmethod
    def _generate_mission():
        return "Reach the goal tile without going through lava tiles"

    def _gen_grid(self, width, height):
        self.grid = Grid(width, height)
        self.grid.wall_rect(0, 0, width, height)

        if self.agent_start_position is not None:
            self.agent_pos = self.agent_start_position
            self.agent_dir = self.agent_start_direction
        else:
            self.place_agent()

        self.place_obj(Goal(), top=(width - 2, height - 2), size=(1, 1))

        lava_tiles = int((self.grid.width - 2) * (self.grid.height - 2) * self.lava_tile_percentage)
        for _ in range(lava_tiles):
            self.place_obj(self.lava_tile(), top=(1, 1), size=(self.grid.width - 2, self.grid.height - 2))
        # todo: check if there is a possible path from start to finish

    def step(
            self, action: ActType
    ) -> tuple[ObsType, SupportsFloat, bool, bool, dict[str, Any]]:
        combined_action = self.combine_actions(action)
        next_obs, leader_reward, terminated, truncated, info = super().step(combined_action)

        # if the agent stepped into lava, the reward for the follower is -1
        follower_reward = -1. if isinstance(self.grid.get(*self.agent_pos), Lava) else 1.
        # TODO: test if correct rewards are given for each actor

        if self.is_eval_env:
            reward = float(leader_reward) if follower_reward >= .0 else -1.
        else:
            # reward needs to be encoded to be compatible with gymnasium
            reward = self.encode_reward(leader_reward, follower_reward)
            assert self.decode_reward(reward) == (leader_reward, follower_reward)

        return next_obs, reward, terminated, truncated, info

    @staticmethod
    def combine_actions(
            action: np.ndarray,
    ):
        combined_action = np.where(action[1] == FollowerActions.obey, action[0], Actions.pickup)
        return combined_action.reshape((1,))

    def _reward(self) -> float:
        # TODO: give a discounted reward and change the encoding accordingly
        return 1.0

    @classmethod
    def encode_reward(cls, leader_reward: SupportsFloat, follower_reward: SupportsFloat) -> float:
        encoded_reward = 0

        leader_reward = float(leader_reward)
        if leader_reward < 0:
            encoded_reward += cls.reward_encoding_base ** 3
        elif leader_reward == 0:
            encoded_reward += cls.reward_encoding_base ** 4
        elif leader_reward > 0:
            encoded_reward += cls.reward_encoding_base ** 5
        else:
            raise ValueError(f"Invalid leader reward: {leader_reward}")

        follower_reward = float(follower_reward)
        if follower_reward < 0:
            encoded_reward += cls.reward_encoding_base ** 0
        elif follower_reward == 0:
            encoded_reward += cls.reward_encoding_base ** 1
        elif follower_reward > 0:
            encoded_reward += cls.reward_encoding_base ** 2
        else:
            raise ValueError(f"Invalid follower reward: {follower_reward}")

        return encoded_reward

    @classmethod
    def decode_reward(cls, encoded_reward: SupportsFloat) -> tuple[SupportsFloat, SupportsFloat]:
        encoded_reward = float(encoded_reward)

        leader_positive_rewards = encoded_reward // cls.reward_encoding_base ** 5
        encoded_reward -= leader_positive_rewards * cls.reward_encoding_base ** 5
        leader_neutral_rewards = encoded_reward // cls.reward_encoding_base ** 4
        encoded_reward -= leader_neutral_rewards * cls.reward_encoding_base ** 4
        leader_negative_rewards = encoded_reward // cls.reward_encoding_base ** 3
        encoded_reward -= leader_negative_rewards * cls.reward_encoding_base ** 3
        leader_reward = leader_positive_rewards - leader_negative_rewards

        follower_positive_rewards = encoded_reward // cls.reward_encoding_base ** 2
        encoded_reward -= follower_positive_rewards * cls.reward_encoding_base ** 2
        follower_neutral_rewards = encoded_reward // cls.reward_encoding_base ** 1
        encoded_reward -= follower_neutral_rewards * cls.reward_encoding_base ** 1
        follower_negative_rewards = encoded_reward
        follower_reward = follower_positive_rewards - follower_negative_rewards

        return leader_reward, follower_reward

    def gen_obs(self):
        obs = super().gen_obs()
        if self.one_hot_encode_tiles:
            obs["image"] = self._one_hot_encoding[obs["image"][..., 0]]

        if obs["image"].dtype != np.float32:
            obs["image"] = obs["image"].astype(np.float32)

        # give obs without leader action here, because there the leader hasn't decided its action yet
        follower_obs = np.concatenate([
            obs["image"],
            np.zeros((obs["image"].shape[0], obs["image"].shape[1], self.leader_action_space.n), dtype=obs["image"].dtype),
        ], axis=-1)
        obs["follower_image"] = follower_obs

        # hide lava tiles from the leader
        lava_tile_encoding = self.lava_tile().encode()
        obs["image"][..., :] = np.where(
            obs["image"][..., :] == (lava_tile_encoding if not self.one_hot_encode_tiles else OBJECT_TO_IDX["lava"]),
            np.array([OBJECT_TO_IDX["empty"], 0, 0]) if not self.one_hot_encode_tiles else OBJECT_TO_IDX["empty"],
            obs["image"][..., :]
        )

        if self.channel_first_obs:
            obs["image"] = np.moveaxis(obs["image"], -1, 0)
            obs["follower_image"] = np.moveaxis(obs["follower_image"], -1, 0)

        if self.only_leader_env:
            return obs["image"]
        elif self.only_follower_env:
            return obs["follower_image"]
        return {
            "image": obs["image"],
            "follower_image": obs["follower_image"],
        }

    @staticmethod
    def prepare_follower_obs(obs: np.ndarray, leader_actions: np.ndarray):
        # Overwrites the action in the observation
        if obs.ndim == 3:
            obs[-len(LeaderActions):, ...] = 0
            obs[:, leader_actions, ...] = 1
        elif obs.ndim == 4:
            obs[:, -len(LeaderActions):, ...] = 0
            obs[:, leader_actions, ...] = 1
        else:
            raise ValueError(f"obs has invalid number of dimensions: {obs.shape}. Should be 3 or 4")
        return obs


def main():
    env = LavaEnv(render_mode="human")

    manual_control = ManualControl(env, seed=42)
    manual_control.start()


if __name__ == '__main__':
    main()

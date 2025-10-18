from enum import IntEnum

import numpy as np
import torch as th
from gymnasium import spaces
from minigrid.core.grid import Grid
from minigrid.core.mission import MissionSpace
from minigrid.core.world_object import Goal, WorldObj, Lava
from minigrid.manual_control import ManualControl
from minigrid.minigrid_env import MiniGridEnv
from stable_baselines3.common.type_aliases import PyTorchObs


class Actions(IntEnum):
    # Turn left, turn right, move forward
    left = 0
    right = 1
    forward = 2
    pickup = 3  # represents the null action if the control says it is dangerous


class Agent(WorldObj):
    def __init__(self, color="grey"):
        super().__init__("agent", color)

    def render(self, img):
        pass


class LavaEnv(MiniGridEnv):
    def __init__(
            self,
            grid_size=7,
            agent_start_position=(1, 1),
            agent_start_direction=0,
            max_steps: int | None = None,
            lava_tile_percentage: float = 0.1,
            **kwargs,
    ):
        self.agent_start_position = agent_start_position
        self.agent_start_direction = agent_start_direction

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

        self.actions = Actions
        self.action_space = spaces.Discrete(len(self.actions))

    @staticmethod
    def _generate_mission():
        return "Lava environment"

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
            self.place_obj(Lava(), top=(1, 1), size=(self.grid.width - 2, self.grid.height - 2))
        # todo: check if there is a possible path from start to finish

    # def gen_obs_grid(self, agent_view_size=None):
    #     # agent has perfect information
    #     agent_view_size = agent_view_size or self.agent_view_size
    #     grid = self.grid.slice(1, 1, agent_view_size, agent_view_size)
    #     grid.set(self.agent_pos[0] - 1, self.agent_pos[1] - 1, Agent())
    #     return grid, None
    #
    # def get_full_render(self, highlight, tile_size):
    #     # Render the whole grid without highlights
    #     img = self.grid.render(
    #         tile_size,
    #         self.agent_pos,
    #         self.agent_dir,
    #     )
    #
    #     return img
    @classmethod
    def prepare_follower_obs(cls, obs: PyTorchObs, leader_action: np.ndarray):
        leader_action_obs = th.zeros((obs.shape[0],
                                      len(Actions),
                                      obs.shape[2],
                                      obs.shape[3]), )
        leader_action_obs[:, leader_action, ...] = 1
        follower_obs = th.cat([obs, leader_action_obs], dim=1)

        return follower_obs


def main():
    env = LavaEnv(render_mode="human")

    manual_control = ManualControl(env, seed=42)
    manual_control.start()


if __name__ == '__main__':
    main()

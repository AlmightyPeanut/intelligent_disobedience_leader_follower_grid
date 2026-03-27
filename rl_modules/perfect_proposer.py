from typing import Dict, Any

import numpy as np
import torch
from ray.rllib import SampleBatch
from ray.rllib.core.rl_module import RLModule
from ray.rllib.utils import override
from ray.rllib.utils.spaces.space_utils import batch as batch_func

from env import ProposerAction


class PerfectProposerRLM(RLModule):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._cached_action_probs: dict[int, np.ndarray] = {}
        self.last_action = None
        self.last_goal_position = None

    @override(RLModule)
    def _forward(self, batch: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        actions = []
        batch_size = len(batch[SampleBatch.OBS]["env"])
        for i in range(batch_size):
            obs = batch[SampleBatch.OBS]["env"][i]
            action = self._get_action(obs)
            actions.append(action)

        return {SampleBatch.ACTIONS: batch_func(actions)}

    def _get_action(self, obs: torch.Tensor) -> int:
        if self.last_action != ProposerAction.forward:
            self.last_action = ProposerAction.forward
            return ProposerAction.forward.value

        agent_pos, goal_pos = self._parse_obs(obs)
        position_hash = hash((agent_pos, goal_pos))

        if position_hash in self._cached_action_probs:
            action_probs = self._cached_action_probs[position_hash].copy()
        else:
            action_probs = self._sample_action_probs(agent_pos, goal_pos)
            self._cached_action_probs[position_hash] = action_probs

        if obs[agent_pos[0] - 1, agent_pos[1], 0] == 1.0:
            # hit a wall
            action_probs[ProposerAction.forward] = 0.0

        if obs[agent_pos[0], agent_pos[1] + 1, 0] == 1.0:
            # wall to the right
            action_probs[ProposerAction.turn_right] = 0.0

        if obs[agent_pos[0], agent_pos[1] - 1, 0] == 1.0:
            # wall to the left
            action_probs[ProposerAction.turn_left] = 0.0

        if self.last_goal_position == goal_pos and self.last_action == ProposerAction.forward:
            action_probs[ProposerAction.forward] = 0.0
        self.last_goal_position = goal_pos

        if np.all(action_probs == 0.0):
            action_probs = np.ones(len(ProposerAction))
        action_probs = action_probs / action_probs.sum()
        action = np.random.choice(len(ProposerAction), p=action_probs)
        self.last_action = action

        return action

    @staticmethod
    def _parse_obs(obs: torch.Tensor) -> tuple[tuple[int, int], tuple[int, int]]:
        agent_pos = tuple(torch.argwhere(obs[..., 1] == 1.0)[0].tolist())
        goal_pos = torch.argwhere(obs[..., 2] == 1.0)

        if len(goal_pos) > 0:
            goal_pos: tuple[int, int] = tuple(goal_pos[0].tolist())
        else:
            goal_pos = (0, 0)

        return agent_pos, goal_pos

    def _sample_action_probs(self, agent_pos: tuple[int, int], goal_pos: tuple[int, int]) -> np.ndarray:
        action_probs = np.zeros(len(ProposerAction))
        dy = agent_pos[0] - goal_pos[0]
        dx = agent_pos[1] - goal_pos[1]

        # Goal is in front of the agent
        if dx == 0 and dy == 1:
            action_probs[ProposerAction.forward] = 1.0
            return action_probs

        # Goal is one step to the left of the agent
        if dx == 1 and dy == 0:
            action_probs[ProposerAction.turn_left] = 1.0
            return action_probs

        # Goal is one step to the right of the agent
        if dx == -1 and dy == 0:
            action_probs[ProposerAction.turn_right] = 1.0
            return action_probs

        # All other cases weighted by proximity, and moving forward should be the most likely
        forward_weight = max(0.0, dy) * 1.1
        right_weight = max(0.0, -dx)
        left_weight = max(0.0, dx)

        weights = np.array([forward_weight, left_weight, right_weight])

        # Ensure all paths to the goal are played with some probability
        epsilon = 0.2 * weights.sum()
        weights += epsilon

        return weights

"""BFS-planning proposer with lava memory.

Lava-blind: learns lava cells from validator disobeys on `forward` and
re-plans each step. 
the agent tracks the absolute (pos, dir) by validator feedback:
    obeys update state, 
    disobeyed forwards leads to the agent marking the target cell as lava.
Episode boundary: 
detected when the Manhattan distance from the agent to the goal in the ego frame equals the max possible (2 * (size - 1))
which is true only at the spawn cell (1, 1) with goal at (size, size).
"""

from collections import deque
from typing import Any, Dict

import torch
from ray.rllib import SampleBatch
from ray.rllib.core.rl_module import RLModule
from ray.rllib.utils import override
from ray.rllib.utils.spaces.space_utils import batch as batch_func

from env import ProposerAction, ValidatorAction


# dir offsets in (row, col) format for UP, RIGHT, DOWN, LEFT = 0, 1, 2, 3 respectively
_DIR_OFFSET = {
    0: (-1, 0),  # UP
    1: (0, +1),  # RIGHT
    2: (+1, 0),  # DOWN
    3: (0, -1),  # LEFT
}


class PerfectProposerRLM(RLModule):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._pos: tuple[int, int] | None = None
        self._dir: int | None = None
        self._known_lava: set[tuple[int, int]] = set()
        self._last_action: int | None = None

    @override(RLModule)
    def _forward(self, batch: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        obs_env = batch[SampleBatch.OBS]["env"]
        validator_action = batch[SampleBatch.OBS]["validator_action"]
        actions = [
            self._get_action(obs_env[i], validator_action[i])
            for i in range(len(obs_env))
        ]
        return {SampleBatch.ACTIONS: batch_func(actions)}

    def _get_action(self, obs: torch.Tensor, validator_action: torch.Tensor) -> int:
        size = int(obs.shape[0])
        # goal is represented in obs as a one-hot in the third channel, find its position if visible
        goal_idx = torch.argwhere(obs[..., 2] == 1.0)
        goal_visible = len(goal_idx) > 0
        actual_goal_ego = goal_idx[0].tolist() if goal_visible else None

        # Update state from the last action's outcome
        disobeyed = (
            torch.argmax(validator_action).item() == ValidatorAction.disobey.value
        )
        if self._pos is not None and self._last_action is not None:
            # Only update internal state on obey/disobey feedback for forward actions 
            # turns do not change position and are always executed successfully
            if self._last_action == ProposerAction.forward.value:
                target = self._forward_cell()
                in_grid = 1 <= target[0] <= size and 1 <= target[1] <= size
                if disobeyed and in_grid:
                    self._known_lava.add(target)
                elif not disobeyed and in_grid:
                    self._pos = target
                # out of grid targets do not change the position, do not update pos or lava
            elif not disobeyed:
                if self._last_action == ProposerAction.turn_left.value:
                    self._dir = (self._dir - 1) % 4
                elif self._last_action == ProposerAction.turn_right.value:
                    self._dir = (self._dir + 1) % 4

        # Sync check to compare tracked state against observed goal position 
        # Goal is only visible when it's ahead of or at the same level of the agent
        # the goal being "not visible" is a legitimate observation, so we also check whether tracking predicts it to be visible
        # if there is a mismatch, we reset the tracked state
        need_reset = self._pos is None
        if not need_reset:
            expected = self._expected_goal_ego(size)
            expected_visible = (
                0 <= expected[0] < size and 0 <= expected[1] < 2 * size - 1
            )
            if goal_visible != expected_visible:
                need_reset = True
            elif goal_visible and list(expected) != actual_goal_ego:
                need_reset = True
        if need_reset:
            self._pos = (1, 1)
            self._dir = 1
            self._known_lava = set()
            self._last_action = None

        goal_world = (size, size)
        next_cell = self._bfs_next(self._pos, goal_world, size, self._known_lava)

        if next_cell is None:
            # No known path to the goal, just move forward by default to explore
            action = ProposerAction.forward.value
        else:
            action = self._action_toward(next_cell)

        self._last_action = action
        return action

    # Given the current tracked position and direction, return the cell in front of the agent in world coordinates
    def _forward_cell(self) -> tuple[int, int]:
        dr, dc = _DIR_OFFSET[self._dir]
        return (self._pos[0] + dr, self._pos[1] + dc)

    # Given the current tracked position and direction, where the goal should be in the ego-centric frame (agent at bottom middle, facing up)
    def _expected_goal_ego(self, size: int) -> tuple[int, int]:
        """Where the goal should appear in ego frame given the tracked (pos, dir)."""
        dr = size - self._pos[0]
        dc = size - self._pos[1]
        fw_dr, fw_dc = _DIR_OFFSET[self._dir]
        rt_dr, rt_dc = _DIR_OFFSET[(self._dir + 1) % 4]
        ego_fwd = dr * fw_dr + dc * fw_dc
        ego_rt = dr * rt_dr + dc * rt_dc
        return (size - 1 - ego_fwd, (size - 1) + ego_rt)
    
    # Given the next cell to move to in world coordinates, determine the action needed to move toward it
    def _action_toward(self, next_cell: tuple[int, int]) -> int:
        dr = next_cell[0] - self._pos[0]
        dc = next_cell[1] - self._pos[1]
        desired_dir = None
        for d, (ddr, ddc) in _DIR_OFFSET.items():
            if (ddr, ddc) == (dr, dc):
                desired_dir = d
                break
        if desired_dir is None:
            return ProposerAction.forward.value
        if desired_dir == self._dir:
            return ProposerAction.forward.value
        diff = (desired_dir - self._dir) % 4
        if diff == 1:
            return ProposerAction.turn_right.value
        if diff == 3:
            return ProposerAction.turn_left.value
        # 180 degree turn, can choose either way, choose right turn by default
        return ProposerAction.turn_right.value

    @staticmethod
    def _bfs_next(
        start: tuple[int, int],
        goal: tuple[int, int],
        size: int,
        blocked: set[tuple[int, int]],
    ) -> tuple[int, int] | None:
        if start == goal:
            return None
        parent: dict[tuple[int, int], tuple[int, int]] = {start: start}
        q = deque([start])
        while q:
            cur = q.popleft()
            if cur == goal:
                node = cur
                while parent[node] != start:
                    node = parent[node]
                return node
            for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                nxt = (cur[0] + dr, cur[1] + dc)
                if not (1 <= nxt[0] <= size and 1 <= nxt[1] <= size):
                    continue
                if nxt in blocked or nxt in parent:
                    continue
                parent[nxt] = cur
                q.append(nxt)
        return None

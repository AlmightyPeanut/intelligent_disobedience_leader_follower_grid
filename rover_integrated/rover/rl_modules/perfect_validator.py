from typing import Dict, Any

import torch
from ray.rllib import SampleBatch
from ray.rllib.core.rl_module import RLModule
from ray.rllib.utils import override
from ray.rllib.utils.spaces.space_utils import batch as batch_func

from env import ProposerAction, ValidatorAction


class PerfectValidatorRLM(RLModule):
    @override(RLModule)
    def _forward(self, batch: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        actions = []
        batch_size = len(batch[SampleBatch.OBS]["env"])
        proposer_actions = torch.argmax(batch[SampleBatch.OBS]["proposer_action"], dim=-1, keepdim=True)
        for i in range(batch_size):
            actions.append(self._get_action(batch[SampleBatch.OBS]["env"][i], proposer_actions[i]))

        return {
            SampleBatch.ACTIONS: batch_func(actions)
        }

    @staticmethod
    def _get_action(obs: torch.Tensor, proposer_action: torch.Tensor) -> int:
        if proposer_action.item() != ProposerAction.forward:
            return ValidatorAction.obey.value

        lava_tiles = obs[..., 3]
        agent_pos = torch.argwhere(obs[..., 1] == 1.0)[0].tolist()
        if lava_tiles[agent_pos[0] - 1, agent_pos[1]] == 1.:
            return ValidatorAction.disobey.value
        return ValidatorAction.obey.value
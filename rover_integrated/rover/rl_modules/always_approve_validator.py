from typing import Any

import tree
from ray.rllib import SampleBatch
from ray.rllib.core.rl_module import RLModule
from ray.rllib.utils import override
from ray.rllib.utils.spaces.space_utils import batch as batch_func

from env import ValidatorAction


class AlwaysApproveValidatorRLM(RLModule):
    @override(RLModule)
    def _forward(self, batch: dict[str, Any], **kwargs) -> dict[str, Any]:
        obs_batch_size = len(tree.flatten(batch[SampleBatch.OBS])[0])

        actions = batch_func([
            ValidatorAction.obey.value for _ in range(obs_batch_size)
        ])
        return {SampleBatch.ACTIONS: actions}

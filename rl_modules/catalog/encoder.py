import torch
from ray.rllib.core import Columns
from ray.rllib.core.models.base import Encoder, Model, ENCODER_OUT
from ray.rllib.core.models.torch.base import TorchModel, nn
from ray.rllib.utils import override

from rl_modules.catalog.catalog import DictEncoderConfig


class TorchDictEncoder(TorchModel, Encoder):
    """Torch encoder for single-level Dict observation spaces.

    Concatenates outputs of sub-encoders.
    """

    framework = "torch"

    def __init__(self, config: DictEncoderConfig) -> None:
        TorchModel.__init__(self, config)
        Encoder.__init__(self, config)

        self.sub_encoders = nn.ModuleDict()

        for key, sub_config in config.sub_encoder_configs.items():
            self.sub_encoders[key] = sub_config.build(framework="torch")

    @override(Model)
    def _forward(self, inputs: dict, **kwargs) -> dict:
        obs_dict = inputs[Columns.OBS]

        encoded_outputs = []

        for key, encoder in self.sub_encoders.items():
            sub_obs = obs_dict[key]

            out = encoder(
                {Columns.OBS: sub_obs},
                **kwargs,
            )[ENCODER_OUT]

            encoded_outputs.append(out)

        concatenated = torch.cat(encoded_outputs, dim=-1)

        return {ENCODER_OUT: concatenated}

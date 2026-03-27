import gymnasium as gym
from ray.rllib.core.models.catalog import Catalog
from ray.rllib.core.models.configs import ModelConfig
from ray.rllib.utils import override


class DictEncoderConfig(ModelConfig):
    """Encoder config for single-level gym.spaces.Dict observation spaces.

    For each key in the Dict space, a sub-encoder config is created.
    Outputs of all sub-encoders are concatenated.
    """

    def __init__(
            self,
            observation_space: gym.Space,
            cnn_config_dict: dict | None = None,
            mlp_config_dict: dict | None = None,
    ):
        assert isinstance(observation_space, gym.spaces.Dict), (
            "DictEncoderConfig requires a gym.spaces.Dict observation space."
        )

        self.observation_space = observation_space
        self.sub_encoder_configs = {}

        total_output_dim = 0

        cnn_config_dict = cnn_config_dict or {}
        mlp_config_dict = mlp_config_dict or {}

        for key, space in observation_space.spaces.items():
            if isinstance(space, gym.spaces.Box) and len(space.shape) == 1:
                encoder_config = Catalog._get_encoder_config(space, mlp_config_dict)
            elif isinstance(space, gym.spaces.Box) and len(space.shape) == 3:
                encoder_config = Catalog._get_encoder_config(space, cnn_config_dict)
            else:
                raise ValueError(f"Unsupported observation space: {space}")

            self.sub_encoder_configs[key] = encoder_config
            total_output_dim += encoder_config.output_dims[0]

        self._output_dims = (total_output_dim,)

    @property
    def input_dims(self) -> list[list[int] | tuple[int]]:
        return [e.input_dims for e in self.sub_encoder_configs.values()]

    @input_dims.setter
    def input_dims(self, dims: list[list[int] | tuple[int]]):
        for e, dim in zip(self.sub_encoder_configs.values(), dims):
            e.input_dims = dim

    @override(ModelConfig)
    def build(self, framework: str = "torch"):
        if framework == "torch":
            from rl_modules.catalog.encoder import TorchDictEncoder
            return TorchDictEncoder(self)
        else:
            raise ValueError(f"Unsupported framework: {framework}")

    @property
    def output_dims(self) -> tuple[int]:
        return self._output_dims

    @output_dims.setter
    def output_dims(self, value: tuple[int]) -> None:
        self._output_dims = value

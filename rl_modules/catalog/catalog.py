import dataclasses

import gymnasium as gym
from ray.rllib.algorithms.dqn.dqn_catalog import DQNCatalog
from ray.rllib.algorithms.ppo.ppo_catalog import PPOCatalog
from ray.rllib.core.models.catalog import Catalog
from ray.rllib.core.models.configs import ModelConfig
from ray.rllib.core.rl_module.default_model_config import DefaultModelConfig

from rl_modules.catalog.configs import DictEncoderConfig


class CatalogWithImageActionEncoder(Catalog):
    @classmethod
    def _get_encoder_config(
            cls,
            observation_space: gym.Space,
            model_config_dict: dict,
            action_space: gym.Space = None,
    ) -> ModelConfig:
        if not isinstance(observation_space, gym.spaces.Dict):
            return Catalog._get_encoder_config(observation_space, model_config_dict, action_space)

        if "dict_encoder_config" not in model_config_dict:
            raise ValueError("dict_encoder_config is required for Dict observation spaces.")

        if "cnn_config_dict" in model_config_dict["dict_encoder_config"]:
            cnn_config_dict = model_config_dict["dict_encoder_config"]["cnn_config_dict"]
        else:
            cnn_config_dict = {}

        if "mlp_config_dict" in model_config_dict["dict_encoder_config"]:
            mlp_config_dict = model_config_dict["dict_encoder_config"]["mlp_config_dict"]
        else:
            mlp_config_dict = {}

        if dataclasses.is_dataclass(cnn_config_dict):
            cnn_config_dict = dataclasses.asdict(cnn_config_dict)
        if dataclasses.is_dataclass(mlp_config_dict):
            mlp_config_dict = dataclasses.asdict(mlp_config_dict)
        default_config = dataclasses.asdict(DefaultModelConfig())

        cnn_config_dict = default_config | cnn_config_dict
        mlp_config_dict = default_config | mlp_config_dict

        return DictEncoderConfig(observation_space, cnn_config_dict, mlp_config_dict)


class DQNCatalogWithImageActionEncoder(CatalogWithImageActionEncoder, DQNCatalog):
    pass

class PPOCatalogWithImageActionEncoder(CatalogWithImageActionEncoder, PPOCatalog):
    pass

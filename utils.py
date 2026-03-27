from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path

import gymnasium as gym

from ray.rllib.algorithms.dqn.torch.default_dqn_torch_rl_module import DefaultDQNTorchRLModule
from ray.rllib.algorithms.ppo.torch.default_ppo_torch_rl_module import DefaultPPOTorchRLModule

from env import ProposerAction, ValidatorAction, EnvironmentAction
from rl_modules.catalog.catalog import DQNCatalogWithImageActionEncoder, PPOCatalogWithImageActionEncoder
from rl_modules.dqn_modules import LearnedValidatorDQN

TUNER_MODEL_PATH = '/Users/benedikt/PycharmProjects/intelligent_disobedience_leader_follower_grid/new/best_tuner_model/'
TRAIN_MODEL_PATH = '/Users/benedikt/PycharmProjects/intelligent_disobedience_leader_follower_grid/new/best_train_model/'

PROPOSER_ACTION_SPACE = gym.spaces.Discrete(len(ProposerAction))
VALIDATOR_ACTION_SPACE = gym.spaces.Discrete(len(ValidatorAction))
SINGLE_AGENT_ACTION_SPACE = gym.spaces.Discrete(len(EnvironmentAction))

PROPOSER_ALGORITHM_MODULES = {
    "dqn": DefaultDQNTorchRLModule,
    "ppo": DefaultPPOTorchRLModule
}

VALIDATOR_ALGORITHM_MODULES = {
    "dqn": LearnedValidatorDQN,
    "ppo": DefaultPPOTorchRLModule,
}

SINGLE_AGENT_ALGORITHM_MODULES = {
    "dqn": DefaultDQNTorchRLModule,
    "ppo": DefaultPPOTorchRLModule,
}

DEFAULT_SINGLE_AGENT_CONV_MODEL_CONFIG = {
    "conv_filters": [
        [16, 3, 1],
        [32, 3, 1],
    ],
    "conv_activation": "relu",
    "head_fcnet_hiddens": [64],
    "fcnet_activation": "relu",
}

DEFAULT_MULTI_AGENT_MODEL_CONFIG = {
    "dict_encoder_config": {
        "cnn_config_dict": DEFAULT_SINGLE_AGENT_CONV_MODEL_CONFIG.copy(),
        "mlp_config_dict": {
            "fcnet_hiddens": [8],
        }
    },
    "head_fcnet_hiddens": [64],
    "fcnet_activation": "relu",
}

CATALOG_CLASS = {
    "dqn": DQNCatalogWithImageActionEncoder,
    "ppo": PPOCatalogWithImageActionEncoder,
}


class ProposerPolicies(StrEnum):
    LEARNED = "learned_proposer"
    PERFECT = "perfect_proposer"
    RANDOM = "random_proposer"


class ValidatorPolicies(StrEnum):
    LEARNED = "learned_validator"
    PERFECT = "perfect_validator"
    ALWAYS_APPROVE = "always_approve_validator"


@dataclass(frozen=True)
class AgentConfig:
    proposer_policy: ProposerPolicies = None
    validator_policy: ValidatorPolicies = None
    algorithm_name: str = "ppo"
    proposer_sees_lava: bool = False


AGENT_CONFIGS = [
    AgentConfig(
        proposer_policy=ProposerPolicies.LEARNED,
        validator_policy=ValidatorPolicies.ALWAYS_APPROVE,
    ),
    AgentConfig(
        proposer_policy=ProposerPolicies.LEARNED,
        validator_policy=ValidatorPolicies.PERFECT,
    ),
    AgentConfig(
        proposer_policy=ProposerPolicies.PERFECT,
        validator_policy=ValidatorPolicies.LEARNED,
    ),
    AgentConfig(
        proposer_policy=ProposerPolicies.RANDOM,
        validator_policy=ValidatorPolicies.LEARNED,
    ),
    AgentConfig(
        proposer_policy=ProposerPolicies.LEARNED,
        validator_policy=ValidatorPolicies.LEARNED,
    ),
    AgentConfig(
        proposer_policy=ProposerPolicies.LEARNED,
        validator_policy=ValidatorPolicies.LEARNED,
        proposer_sees_lava=True,
    ),
]

LOG_DIR = Path("/Users/benedikt/PycharmProjects/intelligent_disobedience_leader_follower_grid/new/logs")
GRID_SIZE = 3
NUM_LAVA_TILES = 2
MAX_ENV_STEPS = 128
TRAINING_ITERATIONS = 1000

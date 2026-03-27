from functools import partial
from typing import Hashable

from ray.rllib.algorithms import AlgorithmConfig, PPOConfig
from ray.rllib.algorithms.dqn import DQNConfig
from ray.rllib.core.rl_module import MultiRLModuleSpec, RLModuleSpec
from ray.rllib.env.multi_agent_episode import MultiAgentEpisode
from ray.rllib.examples.rl_modules.classes.random_rlm import RandomRLModule

from rl_modules.always_approve_validator import AlwaysApproveValidatorRLM
from rl_modules.perfect_proposer import PerfectProposerRLM
from rl_modules.perfect_validator import PerfectValidatorRLM
from utils import (
    AgentConfig,
    ProposerPolicies,
    ValidatorPolicies,
    PROPOSER_ALGORITHM_MODULES,
    VALIDATOR_ALGORITHM_MODULES,
    SINGLE_AGENT_ALGORITHM_MODULES,
    DEFAULT_MULTI_AGENT_MODEL_CONFIG,
    DEFAULT_SINGLE_AGENT_CONV_MODEL_CONFIG, CATALOG_CLASS, )


def create_algorithm_config(algorithm_name: str) -> AlgorithmConfig:
    config = None
    if algorithm_name == "dqn":
        config = DQNConfig().training(
            replay_buffer_config={
                "enable_replay_buffer_api": True,
                "type": "MultiAgentPrioritizedEpisodeReplayBuffer",
                "alpha": 0.6,
                "beta": 0.4,
            }
        )

    if algorithm_name == "ppo":
        config = PPOConfig().training(
            entropy_coeff=0.2,
            train_batch_size=256,
        )

    if config is None:
        raise ValueError(f"Unknown algorithm: {algorithm_name}")

    return config.framework("torch")


def add_env_config(config: AlgorithmConfig) -> AlgorithmConfig:
    config.environment("env")
    return config


def add_single_agent_policies(config: AlgorithmConfig, agent_config: AgentConfig) -> AlgorithmConfig:
    module_class = SINGLE_AGENT_ALGORITHM_MODULES[agent_config.algorithm_name]

    return config.multi_agent(
        policies=["single_agent"],
        policy_mapping_fn=lambda agent_id, episode: "single_agent",
        policies_to_train=["single_agent"],
    ).rl_module(
        rl_module_spec=MultiRLModuleSpec(
            rl_module_specs={
                "single_agent": RLModuleSpec(
                    module_class=module_class,
                    model_config=DEFAULT_SINGLE_AGENT_CONV_MODEL_CONFIG,
                )
            }
        )
    )


def agent_config_policy_mapping(
        agent_id: Hashable,
        episode: MultiAgentEpisode,
        agent_config: AgentConfig,
) -> str:
    if agent_id == "proposer":
        return agent_config.proposer_policy

    if agent_id == "validator":
        return agent_config.validator_policy

    raise ValueError(f"Invalid agent: {agent_id}")


def get_multi_agent_rl_module_specs(policy_names: list[str], agent_config: AgentConfig) -> dict[str, RLModuleSpec]:
    rl_module_specs = {}
    if ProposerPolicies.LEARNED in policy_names:
        rl_module_specs[ProposerPolicies.LEARNED] = RLModuleSpec(
            module_class=PROPOSER_ALGORITHM_MODULES[agent_config.algorithm_name],
            model_config=DEFAULT_MULTI_AGENT_MODEL_CONFIG,
            catalog_class=CATALOG_CLASS[agent_config.algorithm_name],
        )

    if ProposerPolicies.PERFECT in policy_names:
        rl_module_specs[ProposerPolicies.PERFECT] = RLModuleSpec(
            module_class=PerfectProposerRLM,
            inference_only=True,
        )

    if ProposerPolicies.RANDOM in policy_names:
        rl_module_specs[ProposerPolicies.RANDOM] = RLModuleSpec(
            module_class=RandomRLModule,
            inference_only=True,
        )

    if ValidatorPolicies.LEARNED in policy_names:
        rl_module_specs[ValidatorPolicies.LEARNED] = RLModuleSpec(
            module_class=VALIDATOR_ALGORITHM_MODULES[agent_config.algorithm_name],
            model_config=DEFAULT_MULTI_AGENT_MODEL_CONFIG,
            catalog_class=CATALOG_CLASS[agent_config.algorithm_name],
        )

    if ValidatorPolicies.PERFECT in policy_names:
        rl_module_specs[ValidatorPolicies.PERFECT] = RLModuleSpec(
            module_class=PerfectValidatorRLM,
            inference_only=True,
        )

    if ValidatorPolicies.ALWAYS_APPROVE in policy_names:
        rl_module_specs[ValidatorPolicies.ALWAYS_APPROVE] = RLModuleSpec(
            module_class=AlwaysApproveValidatorRLM,
            inference_only=True,
        )

    return rl_module_specs


def add_multi_agent_policies(
        config: AlgorithmConfig,
        agent_config: AgentConfig,
) -> AlgorithmConfig:
    assert agent_config.proposer_policy is not None
    assert agent_config.validator_policy is not None

    policies = [agent_config.proposer_policy, agent_config.validator_policy]
    policy_mapping_fn = partial(agent_config_policy_mapping, agent_config=agent_config)
    policies_to_train = []
    if agent_config.proposer_policy == ProposerPolicies.LEARNED:
        policies_to_train.append(ProposerPolicies.LEARNED)
    if agent_config.validator_policy == ValidatorPolicies.LEARNED:
        policies_to_train.append(ValidatorPolicies.LEARNED)

    config = (
        config
        .multi_agent(
            policies=policies,
            policy_mapping_fn=policy_mapping_fn,
            policies_to_train=policies_to_train,
        )
        .rl_module(
            rl_module_spec=MultiRLModuleSpec(
                rl_module_specs=get_multi_agent_rl_module_specs(policies, agent_config),
            )
        )
    )

    return config


def create_rllib_config(agent_config: AgentConfig) -> AlgorithmConfig:
    config = create_algorithm_config(agent_config.algorithm_name)
    config = add_env_config(config)
    if agent_config.proposer_policy is None and agent_config.validator_policy is None:
        assert agent_config.algorithm_name in SINGLE_AGENT_ALGORITHM_MODULES.keys()
        config = add_single_agent_policies(config, agent_config)
    else:
        assert agent_config.algorithm_name in PROPOSER_ALGORITHM_MODULES.keys()
        assert agent_config.algorithm_name in VALIDATOR_ALGORITHM_MODULES.keys()
        config = add_multi_agent_policies(config, agent_config)

    config.validate()

    return config

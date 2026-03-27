from typing import Dict, Union, Optional, List

import gymnasium as gym
import tree
from ray.rllib import BaseEnv, Policy

from ray.rllib.algorithms.callbacks import DefaultCallbacks
from ray.rllib.core.rl_module import RLModule
from ray.rllib.env.env_runner import EnvRunner
from ray.rllib.evaluation.episode_v2 import EpisodeV2
from ray.rllib.utils.metrics.metrics_logger import MetricsLogger
from ray.rllib.utils.typing import EpisodeType, PolicyID
from ray.tune.experiment import Trial
from ray.tune.logger import TBXLoggerCallback


class CustomTBXLoggerCallback(TBXLoggerCallback):
    def log_trial_result(self, iteration: int, trial: "Trial", result: Dict):
        if "timers" in result:
            del result["timers"]
        if "env_runners" in result:
            for key in list(result["env_runners"].keys()):
                if "timer" in key:
                    del result["env_runners"][key]
            del result["env_runners"]["module_to_env_connector"]
            del result["env_runners"]["env_to_module_connector"]
            if "time_between_sampling" in result["env_runners"]:
                del result["env_runners"]["time_between_sampling"]
        if "replay_buffer" in result:
            del result["replay_buffer"]
        if "learners" in result:
            for agent_id in list(result["learners"].keys()):
                if "learner_connector" in result["learners"][agent_id]:
                    del result["learners"][agent_id]["learner_connector"]
        if "perf" in result:
            del result["perf"]
        super().log_trial_result(iteration, trial, result)


class ActionLoggerCallback(DefaultCallbacks):
    def on_episode_end(
        self,
        *,
        episode: Union[EpisodeType, EpisodeV2],
        prev_episode_chunks: Optional[List[EpisodeType]] = None,
        env_runner: Optional["EnvRunner"] = None,
        metrics_logger: Optional[MetricsLogger] = None,
        env: Optional[gym.Env] = None,
        env_index: int,
        rl_module: Optional[RLModule] = None,
        worker: Optional["EnvRunner"] = None,
        base_env: Optional[BaseEnv] = None,
        policies: Optional[Dict[PolicyID, Policy]] = None,
        **kwargs,
    ) -> None:
        for agent_id, actions in episode.get_actions().items():
            for action in actions:
                metrics_logger.log_value(f"action/{agent_id}", int(action), reduce="item_series")


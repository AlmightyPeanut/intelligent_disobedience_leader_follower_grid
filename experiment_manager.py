import importlib
import os
from collections import OrderedDict
from pathlib import Path
from pprint import pprint
from typing import Any

import optuna
import yaml
from rl_zoo3.callbacks import TrialEvalCallback
from rl_zoo3.exp_manager import ExperimentManager
from rl_zoo3.hyperparams_opt import HYPERPARAMS_SAMPLER
from rl_zoo3.utils import get_callback_list, ALGOS
from sb3_contrib.common.vec_env import AsyncEval
from stable_baselines3 import HerReplayBuffer
from stable_baselines3.common.base_class import BaseAlgorithm

from dummy_vec_env import DummyVecEnvIntRewards
from minigrid_env.environment import LavaEnv, LeaderAction, FollowerAction
from policies.LeaderFollowerAlgorithm import LeaderFollowerAlgorithm


class ExperimentManagerLF(ExperimentManager):
    default_vec_env_cls = DummyVecEnvIntRewards

    def objective(self, trial: optuna.Trial) -> float:
        kwargs = self._hyperparams.copy()

        n_envs = 1 if self.algo == "ars" else self.n_envs

        additional_args = {
            "using_her_replay_buffer": kwargs.get("replay_buffer_class") == HerReplayBuffer,
            "her_kwargs": kwargs.get("replay_buffer_kwargs", {}),
        }
        # Pass n_actions to initialize DDPG/TD3 noise sampler
        # Sample candidate hyperparameters
        sampled_hyperparams_leader = HYPERPARAMS_SAMPLER[self._hyperparams["leader_algorithm"]](trial, len(LeaderAction), n_envs, additional_args)
        kwargs["leader_algorithm_kwargs"].update(sampled_hyperparams_leader)

        sampled_hyperparams_follower = HYPERPARAMS_SAMPLER[self._hyperparams["leader_algorithm"]](trial, len(FollowerAction), n_envs, additional_args)
        kwargs["follower_algorithm_kwargs"].update(sampled_hyperparams_follower)

        env = self.create_envs(n_envs, no_log=True)

        # By default, do not activate verbose output to keep
        # stdout clean with only the trial's results
        trial_verbosity = 0
        # Activate verbose mode for the trial in debug mode
        # See PR #214
        if self.verbose >= 2:
            trial_verbosity = self.verbose

        model = LeaderFollowerAlgorithm(
            env=env,
            tensorboard_log=self.tensorboard_log,
            seed=self.seed,
            verbose=trial_verbosity,
            device=self.device,
            **kwargs,
        )

        eval_env = self.create_envs(n_envs=self.n_eval_envs, eval_env=True)

        optuna_eval_freq = int(self.n_timesteps / self.n_evaluations)
        # Account for parallel envs
        optuna_eval_freq = max(optuna_eval_freq // self.n_envs, 1)
        # Use non-deterministic eval for Atari
        path = None
        if self.optimization_log_path is not None:
            path = os.path.join(self.optimization_log_path, f"trial_{trial.number!s}")
        callbacks = get_callback_list({"callback": self.specified_callbacks})

        # TODO: maybe join into one callback? Might not be deterministic
        eval_callback = TrialEvalCallback(
            eval_env,
            trial,
            best_model_save_path=path,
            log_path=path,
            n_eval_episodes=self.n_eval_episodes,
            eval_freq=optuna_eval_freq,
            deterministic=self.deterministic_eval,
        )
        callbacks.append(eval_callback)

        individual_rewards_eval_callback = TrialEvalCallback(
            env,
            trial,
            best_model_save_path=path,
            log_path=path,
            n_eval_episodes=self.n_eval_episodes,
            eval_freq=optuna_eval_freq,
        )
        callbacks.append(individual_rewards_eval_callback)


        learn_kwargs = {}
        # Special case for ARS
        if self.n_envs > 1:
            if self._hyperparams["leader_algorithm"] == "ars":
                learn_kwargs["async_eval_leader"] = AsyncEval(
                    [lambda: self.create_envs(n_envs=1, no_log=True) for _ in range(self.n_envs)], model.leader_model.policy
                )
            if self._hyperparams["follower_algorithm"] == "ars":
                learn_kwargs["async_eval_follower"] = AsyncEval(
                    [lambda: self.create_envs(n_envs=1, no_log=True) for _ in range(self.n_envs)], model.follower_model.policy
                )

        try:
            model.learn(self.n_timesteps, callback=callbacks, **learn_kwargs)  # type: ignore[arg-type]
            # Free memory
            assert model.env is not None
            model.env.close()
            eval_env.close()
        except (AssertionError, ValueError) as e:
            # Sometimes, random hyperparams can generate NaN
            # Free memory
            assert model.env is not None
            model.env.close()
            eval_env.close()
            # Prune hyperparams that generate NaNs
            print(e)
            print("============")
            print("Sampled hyperparams leader:")
            pprint(sampled_hyperparams_leader)
            print("Sampled hyperparams follower:")
            pprint(sampled_hyperparams_follower)
            raise optuna.exceptions.TrialPruned() from e
        is_pruned = eval_callback.is_pruned
        reward = eval_callback.last_mean_reward

        individual_rewards = individual_rewards_eval_callback.last_mean_reward
        leader_reward, follower_reward = LavaEnv.decode_reward(individual_rewards)
        trial.set_user_attr("leader_reward", leader_reward)
        trial.set_user_attr("follower_reward", follower_reward)

        del model.env, eval_env
        del model

        if is_pruned:
            raise optuna.exceptions.TrialPruned()

        return reward

    def setup_experiment(self) -> tuple[BaseAlgorithm, dict[str, Any]] | None:
        """
        Read hyperparameters, pre-process them (create schedules, wrappers, callbacks, action noise objects)
        create the environment and possibly the model.

        :return: the initialized RL model
        """
        unprocessed_hyperparams, saved_hyperparams = self.read_hyperparameters()
        hyperparams, self.env_wrapper, self.callbacks, self.vec_env_wrapper = self._preprocess_hyperparams(
            unprocessed_hyperparams
        )

        self.create_log_folder()
        self.create_callbacks()

        # Create env to have access to action space for action noise
        n_envs = 1 if self.algo == "ars" or self.optimize_hyperparameters else self.n_envs
        env = self.create_envs(n_envs, no_log=False)

        hyperparams = self._preprocess_action_noise(hyperparams, saved_hyperparams, env)
        self._hyperparams = self._split_hyperparams_for_leader_follower(hyperparams)

        if self.continue_training:
            model = self._load_pretrained_agent(self._hyperparams, env)
        elif self.optimize_hyperparameters:
            env.close()
            return None
        else:
            # Train an agent from scratch
            model = ALGOS[self.algo](
                env=env,
                tensorboard_log=self.tensorboard_log,
                seed=self.seed,
                verbose=self.verbose,
                device=self.device,
                **self._hyperparams,
            )

        self._save_config(saved_hyperparams)
        return model, saved_hyperparams

    def _split_hyperparams_for_leader_follower(self, hyperparams: dict[str, Any]) -> dict[str, Any]:
        policy_params = {k: v for k, v in hyperparams.items() if k not in [
            "leader_policy",
            "leader_algorithm",
            "follower_policy",
            "follower_algorithm",
            "learning_rate",
            "train_leader",
            "train_follower",
        ]}

        hyperparams = {k: v for k, v in hyperparams.items() if k not in policy_params.keys()}
        hyperparams["leader_algorithm_kwargs"] = policy_params.copy()
        hyperparams["follower_algorithm_kwargs"] = policy_params.copy()

        return hyperparams


    def read_hyperparameters(self) -> tuple[dict[str, Any], dict[str, Any]]:
        print(f"Loading hyperparameters from: {self.config}")

        if self.config.endswith(".yml") or self.config.endswith(".yaml"):
            # Load hyperparameters from yaml file
            with open(self.config) as f:
                hyperparams_dict = yaml.safe_load(f)
        elif self.config.endswith(".py"):
            global_variables: dict = {}
            # Load hyperparameters from python file
            exec(Path(self.config).read_text(), global_variables)
            hyperparams_dict = global_variables["hyperparams"]
        else:
            # Load hyperparameters from python package
            hyperparams_dict = importlib.import_module(self.config).hyperparams
            # raise ValueError(f"Unsupported config file format: {self.config}")

        if self.env_name.gym_id in list(hyperparams_dict.keys()):
            hyperparams = hyperparams_dict[self.env_name.gym_id]
        elif self._is_atari:
            hyperparams = hyperparams_dict["atari"]
        else:
            raise ValueError(f"Hyperparameters not found for {self.algo}-{self.env_name.gym_id} in {self.config}")

        if self.storage and self.study_name and not self.optimize_hyperparameters:
            print("Loading from Optuna study...")
            study_hyperparams = self.load_trial(self.storage, self.study_name, self.trial_id)
            hyperparams.update(study_hyperparams)

        if self.custom_hyperparams is not None:
            # Overwrite hyperparams if needed
            hyperparams.update(self.custom_hyperparams)
        # Sort hyperparams that will be saved
        saved_hyperparams = OrderedDict([(key, hyperparams[key]) for key in sorted(hyperparams.keys())])

        # Always print used hyperparameters
        print("Default hyperparameters for environment (ones being tuned will be overridden):")
        pprint(saved_hyperparams)

        return hyperparams, saved_hyperparams

import importlib
import os
import warnings
from pathlib import Path
from pprint import pprint
from typing import Any, Optional, OrderedDict

import optuna
import yaml
from rl_zoo3.callbacks import TrialEvalCallback
from rl_zoo3.exp_manager import ExperimentManager
from rl_zoo3.hyperparams_opt import HYPERPARAMS_CONVERTER, HYPERPARAMS_SAMPLER
from rl_zoo3.utils import get_callback_list
from sb3_contrib.common.vec_env import AsyncEval
from stable_baselines3 import HerReplayBuffer
from stable_baselines3.common.base_class import BaseAlgorithm
from stable_baselines3.common.vec_env import VecEnv

from policies.LeaderFollowerAlgorithm import LeaderFollowerAlgorithm


class ExperimentManagerLF(ExperimentManager):
    def __init__(
            self,
            algorithm_config: dict[str, Any],
            run_config: dict[str, Any],
    ):
        super().__init__(
            algo=f"{algorithm_config["leader"]}_{algorithm_config["follower"]}",
            **run_config,
        )

        if (Path(__file__).parent / "hyperparams").is_dir():
            # Package version
            default_path = Path(__file__).parent
        else:
            # Take the root folder
            default_path = Path(__file__).parent.parent

        self.leader_algorithm = algorithm_config["leader_algorithm"]
        self.leader_policy = algorithm_config["leader_policy"]
        self.leader_config = algorithm_config["leader_config"] or str(
            default_path / f"hyperparams/{self.leader_algorithm}.yml")

        self.follower_algorithm = algorithm_config["follower_algorithm"]
        self.follower_policy = algorithm_config["follower_policy"]
        self.follower_config = algorithm_config["follower_config"] or str(
            default_path / f"hyperparams/{self.follower_algorithm}.yml")

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
        n_envs = 1 if (self.leader_algorithm == "ars"
                       or self.follower_algorithm == "ars"
                       or self.optimize_hyperparameters) else self.n_envs
        env = self.create_envs(n_envs, no_log=False)

        self._hyperparams = self._preprocess_action_noise(hyperparams, saved_hyperparams, env)

        if self.continue_training:
            model = self._load_pretrained_agent(self._hyperparams, env)
        elif self.optimize_hyperparameters:
            env.close()
            return None
        else:
            # Train an agent from scratch
            model = LeaderFollowerAlgorithm(
                leader_algorithm=self.leader_algorithm,
                leader_policy=self.leader_policy,
                follower_algorithm=self.follower_algorithm,
                follower_policy=self.follower_policy,
                env=env,
                tensorboard_log=self.tensorboard_log,
                seed=self.seed,
                verbose=self.verbose,
                device=self.device,
                **self._hyperparams,
            )

        self._save_config(saved_hyperparams)
        return model, saved_hyperparams

    def learn(self, model: LeaderFollowerAlgorithm) -> None:
        """
        :param model: an initialized RL model
        """
        kwargs: dict[str, Any] = {}
        # log_interval == -1 -> default
        # < -2 -> no auto-logging
        if self.log_interval > -1:
            kwargs = {"log_interval": self.log_interval}
        elif self.log_interval < -1:
            # Deactivate auto-logging, helpful when using callback like LogEveryNTimesteps
            kwargs = {"log_interval": None}

        if len(self.callbacks) > 0:
            kwargs["callback"] = self.callbacks

        learn_kwargs_leader = kwargs.copy()
        learn_kwargs_follower = kwargs.copy()
        if self.n_envs > 1:
            if self.leader_algorithm == "ars":
                learn_kwargs_leader["async_eval"] = AsyncEval(
                    [lambda: self.create_envs(n_envs=1, no_log=True) for _ in range(self.n_envs)], model.leader_policy
                )
            if self.follower_algorithm == "ars":
                learn_kwargs_follower["async_eval"] = AsyncEval(
                    [lambda: self.create_envs(n_envs=1, no_log=True) for _ in range(self.n_envs)], model.follower_policy
                )

        try:
            model.learn(self.n_timesteps, leader_kwargs=learn_kwargs_leader, follower_kwargs=learn_kwargs_follower,)
        except KeyboardInterrupt:
            # this allows to save the model when interrupting training
            pass
        finally:
            # Clean progress bar
            if len(self.callbacks) > 0:
                self.callbacks[0].on_training_end()
            # Release resources
            try:
                assert model.env is not None
                model.env.close()
            except EOFError:
                pass

    def read_hyperparameters(self) -> tuple[dict[str, Any], dict[str, Any]]:
        leader_hyperparams = self._read_model_hyperparameters(self.leader_config)
        follower_hyperparams = self._read_model_hyperparameters(self.follower_config)
        leader_hyperparams.update(follower_hyperparams)
        hyperparams = leader_hyperparams

        if self.storage and self.study_name and self.trial_id:
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

    def _read_model_hyperparameters(self, config_path) -> dict[str, Any]:
        print(f"Loading hyperparameters from {config_path}")

        if config_path.endswith(".yml") or config_path.endswith(".yaml"):
            # Load hyperparameters from yaml file
            with open(config_path) as f:
                hyperparams_dict = yaml.safe_load(f)
        elif config_path.endswith(".py"):
            global_variables: dict = {}
            # Load hyperparameters from python file
            exec(Path(config_path).read_text(), global_variables)
            hyperparams_dict = global_variables["hyperparams"]
        else:
            # Load hyperparameters from python package
            hyperparams_dict = importlib.import_module(config_path).hyperparams
            # raise ValueError(f"Unsupported config file format: {config_path}")

        if self.env_name.gym_id in list(hyperparams_dict.keys()):
            hyperparams = hyperparams_dict[self.env_name.gym_id]
        elif self._is_atari:
            hyperparams = hyperparams_dict["atari"]
        else:
            raise ValueError(f"Hyperparameters not found for {self.algo}-{self.env_name.gym_id} in {config_path}")

        return hyperparams

    def load_trial(
            self, storage: str, study_name: str, trial_id: int = None, convert: bool = True
    ) -> dict[str, Any]:
        if storage.endswith(".log"):
            optuna_storage = optuna.storages.JournalStorage(optuna.storages.journal.JournalFileBackend(storage))
        else:
            optuna_storage = storage  # type: ignore[assignment]
        study = optuna.load_study(storage=optuna_storage, study_name=study_name)
        if trial_id is not None:
            params = study.trials[trial_id].params
        else:
            params = study.best_trial.params

        if convert:
            # TODO: test
            params = HYPERPARAMS_CONVERTER[self.leader_algorithm](params)
            params = HYPERPARAMS_CONVERTER[self.follower_algorithm](params)
            return params
        return params

    def _preprocess_action_noise(
            self, hyperparams: dict[str, Any], saved_hyperparams: dict[str, Any], env: VecEnv
    ) -> dict[str, Any]:
        # TODO
        raise NotImplementedError()

    def _load_pretrained_agent(self, hyperparams: dict[str, Any], env: VecEnv) -> BaseAlgorithm:
        # Continue training
        print("Loading pretrained agent")
        # Policy should not be changed
        del hyperparams["policy"]

        if "policy_kwargs" in hyperparams.keys():
            del hyperparams["policy_kwargs"]

        model = LeaderFollowerAlgorithm.load(
            self.trained_agent,
            env=env,
            seed=self.seed,
            tensorboard_log=self.tensorboard_log,
            verbose=self.verbose,
            device=self.device,
            **hyperparams,
        )

        replay_buffer_path = os.path.join(os.path.dirname(self.trained_agent), "replay_buffer.pkl")

        if os.path.exists(replay_buffer_path):
            print("Loading replay buffer")
            # `truncate_last_traj` will be taken into account only if we use HER replay buffer
            assert hasattr(
                model, "load_replay_buffer"
            ), "The current model doesn't have a `load_replay_buffer` to load the replay buffer"
            model.load_replay_buffer(replay_buffer_path, truncate_last_traj=self.truncate_last_trajectory)
        return model

    def objective(self, trial: optuna.Trial) -> float:
        kwargs = self._hyperparams.copy()

        n_envs = 1 if (self.leader_algorithm == "ars" or self.follower_algorithm == "ars") else self.n_envs

        additional_args = {
            "using_her_replay_buffer": kwargs.get("replay_buffer_class") == HerReplayBuffer,
            "her_kwargs": kwargs.get("replay_buffer_kwargs", {}),
        }
        # Pass n_actions to initialize DDPG/TD3 noise sampler
        # Sample candidate hyperparameters
        sampled_hyperparams_leader = HYPERPARAMS_SAMPLER[self.leader_algorithm](trial, self.leader_n_actions, n_envs,
                                                                                additional_args)
        leader_kwargs = kwargs.copy()
        leader_kwargs.update(sampled_hyperparams_leader)

        sampled_hyperparams_follower = HYPERPARAMS_SAMPLER[self.follower_algorithm](trial, self.follower_n_actions,
                                                                                    n_envs, additional_args)
        follower_kwargs = kwargs.copy()
        follower_kwargs.update(sampled_hyperparams_follower)

        env = self.create_envs(n_envs, no_log=True)

        # By default, do not activate verbose output to keep
        # stdout clean with only the trial's results
        trial_verbosity = 0
        # Activate verbose mode for the trial in debug mode
        # See PR #214
        if self.verbose >= 2:
            trial_verbosity = self.verbose

        model = LeaderFollowerAlgorithm(
            leader_algorithm=self.leader_algorithm,
            leader_policy=self.leader_policy,
            follower_algorithm=self.follower_algorithm,
            follower_policy=self.follower_policy,
            env=env,
            tensorboard_log=None,
            # We do not seed the trial
            seed=None,
            verbose=trial_verbosity,
            device=self.device,
            leader_kwargs=leader_kwargs,
            follower_kwargs=follower_kwargs,
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

        learn_kwargs_leader = {}
        learn_kwargs_follower = {}
        # Special case for ARS
        if self.n_envs > 1:
            if self.leader_algorithm == "ars":
                learn_kwargs_leader["async_eval"] = AsyncEval(
                    [lambda: self.create_envs(n_envs=1, no_log=True) for _ in range(self.n_envs)], model.leader_policy
                )
            if self.follower_algorithm == "ars":
                learn_kwargs_follower["async_eval"] = AsyncEval(
                    [lambda: self.create_envs(n_envs=1, no_log=True) for _ in range(self.n_envs)], model.follower_policy
                )

        try:
            model.learn(self.n_timesteps, callback=callbacks, leader_kwargs=learn_kwargs_leader,
                        follower_kwargs=learn_kwargs_follower)
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

        del model.env, eval_env
        del model

        if is_pruned:
            raise optuna.exceptions.TrialPruned()

        return reward

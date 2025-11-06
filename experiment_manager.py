import os
from pprint import pprint

import optuna
from rl_zoo3.callbacks import TrialEvalCallback
from rl_zoo3.exp_manager import ExperimentManager
from rl_zoo3.hyperparams_opt import HYPERPARAMS_SAMPLER
from rl_zoo3.utils import get_callback_list
from sb3_contrib.common.vec_env import AsyncEval
from stable_baselines3 import HerReplayBuffer

from policies.LeaderFollowerAlgorithm import LeaderFollowerAlgorithm


class ExperimentManagerLF(ExperimentManager):
    def objective(self, trial: optuna.Trial) -> float:
        kwargs = self._hyperparams.copy()
        del kwargs["leader_algorithm"]
        del kwargs["leader_policy"]
        del kwargs["follower_algorithm"]
        del kwargs["follower_policy"]
        # TODO: find a better solution where kwargs are actually used in LF

        n_envs = 1 if self.algo == "ars" else self.n_envs


        additional_args = {
            "using_her_replay_buffer": kwargs.get("replay_buffer_class") == HerReplayBuffer,
            "her_kwargs": kwargs.get("replay_buffer_kwargs", {}),
        }
        # Pass n_actions to initialize DDPG/TD3 noise sampler
        # Sample candidate hyperparameters
        hyperparams_leader = kwargs.copy()
        sampled_hyperparams_leader = HYPERPARAMS_SAMPLER[self._hyperparams["leader_algorithm"]](trial, self.n_actions, n_envs, additional_args)
        hyperparams_leader.update(sampled_hyperparams_leader)
        if "policy_kwargs" in kwargs:
            hyperparams_leader["policy_kwargs"] = hyperparams_leader["policy_kwargs"] | kwargs["policy_kwargs"]

        hyperparams_follower = kwargs.copy()
        # TODO: make the follower actions a variable?
        sampled_hyperparams_follower = HYPERPARAMS_SAMPLER[self._hyperparams["leader_algorithm"]](trial, 2, n_envs, additional_args)
        hyperparams_follower.update(sampled_hyperparams_follower)
        if "policy_kwargs" in kwargs:
            hyperparams_follower["policy_kwargs"] = hyperparams_follower["policy_kwargs"] | kwargs["policy_kwargs"]

        env = self.create_envs(n_envs, no_log=True)

        # By default, do not activate verbose output to keep
        # stdout clean with only the trial's results
        trial_verbosity = 0
        # Activate verbose mode for the trial in debug mode
        # See PR #214
        if self.verbose >= 2:
            trial_verbosity = self.verbose

        model = LeaderFollowerAlgorithm(
            leader_algorithm=self._hyperparams["leader_algorithm"],
            leader_policy=self._hyperparams["leader_policy"],
            follower_algorithm=self._hyperparams["follower_algorithm"],
            follower_policy=self._hyperparams["follower_policy"],
            learning_rate=sampled_hyperparams_leader["learning_rate"],
            env=env,
            leader_algorithm_kwargs=hyperparams_leader,
            follower_algorithm_kwargs=hyperparams_follower,

            tensorboard_log=self.tensorboard_log,
            seed=self.seed,
            verbose=trial_verbosity,
            device=self.device,
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

        del model.env, eval_env
        del model

        if is_pruned:
            raise optuna.exceptions.TrialPruned()

        return reward

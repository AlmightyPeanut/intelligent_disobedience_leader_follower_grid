import io
import os
import pathlib
import sys
import time
import warnings
from copy import deepcopy
from typing import Any, TypeVar, Union, Optional, Iterable

import gymnasium as gym
import numpy as np
import torch as th
from gymnasium import spaces
from rl_zoo3 import ALGOS
from sb3_contrib.common.vec_env import AsyncEval
from stable_baselines3.common.base_class import BaseAlgorithm, SelfBaseAlgorithm
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.off_policy_algorithm import OffPolicyAlgorithm
from stable_baselines3.common.policies import BasePolicy
from stable_baselines3.common.type_aliases import MaybeCallback, GymEnv, Schedule, TrainFreq, RolloutReturn, \
    TrainFrequencyUnit
from stable_baselines3.common.utils import should_collect_more_steps, safe_mean
from stable_baselines3.common.vec_env import VecEnv
from stable_baselines3.dqn import MultiInputPolicy

from dummy_vec_env import DummyVecEnvIntRewards
from minigrid_env.environment import LavaEnv, FollowerAction, LeaderAction
from policies.feature_extractors.LavaEnvFeaturesExtractor import LavaEnvCNNFeaturesExtractor

SelfLeaderFollowerAlgorithm = TypeVar("SelfLeaderFollowerAlgorithm", bound="LeaderFollowerAlgorithm")


class LeaderFollowerAlgorithm(BaseAlgorithm):
    policy_aliases = {
        # policy is not used but needed for sb3
        "MultiInput": MultiInputPolicy,
    }

    def __init__(
            self,
            leader_algorithm: str,
            leader_policy: str | type[BasePolicy],
            follower_algorithm: str,
            follower_policy: str | type[BasePolicy],
            env: GymEnv | str | None,
            learning_rate: float | Schedule,

            leader_algorithm_kwargs: dict[str, Any] | None = None,
            follower_algorithm_kwargs: dict[str, Any] | None = None,

            stats_window_size: int = 100,
            tensorboard_log: str = None,
            verbose: int = 0,
            device: th.device | str = "auto",
            support_multi_env: bool = True,
            monitor_wrapper: bool = True,
            seed: int = None,
            use_sde: bool = False,
            sde_sample_freq: int = -1,
            supported_action_spaces: tuple[type[spaces.Space], ...] = None,

            # used to validate reward signals
            train_leader: bool = True,
            train_follower: bool = True,
    ):
        super().__init__(
            policy="MultiInput",  # will not be used
            env=env,
            learning_rate=learning_rate,
            stats_window_size=stats_window_size,
            tensorboard_log=os.path.join(tensorboard_log, "common") if tensorboard_log is not None else None,
            verbose=verbose,
            device=device,
            support_multi_env=support_multi_env,
            monitor_wrapper=monitor_wrapper,
            seed=seed,
            use_sde=use_sde,
            sde_sample_freq=sde_sample_freq,
            supported_action_spaces=supported_action_spaces,
        )

        self.follower_is_off_policy = False
        self.leader_is_off_policy = False
        self.state_split_point = 0
        assert train_leader or train_follower, "At least one of the two algorithms must be trained"
        self.train_leader = train_leader
        self.train_follower = train_follower
        seed_seq = np.random.SeedSequence(seed)
        self._np_random = np.random.Generator(np.random.PCG64(seed_seq))

        if not isinstance(env.action_space, spaces.MultiDiscrete):
            raise ValueError("Leader and follower environment must use a MultiDiscrete action space")

        ############## Leader setup ##############
        if self.train_leader:
            self.leader_algorithm_kwargs = leader_algorithm_kwargs or {}
            if "policy_kwargs" not in self.leader_algorithm_kwargs:
                self.leader_algorithm_kwargs["policy_kwargs"] = {}
            self.leader_algorithm_kwargs["policy_kwargs"]["features_extractor_class"] = LavaEnvCNNFeaturesExtractor

            # This won't be used anyway, but sb3 needs it
            def make_leader_env() -> gym.Env:
                return LavaEnv(only_leader_env=True)

            leader_env = make_vec_env(
                make_leader_env,
                n_envs=self.n_envs,
                seed=self.seed,
                vec_env_cls=DummyVecEnvIntRewards,
            )

            self.leader_model = ALGOS[leader_algorithm](
                policy=leader_policy,
                env=leader_env,
                tensorboard_log=os.path.join(tensorboard_log, "leader") if tensorboard_log is not None else None,
                **self.leader_algorithm_kwargs,
            )
        else:
            self.leader_model = None

        ############# Follower setup #############
        if self.train_follower:
            self.follower_algorithm_kwargs = follower_algorithm_kwargs or {}
            if "policy_kwargs" not in self.follower_algorithm_kwargs:
                self.follower_algorithm_kwargs["policy_kwargs"] = {}
            self.follower_algorithm_kwargs["policy_kwargs"]["features_extractor_class"] = LavaEnvCNNFeaturesExtractor

            # This won't be used anyway, but sb3 needs it
            def make_follower_env() -> gym.Env:
                return LavaEnv(only_follower_env=True)

            follower_env = make_vec_env(
                make_follower_env,
                n_envs=self.n_envs,
                seed=self.seed,
                vec_env_cls=DummyVecEnvIntRewards,
            )

            self.follower_model = ALGOS[follower_algorithm](
                policy=follower_policy,
                env=follower_env,
                tensorboard_log=os.path.join(tensorboard_log, "follower") if tensorboard_log is not None else None,
                **self.follower_algorithm_kwargs,
            )
        else:
            self.follower_model = None

    def _setup_model(self) -> None:
        if self.train_leader:
            self.leader_model._setup_model()
        if self.train_follower:
            self.follower_model._setup_model()

    def learn(
            self: SelfLeaderFollowerAlgorithm,
            total_timesteps: int,
            callback: MaybeCallback = None,
            log_interval: int = 1,
            tb_log_name: str = "run",
            reset_num_timesteps: bool = True,
            progress_bar: bool = False,
            async_eval_leader: AsyncEval = None,
            async_eval_follower: AsyncEval = None,
    ) -> SelfLeaderFollowerAlgorithm:
        if self.train_leader:
            _, callback = self.leader_model._setup_learn(
                total_timesteps,
                callback,
                reset_num_timesteps,
                tb_log_name,
                progress_bar,
            )

        if self.train_follower:
            _, callback = self.follower_model._setup_learn(
                total_timesteps,
                callback,
                reset_num_timesteps,
                tb_log_name,
                progress_bar,
            )

        total_timesteps, callback = self._setup_learn(
            total_timesteps,
            callback,
            reset_num_timesteps,
            tb_log_name,
            progress_bar,
        )

        callback.on_training_start(locals(), globals())

        assert self.env is not None, "You must set the environment before calling learn()"

        train_freq = None
        if self.train_leader and issubclass(type(self.leader_model), OffPolicyAlgorithm):
            assert isinstance(self.leader_model.train_freq, TrainFreq)
            self.leader_is_off_policy = True
            train_freq = self.leader_model.train_freq

        if self.train_follower and issubclass(type(self.follower_model), OffPolicyAlgorithm):
            assert isinstance(self.follower_model.train_freq, TrainFreq)
            self.follower_is_off_policy = True
            train_freq = self.follower_model.train_freq

        if (self.leader_is_off_policy or not self.train_leader) and (self.follower_is_off_policy or not self.train_follower):
            self._learn_off_policy(
                train_freq=train_freq,
                callback=callback,
                total_timesteps=total_timesteps,
                log_interval=log_interval,
            )
        elif (not self.leader_is_off_policy or not self.train_leader) and (not self.follower_is_off_policy or not self.train_follower):
            self._learn_on_policy(
                callback,
            )
        else:
            raise ValueError("Both leader and follower must be either off-policy or on-policy")

        callback.on_training_end()

        return self

    def _learn_on_policy(
            self,
            callback: BaseCallback,
    ) -> SelfLeaderFollowerAlgorithm:
        # TODO
        raise NotImplementedError()

    def _learn_off_policy(
            self,
            train_freq: TrainFreq,
            callback: BaseCallback,
            total_timesteps: int,
            log_interval: int,
    ) -> None:
        while self.num_timesteps < total_timesteps:
            rollout = self.collect_rollouts(
                self.env,
                train_freq=train_freq,
                callback=callback,
                learning_starts=self.leader_model.learning_starts if self.train_leader else self.follower_model.learning_starts,
                log_interval=log_interval,
            )

            if not rollout.continue_training:
                break

            if self.num_timesteps > 0 and self.num_timesteps > (self.leader_model.learning_starts if self.train_leader else self.follower_model.learning_starts):
                # If no `gradient_steps` is specified,
                # do as many gradients steps as steps performed during the rollout
                if self.train_leader:
                    gradient_steps = self.leader_model.gradient_steps if self.leader_model.gradient_steps > 0 else rollout.episode_timesteps
                    self.leader_model.train(batch_size=self.leader_model.batch_size, gradient_steps=gradient_steps)
                if self.train_follower:
                    gradient_steps = self.follower_model.gradient_steps if self.follower_model.gradient_steps > 0 else rollout.episode_timesteps
                    self.follower_model.train(batch_size=self.follower_model.batch_size, gradient_steps=gradient_steps)


    def collect_rollouts(
            self,
            env,
            train_freq,
            callback,
            learning_starts,
            log_interval,
    ) -> RolloutReturn:
        # Switch to eval mode (this affects batch norm / dropout)
        if self.train_leader:
            self.leader_model.policy.set_training_mode(False)
        if self.train_follower:
            self.follower_model.policy.set_training_mode(False)

        num_collected_steps, num_collected_episodes = 0, 0

        assert isinstance(self.env, VecEnv), "You must pass a VecEnv"
        assert train_freq.frequency > 0, "Should at least collect one step or episode."

        if env.num_envs > 1:
            assert train_freq.unit == TrainFrequencyUnit.STEP, "You must use only one env when doing episodic training."

        if self.train_leader:
            if self.leader_model.use_sde:
                self.leader_model.actor.reset_noise(env.num_envs)
        if self.train_follower:
            if self.follower_model.use_sde:
                self.follower_model.actor.reset_noise(env.num_envs)

        callback.on_rollout_start()
        continue_training = True
        while should_collect_more_steps(train_freq, num_collected_steps, num_collected_episodes):
            # Sample a new noise matrix
            if self.train_leader:
                if self.leader_model.use_sde and self.leader_model.sde_sample_freq > 0 and num_collected_steps % self.leader_model.sde_sample_freq == 0:
                    self.leader_model.actor.reset_noise(env.num_envs)
            if self.train_follower:
                if self.follower_model.use_sde and self.follower_model.sde_sample_freq > 0 and num_collected_steps % self.follower_model.sde_sample_freq == 0:
                    self.follower_model.actor.reset_noise(env.num_envs)

            # Select action randomly or according to policy
            ((leader_actions, leader_buffer_actions),
             (follower_actions, follower_buffer_actions)) = self._sample_actions(learning_starts, env.num_envs)

            # Rescale and perform action
            actions = np.stack((leader_actions, follower_actions), axis=-1)
            (new_obs, encoded_rewards, dones, infos) = env.step(actions)
            decoded_rewards = np.array(list(map(LavaEnv.decode_reward, encoded_rewards)), dtype=encoded_rewards.dtype)

            self.num_timesteps += env.num_envs
            if self.train_leader:
                self.leader_model.num_timesteps = self.num_timesteps
            if self.train_follower:
                self.follower_model.num_timesteps = self.num_timesteps
            num_collected_steps += 1

            # Give access to local variables
            callback.update_locals(locals())
            # Only stop training if the return value is False, not when it is None.
            if not callback.on_step():
                return RolloutReturn(num_collected_steps * env.num_envs, num_collected_episodes,
                                     continue_training=False)

            # Retrieve reward and episode length if using Monitor wrapper
            self._update_info_buffer(infos, dones)
            if self.train_leader:
                self.leader_model._update_info_buffer(infos, dones)
            if self.train_follower:
                self.follower_model._update_info_buffer(infos, dones)

            # Store data in replay buffer (normalized action and unnormalized observation)
            self._store_transition(
                leader_buffer_actions,
                follower_buffer_actions,
                new_obs,
                decoded_rewards,
                dones, infos,
            )

            self._update_current_progress_remaining(self.num_timesteps, self._total_timesteps)
            if self.train_leader:
                self.leader_model._update_current_progress_remaining(self.num_timesteps, self._total_timesteps)
                self.leader_model._on_step()
            if self.train_follower:
                self.follower_model._update_current_progress_remaining(self.num_timesteps, self._total_timesteps)
                self.follower_model._on_step()


            for idx, done in enumerate(dones):
                if done:
                    # Update stats
                    num_collected_episodes += 1
                    self._episode_num += 1
                    if self.train_leader:
                        self.leader_model._episode_num += 1
                    if self.train_follower:
                        self.follower_model._episode_num += 1

                    kwargs = dict(indices=[idx]) if env.num_envs == 1 else dict()
                    if self.train_leader:
                        if self.leader_model.action_noise is not None:
                            self.leader_model.action_noise.reset(**kwargs)
                    if self.train_follower:
                        if self.follower_model.action_noise is not None:
                            self.follower_model.action_noise.reset(**kwargs)

                    # Log training infos
                    if log_interval is not None and self._episode_num % log_interval == 0:
                        self.dump_logs()

        callback.on_rollout_end()

        return RolloutReturn(num_collected_steps * env.num_envs, num_collected_episodes, continue_training)

    def dump_logs(self) -> None:
        """
        Write log data.
        """
        assert self.ep_info_buffer is not None
        assert self.ep_success_buffer is not None

        time_elapsed = max((time.time_ns() - self.start_time) / 1e9, sys.float_info.epsilon)
        fps = int((self.num_timesteps - self._num_timesteps_at_start) / time_elapsed)
        self.logger.record("time/episodes", self._episode_num, exclude="tensorboard")
        if len(self.ep_info_buffer) > 0 and len(self.ep_info_buffer[0]) > 0:
            decoded_rewards = np.array([LavaEnv.decode_reward(ep_info["r"]) for ep_info in self.ep_info_buffer])
            leader_rewards = decoded_rewards[..., 0]
            follower_rewards = decoded_rewards[..., 1]
            if self.train_leader:
                self.logger.record("rollout/leader/ep_rew_mean", safe_mean(leader_rewards))
                self.logger.record("rollout/times_goal_reached", np.sum(leader_rewards == 1))
                self.logger.record("rollout/goal_reached_pct", safe_mean(leader_rewards == 1))

                self.logger.record("rollout/times_stepped_in_lava", np.sum(leader_rewards == -1))
                self.logger.record("rollout/stepped_in_lava_pct", safe_mean(leader_rewards == -1))

                self.logger.record("rollout/times_episode_ended", np.sum(leader_rewards == 1))
                self.logger.record("rollout/episode_ended_pct", safe_mean(leader_rewards == 1))
            if self.train_follower:
                self.logger.record("rollout/follower/ep_rew_mean", safe_mean(follower_rewards))
                self.logger.record("rollout/follower/good_disobedience", np.sum(follower_rewards > 0))
                self.logger.record("rollout/follower/good_disobedience_pct", safe_mean(follower_rewards > 0))
                self.logger.record("rollout/follower/bad_disobedience", np.sum(follower_rewards < 0))
                self.logger.record("rollout/follower/bad_disobedience_pct", safe_mean(follower_rewards < 0))
            self.logger.record("rollout/ep_len_mean", safe_mean([ep_info["l"] for ep_info in self.ep_info_buffer]))
        self.logger.record("time/fps", fps)
        self.logger.record("time/time_elapsed", int(time_elapsed), exclude="tensorboard")
        self.logger.record("time/total_timesteps", self.num_timesteps, exclude="tensorboard")
        if self.use_sde:
            if self.train_leader:
                self.logger.record("train/leader/std", (self.leader_model.actor.get_std()).mean().item())  # type: ignore[operator]
            if self.train_follower:
                self.logger.record("train/follower/std", (self.follower_model.actor.get_std()).mean().item())  # type: ignore[operator]

        if len(self.ep_success_buffer) > 0:
            self.logger.record("rollout/success_rate", safe_mean(self.ep_success_buffer))
        # Pass the number of timesteps for tensorboard
        self.logger.dump(step=self.num_timesteps)
        if self.train_leader:
            if self.leader_is_off_policy:
                self.leader_model.dump_logs()
            else:
                self.leader_model.dump_logs(iteration=self.num_timesteps)
        if self.train_follower:
            if self.follower_is_off_policy:
                self.follower_model.dump_logs()
            else:
                self.follower_model.dump_logs(iteration=self.num_timesteps)


    def _store_transition(
            self,
            leader_actions: np.ndarray,
            follower_actions: np.ndarray,
            new_obs: np.ndarray,
            rewards: np.ndarray,
            dones: np.ndarray,
            infos: list[dict],
    ) -> None:
        # Store only the unnormalized version
        if self._vec_normalize_env is not None:
            new_obs_ = self._vec_normalize_env.get_original_obs()
            rewards_ = self._vec_normalize_env.get_original_reward()
        else:
            # Avoid changing the original ones
            self._last_original_obs, new_obs_, rewards_ = self._last_obs, new_obs, rewards

        # Avoid modification by reference
        next_obs = deepcopy(new_obs_)

        # As the VecEnv resets automatically, new_obs is already the
        # first observation of the next episode
        for i, done in enumerate(dones):
            if done and infos[i].get("terminal_observation") is not None:
                if isinstance(next_obs, dict):
                    next_obs_ = infos[i]["terminal_observation"]
                    # VecNormalize normalizes the terminal observation
                    if self._vec_normalize_env is not None:
                        next_obs_ = self._vec_normalize_env.unnormalize_obs(next_obs_)
                    # Replace next obs for the correct envs
                    for key in next_obs.keys():
                        next_obs[key][i] = next_obs_[key]
                else:
                    next_obs[i] = infos[i]["terminal_observation"]
                    # VecNormalize normalizes the terminal observation
                    if self._vec_normalize_env is not None:
                        next_obs[i] = self._vec_normalize_env.unnormalize_obs(next_obs[i, :])

        last_obs = self._last_original_obs
        last_obs["follower_image"] = LavaEnv.prepare_follower_obs(last_obs["follower_image"], leader_actions)
        leader_rewards = rewards_[..., 0]
        follower_rewards = rewards_[..., 1]

        if self.train_leader:
            self.leader_model.replay_buffer.add(
                last_obs["image"],
                next_obs["image"],
                leader_actions,
                leader_rewards,
                dones,
                infos,
            )

        if self.train_follower:
            self.follower_model.replay_buffer.add(
                last_obs["follower_image"],
                next_obs["follower_image"],
                follower_actions,
                follower_rewards,
                dones,
                infos,
            )

        self._last_obs = new_obs
        # Save the unnormalized observation
        if self._vec_normalize_env is not None:
            self._last_original_obs = new_obs_

    def _sample_actions(
            self,
            learning_starts: int,
            n_envs: int = 1,
    ) -> tuple[tuple[np.ndarray, np.ndarray], tuple[np.ndarray, np.ndarray]]:
        # Select action randomly or according to policy
        if not self.train_leader:
            # Sampling is taken from discrete gymnasium action space
            unscaled_leader_action = np.array([self._sample_leader_action() for _ in range(n_envs)])
        else:
            if self.num_timesteps < learning_starts and not (self.leader_model.use_sde and self.leader_model.use_sde_at_warmup):
                # Warmup phase
                unscaled_leader_action = np.array([self.leader_model.action_space.sample() for _ in range(n_envs)])
            else:
                # Note: when using continuous actions,
                # we assume that the policy uses tanh to scale the action
                # We use non-deterministic action in the case of SAC, for TD3, it does not matter
                assert self._last_obs is not None, "self._last_obs was not set"
                unscaled_leader_action, _ = self.leader_model.predict(self._last_obs["image"], deterministic=False)
                _last_follower_obs = LavaEnv.prepare_follower_obs(self._last_obs["follower_image"], unscaled_leader_action)

        if not self.train_follower:
            # Always obey to let the leader freely act in the environment
            unscaled_follower_action = np.array([FollowerAction.obey for _ in range(n_envs)])
        else:
            if self.num_timesteps < learning_starts and not (self.follower_model.use_sde and self.follower_model.use_sde_at_warmup):
                # Warmup phase
                unscaled_follower_action = np.array([self.follower_model.action_space.sample() for _ in range(n_envs)])
            else:
                # Note: when using continuous actions,
                # we assume that the policy uses tanh to scale the action
                # We use non-deterministic action in the case of SAC, for TD3, it does not matter
                assert self._last_obs is not None, "self._last_obs was not set"
                _last_follower_obs = LavaEnv.prepare_follower_obs(self._last_obs["follower_image"], unscaled_leader_action)
                unscaled_follower_action, _ = self.follower_model.predict(_last_follower_obs, deterministic=False)


        # Rescale the action from [low, high] to [-1, 1]
        leader_buffer_action = unscaled_leader_action
        leader_action = leader_buffer_action
        if self.train_leader:
            if isinstance(self.leader_model.action_space, spaces.Box):
                scaled_leader_action = self.leader_model.policy.scale_action(unscaled_leader_action)

                # Add noise to the action (improve exploration)
                if self.leader_model.action_noise is not None:
                    scaled_leader_action = np.clip(scaled_leader_action + self.leader_model.action_noise(), -1, 1)

                # We store the scaled action in the buffer
                leader_buffer_action = scaled_leader_action
                leader_action = self.leader_model.policy.unscale_action(scaled_leader_action)

        # Discrete case, no need to normalize or clip
        follower_buffer_action = unscaled_follower_action
        follower_action = follower_buffer_action
        if self.train_follower:
            if isinstance(self.follower_model.action_space, spaces.Box):
                scaled_follower_action = self.follower_model.policy.scale_action(unscaled_follower_action)

                # Add noise to the action (improve exploration)
                if self.follower_model.action_noise is not None:
                    scaled_follower_action = np.clip(scaled_follower_action + self.follower_model.action_noise(), -1, 1)

                # We store the scaled action in the buffer
                follower_buffer_action = scaled_follower_action
                follower_action = self.follower_model.policy.unscale_action(scaled_follower_action)
        return (leader_action, leader_buffer_action), (follower_action, follower_buffer_action)

    def _sample_leader_action(self) -> int:
        # Make it more likely that the leader moves initially
        return self._np_random.choice(len(LeaderAction), p=[.25, .25, .5])

    def predict(
            self,
            observation: np.ndarray | dict[str, np.ndarray],
            state: tuple[np.ndarray, ...] = None,
            episode_start: np.ndarray = None,
            deterministic: bool = False,
    ) -> tuple[np.ndarray, tuple[np.ndarray, ...] | None]:
        leader_state, follower_state = self.get_model_states(state)

        if self.train_leader:
            leader_action, leader_state = self.leader_model.predict(observation["image"], leader_state, episode_start,
                                                                    deterministic)
        else:
            leader_action = np.array([self._sample_leader_action() for _ in range(observation["image"].shape[0])])
            leader_state = None

        if self.train_follower:
            follower_obs = LavaEnv.prepare_follower_obs(observation["follower_image"], leader_action)
            follower_action, follower_state = self.follower_model.predict(follower_obs, follower_state, episode_start,
                                                                          deterministic)
        else:
            follower_action = np.array([FollowerAction.obey for _ in range(observation["follower_image"].shape[0])])
            follower_state = None

        action = np.stack((leader_action, follower_action), axis=-1)

        state = self.combine_model_states(leader_state, follower_state)
        return action, state

    def combine_model_states(self,
                             leader_state: tuple[np.ndarray, ...] | None,
                             follower_state: tuple[np.ndarray, ...] | None) -> tuple[np.ndarray, ...] | None:
        """
        Combines leader and follower states into a single state tuple.

        Args:
            leader_state: State tuple from leader model or None
            follower_state: State tuple from follower model or None

        Returns:
            Combined state tuple or None if both inputs are None
        """
        if leader_state is None and follower_state is None:
            return None

        combined_state = []

        if leader_state is not None:
            combined_state.extend(leader_state)
        if follower_state is not None:
            combined_state.extend(follower_state)
        self.state_split_point = len(leader_state) if leader_state is not None else 0

        return tuple(combined_state) if combined_state else None

    def get_model_states(self,
                         state: tuple[np.ndarray, ...] = None
                         ) -> tuple[tuple[np.ndarray, ...] | None, tuple[np.ndarray, ...] | None]:
        """
        Splits the combined state into leader and follower states.

        Args:
            state: Combined state tuple or None

        Returns:
            Tuple of (leader_state, follower_state), both can be None
        """
        if state is None:
            return None, None

        leader_state = state[:self.state_split_point] if self.state_split_point > 0 else None
        follower_state = state[self.state_split_point:] if self.state_split_point < len(state) else None

        return leader_state, follower_state

    def save(
            self,
            path: Union[str, pathlib.Path, io.BufferedIOBase],
            exclude: Optional[Iterable[str]] = None,
            include: Optional[Iterable[str]] = None,
    ) -> None:
        if self.train_leader:
            self.leader_model.save(os.path.join(path, 'leader'), exclude, include)
        if self.train_follower:
            self.follower_model.save(os.path.join(path, 'follower'), exclude, include)

    def load(  # noqa: C901
            cls: type[SelfBaseAlgorithm],
            path: Union[str, pathlib.Path, io.BufferedIOBase],
            env: Optional[GymEnv] = None,
            device: Union[th.device, str] = "auto",
            custom_objects: Optional[dict[str, Any]] = None,
            print_system_info: bool = False,
            force_reset: bool = True,
            **kwargs,
    ) -> SelfBaseAlgorithm:
        # TODO
        warnings.warn("Trying to load a policy from a saved model is not supported yet.")
        raise NotImplementedError()

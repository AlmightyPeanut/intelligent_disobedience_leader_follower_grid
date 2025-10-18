import io
import os
import pathlib
from typing import Any, TypeVar, Union, Optional, Iterable

import numpy as np
import torch as th
from rl_zoo3 import ALGOS
from stable_baselines3.common.base_class import BaseAlgorithm, SelfBaseAlgorithm
from stable_baselines3.common.policies import BasePolicy
from stable_baselines3.common.type_aliases import MaybeCallback, GymEnv, Schedule

from minigrid_env.environment import LavaEnv

SelfLeaderFollowerAlgorithm = TypeVar("SelfLeaderFollowerAlgorithm", bound="LeaderFollowerAlgorithm")


class LeaderFollowerAlgorithm(BaseAlgorithm):
    leader_model: BaseAlgorithm
    follower_model: BaseAlgorithm

    def _setup_model(self) -> None:
        self.leader_model._setup_model()
        self.follower_model._setup_model()

    def __init__(
            self,
            leader_algorithm: str,
            leader_policy: str | type[BasePolicy],
            follower_algorithm: str,
            follower_policy: str | type[BasePolicy],
            env: GymEnv | str | None,
            learning_rate: float | Schedule,

            leader_algorithm_kwargs: dict[str, Any] | None = None,
            leader_policy_kwargs: dict[str, Any] | None = None,
            follower_algorithm_kwargs: dict[str, Any] | None = None,
            follower_policy_kwargs: dict[str, Any] | None = None,

            _init_setup_model: bool = True,
            **kwargs,
    ):
        super().__init__(
            policy=leader_policy,
            env=env,
            learning_rate=learning_rate,
            **kwargs,
        )

        del self.policy
        del self.action_space

        self.state_split_point = 0
        self.leader_algorithm_kwargs = leader_algorithm_kwargs or {}
        self.leader_algorithm_kwargs.update(kwargs)
        self.leader_model = ALGOS[leader_algorithm](
            policy=leader_policy,
            env=env,
            learning_rate=learning_rate,
            policy_kwargs=leader_policy_kwargs,
            **leader_algorithm_kwargs,
        )

        # TODO: adjust the action space for the follower
        self.follower_algorithm_kwargs = follower_algorithm_kwargs or {}
        self.follower_algorithm_kwargs.update(kwargs)
        self.follower_model = ALGOS[follower_algorithm](
            policy=follower_policy,
            env=env,
            learning_rate=learning_rate,
            policy_kwargs=follower_policy_kwargs,
            **follower_algorithm_kwargs,
        )

        if _init_setup_model:
            self._setup_model()

    def learn(
            self: SelfLeaderFollowerAlgorithm,
            total_timesteps: int,
            callback: MaybeCallback = None,
            log_interval: int = 100,
            tb_log_name: str = "run",
            reset_num_timesteps: bool = True,
            progress_bar: bool = False,
    ) -> SelfLeaderFollowerAlgorithm:
        # TODO: learn simultaneously?
        self.leader_model.learn(
            total_timesteps=total_timesteps,
            callback=callback,
            log_interval=log_interval,
            tb_log_name=tb_log_name,
            reset_num_timesteps=reset_num_timesteps,
            progress_bar=progress_bar,
        )

        self.follower_model.learn(
            total_timesteps=total_timesteps,
            callback=callback,
            log_interval=log_interval,
            tb_log_name=tb_log_name,
            reset_num_timesteps=reset_num_timesteps,
            progress_bar=progress_bar,
        )

        return self

    def predict(
            self,
            observation: np.ndarray | dict[str, np.ndarray],
            state: tuple[np.ndarray, ...] = None,
            episode_start: np.ndarray = None,
            deterministic: bool = False,
    ) -> tuple[np.ndarray, tuple[np.ndarray, ...] | None]:
        leader_state, follower_state = self.get_model_states(state)

        leader_action, leader_state = self.leader_model.predict(observation, leader_state, episode_start, deterministic)

        follower_obs = LavaEnv.prepare_follower_obs(observation, leader_action)
        follower_action, follower_state = self.follower_model.predict(follower_obs, follower_state, episode_start,
                                                                      deterministic)

        action = self.combine_model_actions(leader_action, follower_action)

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

    def combine_model_actions(self, leader_action: np.ndarray, follower_action: np.ndarray) -> np.ndarray:
        # TODO: make this customisable
        return leader_action

    def save(
        self,
        path: Union[str, pathlib.Path, io.BufferedIOBase],
        exclude: Optional[Iterable[str]] = None,
        include: Optional[Iterable[str]] = None,
    ) -> None:
        self.leader_model.save(os.path.join(path, 'leader'), exclude, include)
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
        raise NotImplementedError()

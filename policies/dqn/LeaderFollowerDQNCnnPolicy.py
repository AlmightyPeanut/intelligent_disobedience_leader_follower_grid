import torch as th

from typing import Any

from gymnasium import spaces
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.type_aliases import Schedule, PyTorchObs
from stable_baselines3.dqn.policies import DQNPolicy, QNetwork
from torch import nn

from minigrid_env.environment import Actions
from policies.dqn.LavaEnvFeaturesExtractor import LavaEnvCNNFeaturesExtractor


class LeaderFollowerDQNCnnPolicy(DQNPolicy):
    """
    Policy class for DQN when using images as input.

    :param observation_space: Observation space
    :param action_space: Action space
    :param lr_schedule: Learning rate schedule (could be constant)
    :param net_arch: The specification of the policy and value networks.
    :param activation_fn: Activation function
    :param features_extractor_class: Features extractor to use.
    :param normalize_images: Whether to normalize images or not,
         dividing by 255.0 (True by default)
    :param optimizer_class: The optimizer to use,
        ``th.optim.Adam`` by default
    :param optimizer_kwargs: Additional keyword arguments,
        excluding the learning rate, to pass to the optimizer
    """

    follower_q_net: QNetwork
    follower_q_net_target: QNetwork

    def __init__(
            self,
            observation_space: spaces.Space,
            action_space: spaces.Discrete,
            lr_schedule: Schedule,
            net_arch: list[int] | None = None,
            activation_fn: type[nn.Module] = nn.ReLU,
            features_extractor_class: type[BaseFeaturesExtractor] = LavaEnvCNNFeaturesExtractor,
            features_extractor_kwargs: dict[str, Any] | None = None,
            normalize_images: bool = True,
            optimizer_class: type[th.optim.Optimizer] = th.optim.Adam,
            optimizer_kwargs: dict[str, Any] | None = None,
    ) -> None:
        # The leader does not have the null action but should always decide to move somewhere
        action_space.n -= 1
        super().__init__(
            observation_space,
            action_space,
            lr_schedule,
            net_arch,
            activation_fn,
            features_extractor_class,
            features_extractor_kwargs,
            normalize_images,
            optimizer_class,
            optimizer_kwargs,
        )

    def _build(self, lr_schedule: Schedule) -> None:
        super()._build(lr_schedule)
        self.follower_q_net = self.make_follower_q_net()
        self.follower_q_net_target = self.make_follower_q_net()
        self.follower_q_net_target.load_state_dict(self.follower_q_net.state_dict())
        self.follower_q_net_target.set_training_mode(False)

        # Setup optimizer with initial learning rate
        self.follower_optimizer = self.optimizer_class(  # type: ignore[call-arg]
            self.follower_q_net.parameters(),
            lr=lr_schedule(1),
            **self.optimizer_kwargs,
        )

    def make_follower_q_net(self) -> QNetwork:
        # actions are encoded in separate input channels as fully 0s or 1s
        net_args = {
            "action_space": spaces.Discrete(2),  # safe or unsafe
            "observation_space": spaces.Box(
                low=self.net_args["observation_space"].low.min(),
                high=self.net_args["observation_space"].high.max(),
                shape=(
                    self.net_args["observation_space"].shape[0] + self.net_args["action_space"].n,
                    self.net_args["observation_space"].shape[1],
                    self.net_args["observation_space"].shape[2]),
            ),
        }

        # Make sure we always have separate networks for features extractors etc
        features_extractor = self.features_extractor_class(net_args["observation_space"],
                                                           **self.features_extractor_kwargs)
        net_args = self._update_features_extractor(net_args, features_extractor=features_extractor)
        return QNetwork(**net_args).to(self.device)

    def _predict(self, obs: PyTorchObs, deterministic: bool = True) -> th.Tensor:
        leader_action = self.q_net._predict(obs, deterministic=deterministic)

        # TODO: make the follower learn with their own optimizer

        follower_obs = self.prepare_follower_obs(obs, leader_action)
        follower_prediction = self.follower_q_net._predict(follower_obs, deterministic)

        # simulate follower
        # TODO: make it work for multiple samples
        if follower_prediction.item() == 1:
            leader_action = th.tensor([Actions.pickup])

        # TODO might need to return both predictions
        return leader_action

    def prepare_follower_obs(self, obs: PyTorchObs, leader_action: th.Tensor) -> PyTorchObs:
        leader_action_obs = th.zeros((obs.shape[0],
                                      self.action_space.n,
                                      obs.shape[2],
                                      obs.shape[3]), )
        leader_action_obs[:, leader_action, ...] = 1
        follower_obs = th.cat([obs, leader_action_obs], dim=1)

        return follower_obs
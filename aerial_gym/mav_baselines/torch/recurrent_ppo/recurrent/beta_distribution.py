from stable_baselines3.common.distributions import Distribution
from stable_baselines3.common.distributions import sum_independent_dims
from stable_baselines3.common.distributions import get_action_dim
import torch as th
import torch.nn.functional as F
import gym
import gym.spaces as spaces
from typing import Any, Dict, List, Optional, Tuple, Type, Union

class BetaDistribution(Distribution):
    """
    Beta distribution.
    :param alpha: (Tensor) alpha parameter of the Beta distribution
    :param beta: (Tensor) beta parameter of the Beta distribution
    """

    def __init__(self, action_dim: int):
        super(BetaDistribution, self).__init__()
        self.action_dim = action_dim
        self.alpha = None
        self.beta = None

    def proba_distribution_net(self, latent_dim: int):
        """
        Create the layer that represents the distribution.
        :param latent_dim: (int) Dimension of the last layer
            of the policy network (before the action layer)
        :return: (nn.Module)
        """
        action_logits = th.nn.Sequential(
            th.nn.Linear(latent_dim, self.action_dim * 2),
            th.nn.Softplus(),
        )

        return action_logits

    def proba_distribution(self, action_logits: th.Tensor):
        """
        Create a distribution given the action logits (before squashing).
        :param action_logits: (th.Tensor) The logits value
            for each action
        :return: (Distribution)
        """
        self.alpha, self.beta = th.split(action_logits, self.action_dim, dim=1)
        self.distribution = th.distributions.Beta(self.alpha, self.beta)
        return self

    def mode(self) -> th.Tensor:
        return self.alpha / (self.alpha + self.beta)

    def sample(self) -> th.Tensor:
        return self.distribution.sample()

    def log_prob(self, action: th.Tensor) -> th.Tensor:
        return sum_independent_dims(self.distribution.log_prob(action))

    def entropy(self) -> th.Tensor:
        return sum_independent_dims(self.distribution.entropy())

    def kl_divergence(self, other: "BetaDistribution") -> th.Tensor:
        return th.distributions.kl_divergence(
            th.distributions.Beta(self.alpha, self.beta),
            th.distributions.Beta(other.alpha, other.beta),
        )

    def get_std(self) -> th.Tensor:
        return th.sqrt(self.alpha * self.beta / ((self.alpha + self.beta) ** 2 * (self.alpha + self.beta + 1)))
    
    def actions_from_params(self, *args, **kwargs) -> th.Tensor:
        return super().actions_from_params(*args, **kwargs)
    
    def log_prob_from_params(self, *args, **kwargs) -> Tuple[th.Tensor, th.Tensor]:
        return super().log_prob_from_params(*args, **kwargs)

    # def entropy(self) -> th.Tensor:
    #     return th.distributions.Beta(self.alpha, self.beta).entropy()

    # def log_prob_from_latent(self, latent_pi: th.Tensor, actions: th.Tensor) -> th.Tensor:
    #     """
    #     Get the log probability of the action

def make_proba_distribution(
    action_space: gym.spaces.Space
) -> Distribution:
    if isinstance(action_space, spaces.Box):
        return BetaDistribution(get_action_dim(action_space))

from typing import Any, Dict, List, Optional, Tuple, Type, Union
import time
import numpy as np
import torch as th
import warnings
from gymnasium import spaces
from functools import partial
from stable_baselines3.common.distributions import Distribution
from stable_baselines3.common.policies import ActorCriticPolicy, BasePolicy
from stable_baselines3.common.preprocessing import preprocess_obs

from stable_baselines3.common.distributions import (
    BernoulliDistribution,
    CategoricalDistribution,
    DiagGaussianDistribution,
    Distribution,
    MultiCategoricalDistribution,
    StateDependentNoiseDistribution,
)

from stable_baselines3.common.torch_layers import (
    BaseFeaturesExtractor,
    CombinedExtractor,
    FlattenExtractor,
    MlpExtractor,
    NatureCNN,
)
from stable_baselines3.common.type_aliases import Schedule
from stable_baselines3.common.utils import zip_strict
from torch import nn

from aerial_gym.mav_baselines.torch.recurrent_ppo.recurrent.type_aliases import RNNStates
from aerial_gym.mav_baselines.torch.recurrent_ppo.recurrent.rnn_extractor import Encoder320, Decoder320, EncoderResnet, DecoderResnet
from aerial_gym.mav_baselines.torch.recurrent_ppo.recurrent.beta_distribution import BetaDistribution, make_proba_distribution
from aerial_gym.mav_baselines.torch.controlNet.ldm.modules.distributions.distributions import DiagonalGaussianDistribution

class RecurrentActorCriticPolicy(ActorCriticPolicy):
    """
    Recurrent policy class for actor-critic algorithms (has both policy and value prediction).
    To be used with A2C, PPO and the likes.
    It assumes that both the actor and the critic LSTM
    have the same architecture.

    :param observation_space: Observation space
    :param action_space: Action space
    :param lr_schedule: Learning rate schedule (could be constant)
    :param net_arch: The specification of the policy and value networks.
    :param activation_fn: Activation function
    :param ortho_init: Whether to use or not orthogonal initialization
    :param use_sde: Whether to use State Dependent Exploration or not
    :param log_std_init: Initial value for the log standard deviation
    :param full_std: Whether to use (n_features x n_actions) parameters
        for the std instead of only (n_features,) when using gSDE
    :param use_expln: Use ``expln()`` function instead of ``exp()`` to ensure
        a positive standard deviation (cf paper). It allows to keep variance
        above zero and prevent it from growing too fast. In practice, ``exp()`` is usually enough.
    :param squash_output: Whether to squash the output using a tanh function,
        this allows to ensure boundaries when using gSDE.
    :param features_extractor_class: Features extractor to use.
    :param features_extractor_kwargs: Keyword arguments
        to pass to the features extractor.
    :param share_features_extractor: If True, the features extractor is shared between the policy and value networks.
    :param normalize_images: Whether to normalize images or not,
         dividing by 255.0 (True by default)
    :param optimizer_class: The optimizer to use,
        ``th.optim.Adam`` by default
    :param optimizer_kwargs: Additional keyword arguments,
        excluding the learning rate, to pass to the optimizer
    :param lstm_hidden_size: Number of hidden units for each LSTM layer.
    :param n_lstm_layers: Number of LSTM layers.
    :param shared_lstm: Whether the LSTM is shared between the actor and the critic
        (in that case, only the actor gradient is used)
        By default, the actor and the critic have two separate LSTM.
    :param enable_critic_lstm: Use a seperate LSTM for the critic.
    :param lstm_kwargs: Additional keyword arguments to pass the the LSTM
        constructor.
    """

    def __init__(
        self,
        observation_space: spaces.Space,
        action_space: spaces.Space,
        lr_schedule: Schedule,
        net_arch: Optional[Union[List[int], Dict[str, List[int]]]] = None,
        activation_fn: Type[nn.Module] = nn.Tanh,
        ortho_init: bool = True,
        use_sde: bool = False,
        log_std_init: float = 0.0,
        full_std: bool = True,
        use_expln: bool = False,
        squash_output: bool = False,
        features_extractor_class: Type[BaseFeaturesExtractor] = FlattenExtractor,
        features_extractor_kwargs: Optional[Dict[str, Any]] = None,
        share_features_extractor: bool = True,
        normalize_images: bool = True,
        optimizer_class: Type[th.optim.Optimizer] = th.optim.Adam,
        optimizer_kwargs: Optional[Dict[str, Any]] = None,
        lstm_hidden_size: int = 256,
        n_lstm_layers: int = 1,
        shared_lstm: bool = False,
        enable_critic_lstm: bool = True,
        states_dim: int = 0,
        features_dim: int = 32,
        only_lstm_training: bool = False,
        use_beta: bool = False,
        reconstruction_members: Optional[List[bool]] = None,
        reconstruction_steps: int = 2,
        use_kl_loss: bool = False,
        lstm_kwargs: Optional[Dict[str, Any]] = None,
    ):
        self.lstm_output_dim = lstm_hidden_size
        self.use_beta = use_beta
        super().__init__(
            observation_space,
            action_space,
            lr_schedule,
            net_arch,
            activation_fn,
            ortho_init,
            use_sde,
            log_std_init,
            full_std,
            use_expln,
            squash_output,
            features_extractor_class,
            # {},
            features_extractor_kwargs,
            share_features_extractor,
            normalize_images,
            optimizer_class,
            optimizer_kwargs,
        )
        self.states_dim = states_dim
        self.features_dim = features_dim
        self.only_lstm_training = only_lstm_training
        self.share_features_extractor = share_features_extractor
        self.reconstruction_members = reconstruction_members
        self.reconstruction_steps = reconstruction_steps
        self.use_kl_loss = use_kl_loss
        # if self.share_features_extractor:
        #     self.vf_features_extractor = self.features_extractor
        # else:
        #     self.vf_features_extractor = features_extractor_class(self.observation_space, **self.features_extractor_kwargs)
        self.lstm_kwargs = lstm_kwargs or {}
        self.shared_lstm = shared_lstm
        self.enable_critic_lstm = enable_critic_lstm
        if not self.use_kl_loss:
            self.lstm_actor = nn.LSTM(
                self.features_dim + states_dim,
                lstm_hidden_size,
                num_layers=n_lstm_layers,
                **self.lstm_kwargs,
            )
            self.mu_linear = nn.Linear(lstm_hidden_size, 3 * self.features_dim)
        else:
            self.lstm_mean = nn.LSTM(
                self.features_dim + states_dim,
                lstm_hidden_size,
                num_layers=n_lstm_layers,
                **self.lstm_kwargs,
            )
            self.lstm_std = nn.LSTM(
                self.features_dim + states_dim,
                lstm_hidden_size,
                num_layers=n_lstm_layers,
                **self.lstm_kwargs,
            )
            self.mu_linear = nn.Linear(lstm_hidden_size, 3 * self.features_dim)
            self.logvar_linear = nn.Linear(lstm_hidden_size, 3 * self.features_dim)
        # For the predict() method, to initialize hidden states
        # (n_lstm_layers, batch_size, lstm_hidden_size)
        self.lstm_hidden_state_shape = (n_lstm_layers, 1, lstm_hidden_size)
        self.critic = None
        self.lstm_critic = None
        assert not (
            self.shared_lstm and self.enable_critic_lstm
        ), "You must choose between shared LSTM, seperate or no LSTM for the critic."

        assert not (
            self.shared_lstm and not self.share_features_extractor
        ), "If the features extractor is not shared, the LSTM cannot be shared."

        # No LSTM for the critic, we still need to convert
        # output of features extractor to the correct size
        # (size of the output of the actor lstm)
        if not (self.shared_lstm or self.enable_critic_lstm):
            self.critic = nn.Linear(self.features_dim, lstm_hidden_size)
        # Use a separate LSTM for the critic
        if self.enable_critic_lstm:
            self.lstm_critic = nn.LSTM(
                self.features_dim + states_dim,
                lstm_hidden_size,
                num_layers=n_lstm_layers,
                **self.lstm_kwargs,
            )
        if features_extractor_kwargs is None:
            self.feature_decoder0 = Decoder320(self.observation_space, self.features_dim)
        else:
            self.feature_decoder0 = DecoderResnet(features_extractor_kwargs['ddconfig'])
        # self.feature_decoder1 = Decoder(self.observation_space, self.features_dim + states_dim)

        # Setup optimizer with initial learning rate
        self.optimizer = self.optimizer_class(self.parameters(), lr=lr_schedule(1), **self.optimizer_kwargs)

    def _build(self, lr_schedule: Schedule) -> None:
        """
        Create the networks and the optimizer.

        :param lr_schedule: Learning rate schedule
            lr_schedule(1) is the initial learning rate
        """
        self._build_mlp_extractor()
        if self.use_beta:
            self.action_dist = make_proba_distribution(self.action_space)

        latent_dim_pi = self.mlp_extractor.latent_dim_pi

        if isinstance(self.action_dist, DiagGaussianDistribution):
            self.action_net, self.log_std = self.action_dist.proba_distribution_net(
                latent_dim=latent_dim_pi, log_std_init=self.log_std_init
            )
            # print("self.log_std: ", self.log_std)
            # self.log_std = th.tensor(0.0, dtype=th.float32, device=self.device)
        elif isinstance(self.action_dist, StateDependentNoiseDistribution):
            self.action_net, self.log_std = self.action_dist.proba_distribution_net(
                latent_dim=latent_dim_pi, latent_sde_dim=latent_dim_pi, log_std_init=self.log_std_init
            )
        elif isinstance(self.action_dist, (CategoricalDistribution, MultiCategoricalDistribution, BernoulliDistribution)):
            self.action_net = self.action_dist.proba_distribution_net(latent_dim=latent_dim_pi)
        elif isinstance(self.action_dist, BetaDistribution):
            self.action_net = self.action_dist.proba_distribution_net(latent_dim=latent_dim_pi)
        else:
            raise NotImplementedError(f"Unsupported distribution '{self.action_dist}'.")

        self.value_net = nn.Linear(self.mlp_extractor.latent_dim_vf, 1)
        # Init weights: use orthogonal initialization
        # with small initial weight for the output
        if self.ortho_init:
            # TODO: check for features_extractor
            # Values from stable-baselines.
            # features_extractor/mlp values are
            # originally from openai/baselines (default gains/init_scales).
            module_gains = {
                self.features_extractor: np.sqrt(2),
                self.mlp_extractor: np.sqrt(2),
                self.action_net: 0.01,
                self.value_net: 1,
            }
            for module, gain in module_gains.items():
                module.apply(partial(self.init_weights, gain=gain))

        # Setup optimizer with initial learning rate
        self.optimizer = self.optimizer_class(self.parameters(), lr=lr_schedule(1), **self.optimizer_kwargs)

    def _build_mlp_extractor(self) -> None:
        """
        Create the policy and value networks.
        Part of the layers can be shared.
        """
        self.mlp_extractor = MlpExtractor(
            self.lstm_output_dim + 14,
            net_arch=self.net_arch,
            activation_fn=self.activation_fn,
            device=self.device,
        )

    @staticmethod
    def _process_sequence(
        features: th.Tensor,
        lstm_states: Tuple[th.Tensor, th.Tensor],
        episode_starts: th.Tensor,
        lstm: nn.LSTM,
    ) -> Tuple[th.Tensor, th.Tensor]:
        """
        Do a forward pass in the LSTM network.
        :param features: Input tensor
        :param lstm_states: previous cell and hidden states of the LSTM
        :param episode_starts: Indicates when a new episode starts,
            in that case, we need to reset LSTM states.
        :param lstm: LSTM object.
        :return: LSTM output and updated LSTM states.
        """
        # LSTM logic
        # (sequence length, batch size, features dim)
        # (batch size = n_envs for data collection or n_seq when doing gradient update)
        n_seq = lstm_states[0].shape[1]
        # Batch to sequence
        # (padded batch size, features_dim) -> (n_seq, max length, features_dim) -> (max length, n_seq, features_dim)
        # note: max length (max sequence length) is always 1 during data collection
        # print("lstm_states[0]: ",lstm_states[0].shape)
        # print("features: ", features.shape)
        features_sequence = features.reshape((n_seq, -1, lstm.input_size)).swapaxes(0, 1)
        episode_starts = episode_starts.reshape((n_seq, -1)).swapaxes(0, 1)
        # If we don't have to reset the state in the middle of a sequence
        # we can avoid the for loop, which speeds up things
        if th.all(episode_starts == 0.0):
            lstm_output, lstm_states = lstm(features_sequence, lstm_states)
            lstm_output = th.flatten(lstm_output.transpose(0, 1), start_dim=0, end_dim=1)
            return lstm_output, lstm_states

        lstm_output = []
        # Iterate over the sequence
        for features, episode_start in zip_strict(features_sequence, episode_starts):
            hidden, lstm_states = lstm(
                features.unsqueeze(dim=0),
                (
                    # Reset the states at the beginning of a new episode
                    (1.0 - episode_start).view(1, n_seq, 1) * lstm_states[0],
                    (1.0 - episode_start).view(1, n_seq, 1) * lstm_states[1],
                ),
            )
            lstm_output += [hidden]
        # Sequence to batch
        # (sequence length, n_seq, lstm_out_dim) -> (batch_size, lstm_out_dim)
        lstm_output = th.flatten(th.cat(lstm_output).transpose(0, 1), start_dim=0, end_dim=1)
        return lstm_output, lstm_states
    
    def forward_rnn(
        self,
        obs: th.Tensor,
        lstm_states: Tuple[th.Tensor, th.Tensor],
        episode_starts: th.Tensor,
    ) -> Tuple[th.Tensor, Tuple[th.Tensor, th.Tensor]]:
        cat_pi = []
        cat_vf = []
        for key, _obs in obs.items():
            if key == 'image':
                features = self.extract_features(_obs)
                if self.share_features_extractor:
                    pi_features = vf_features = features
                else:
                    pi_features, vf_features = features
            else:
                state_shape = _obs.shape
                _obs = _obs.reshape([state_shape[0], state_shape[2]]).float()
                if self.states_dim > 0:
                    pi_features = th.cat([pi_features, _obs[:, :self.states_dim]], dim=1)
                    vf_features = th.cat([vf_features, _obs[:, :self.states_dim]], dim=1)
                latent_pi, lstm_states_pi = self._process_sequence(pi_features, lstm_states.pi,
                                            episode_starts, self.lstm_actor)
                if self.lstm_critic is not None:
                    latent_vf, lstm_states_vf = self._process_sequence(vf_features, lstm_states.vf,
                                            episode_starts, self.lstm_critic)
                elif self.shared_lstm:
                    latent_vf = latent_pi.detach()
                    lstm_states_vf = (lstm_states_pi[0].detach(), lstm_states_pi[1].detach())
                else:
                    latent_vf = self.critic(vf_features)
                    lstm_states_vf = lstm_states_pi
                cat_pi = [latent_pi, _obs]
                cat_vf = [latent_vf, _obs]
        latent_pi = th.cat(cat_pi, dim=1)
        latent_vf = th.cat(cat_vf, dim=1)
        return latent_pi, latent_vf, RNNStates(lstm_states_pi, lstm_states_vf)
    
    def forward_rnn_latent(
            self,
            obs: th.Tensor,
            lstm_states: Tuple[th.Tensor, th.Tensor],
            episode_starts: th.Tensor,
    ) -> Tuple[th.Tensor, Tuple[th.Tensor, th.Tensor]]:
        obs_depth = obs[:, 0, :(self.features_dim + self.states_dim)]
        latent_pi, lstm_states_pi = self._process_sequence(obs_depth, lstm_states.pi, episode_starts, self.lstm_actor)
        if self.lstm_critic is not None:
            latent_vf, lstm_states_vf = self._process_sequence(obs_depth, lstm_states.vf, episode_starts, self.lstm_critic)
        elif self.shared_lstm:
            latent_vf = latent_pi.detach()
            lstm_states_vf = (lstm_states_pi[0].detach(), lstm_states_pi[1].detach())
        else:
            latent_vf = self.critic(obs_depth)
            lstm_states_vf = lstm_states_pi
        latent_pi = th.cat([latent_pi , obs[:, 0, self.features_dim:]], dim=1)
        latent_vf = th.cat([latent_vf , obs[:, 0, self.features_dim:]], dim=1)
        return latent_pi, latent_vf, RNNStates(lstm_states_pi, lstm_states_vf)
    
    def forward_rnn_latent_kl_latent(
            self,
            obs: th.Tensor,
            lstm_states_means: Tuple[th.Tensor, th.Tensor],
            lstm_states_stds: Tuple[th.Tensor, th.Tensor],
            episode_starts: th.Tensor,
    ) -> Tuple[th.Tensor, Tuple[th.Tensor, th.Tensor]]:
        obs_mean = obs[:, 0, 0, :(self.features_dim + self.states_dim)]
        obs_std = obs[:, 0, 1, :(self.features_dim + self.states_dim)]
        latent_pi_mean, lstm_states_pi_mean = self._process_sequence(obs_mean, lstm_states_means.pi, episode_starts, self.lstm_mean)
        latent_pi_std, lstm_states_pi_std = self._process_sequence(obs_std, lstm_states_stds.pi, episode_starts, self.lstm_std)
        latent = th.cat([latent_pi_mean.unsqueeze(-2), latent_pi_std.unsqueeze(-2)], dim=-2)
        latent_pi = DiagonalGaussianDistribution(latent).mode().squeeze(-2)
        latent_vf = latent_pi.detach()
        lstm_states_vf_mean = (lstm_states_pi_mean[0].detach(), lstm_states_pi_mean[1].detach())
        lstm_states_vf_std = (lstm_states_pi_std[0].detach(), lstm_states_pi_std[1].detach())
        latent_pi = th.cat([latent_pi , obs[:, 0, 0, self.features_dim:]], dim=1)
        latent_vf = th.cat([latent_vf , obs[:, 0, 0, self.features_dim:]], dim=1)
        return latent_pi, latent_vf, RNNStates(lstm_states_pi_mean, lstm_states_vf_mean), RNNStates(lstm_states_pi_std, lstm_states_vf_std)
    
    def forward(self, latent_pi: th.Tensor, latent_vf: th.Tensor,deterministic: bool = False) -> Tuple[th.Tensor, th.Tensor, th.Tensor]:
        latent_pi_ = self.mlp_extractor.forward_actor(latent_pi)
        latent_vf_ = self.mlp_extractor.forward_critic(latent_vf)
        # Evaluate the values for the given observations
        values = self.value_net(latent_vf_)
        distribution = self._get_action_dist_from_latent(latent_pi_)
        actions = distribution.get_actions(deterministic=deterministic)
        log_prob = distribution.log_prob(actions)
        return actions, values, log_prob

    def forward_rnn_cmaes(
        self,
        obs: th.Tensor,
        lstm_states: Tuple[th.Tensor, th.Tensor],
        episode_starts: th.Tensor,
    ) -> Tuple[th.Tensor, Tuple[th.Tensor, th.Tensor]]:
        cat_pi = []
        for key, _obs in obs.items():
            if key == 'image':
                features = self.extract_features(_obs)
            else:
                state_shape = _obs.shape
                _obs = _obs.reshape([state_shape[0], state_shape[2]]).float()
                if self.states_dim > 0:
                    features = th.cat([features, _obs[:, :self.states_dim]], dim=1)
                latent_pi, lstm_states_pi = self._process_sequence(features, lstm_states,
                                            episode_starts, self.lstm_actor)

                cat_pi = [latent_pi, _obs]
        latent_pi = th.cat(cat_pi, dim=1)
        return latent_pi, lstm_states_pi
    
    def _get_action_dist_from_latent(self, latent_pi: th.Tensor) -> Distribution:
        """
        Retrieve action distribution given the latent codes.

        :param latent_pi: Latent code for the actor
        :return: Action distribution
        """
        mean_actions = self.action_net(latent_pi)
        if isinstance(self.action_dist, DiagGaussianDistribution):
            return self.action_dist.proba_distribution(mean_actions, self.log_std)
        elif isinstance(self.action_dist, CategoricalDistribution):
            # Here mean_actions are the logits before the softmax
            return self.action_dist.proba_distribution(action_logits=mean_actions)
        elif isinstance(self.action_dist, MultiCategoricalDistribution):
            # Here mean_actions are the flattened logits
            return self.action_dist.proba_distribution(action_logits=mean_actions)
        elif isinstance(self.action_dist, BernoulliDistribution):
            # Here mean_actions are the logits (before rounding to get the binary actions)
            return self.action_dist.proba_distribution(action_logits=mean_actions)
        elif isinstance(self.action_dist, StateDependentNoiseDistribution):
            return self.action_dist.proba_distribution(mean_actions, self.log_std, latent_pi)
        elif isinstance(self.action_dist, BetaDistribution):
            return self.action_dist.proba_distribution(action_logits=(mean_actions+1.0))
        else:
            raise ValueError("Invalid action distribution")

    def get_distribution(
        self,
        obs: th.Tensor,
        lstm_states: Tuple[th.Tensor, th.Tensor],
        episode_starts: th.Tensor,
    ) -> Tuple[Distribution, Tuple[th.Tensor, ...]]:
        """
        Get the current policy distribution given the observations.

        :param obs: Observation.
        :param lstm_states: The last hidden and memory states for the LSTM.
        :param episode_starts: Whether the observations correspond to new episodes
            or not (we reset the lstm states in that case).
        :return: the action distribution and new hidden states.
        """
        # Call the method from the parent of the parent class
        # latent_pi, lstm_states = self._process_sequence(features, lstm_states, episode_starts, self.lstm_actor)

        cat_pi = []
        for key, _obs in obs.items():
            if key == 'image':
                features = self.extract_features(_obs)
            else:
                state_shape = _obs.shape
                _obs = _obs.reshape([state_shape[0], state_shape[2]]).float()
                if self.states_dim > 0:
                    features = th.cat([features, _obs[:, :self.states_dim]], dim=1)
                latent_pi, lstm_states = self._process_sequence(features, lstm_states, episode_starts, self.lstm_actor)
                cat_pi = [latent_pi, _obs]
        latent_pi = th.cat(cat_pi, dim=1)
        latent_pi = self.mlp_extractor.forward_actor(latent_pi)
        return self._get_action_dist_from_latent(latent_pi), lstm_states


    def extract_features(self, obs: th.Tensor, features_extractor: Optional[BaseFeaturesExtractor] = None) -> th.Tensor:
        """
        Preprocess the observation if needed and extract features.
         :param obs: The observation
         :param features_extractor: The features extractor to use. If it is set to None,
            the features extractor of the policy is used.
         :return: The features
        """
        if features_extractor is None:
            warnings.warn(
                (
                    "When calling extract_features(), you should explicitely pass a features_extractor as parameter. "
                    "This will be mandatory in Stable-Baselines v1.8.0"
                ),
                DeprecationWarning,
            )

        features_extractor = features_extractor or self.features_extractor
        assert features_extractor is not None, "No features extractor was set"
        observation_space = self.observation_space['image']
        preprocessed_obs = preprocess_obs(obs, observation_space, normalize_images=self.normalize_images)
        return features_extractor(preprocessed_obs)

    def predict_values(
        self,
        obs: th.Tensor,
        lstm_states: Tuple[th.Tensor, th.Tensor],
        episode_starts: th.Tensor,
    ) -> th.Tensor:
        """
        Get the estimated values according to the current policy given the observations.

        :param obs: Observation.
        :param lstm_states: The last hidden and memory states for the LSTM.
        :param episode_starts: Whether the observations correspond to new episodes
            or not (we reset the lstm states in that case).
        :return: the estimated values.
        """
        # Call the method from the parent of the parent class
        cat_vf = []
        for key, _obs in obs.items():
            if key == 'image':
                features = self.extract_features(_obs, self.features_extractor)
            else:
                state_shape = _obs.shape
                _obs = _obs.reshape([state_shape[0], state_shape[2]]).float()
                if self.states_dim > 0:
                    features = th.cat([features, _obs[:, :self.states_dim]], dim=1)
                if self.lstm_critic is not None:
                    latent_vf, _ = self._process_sequence(features, lstm_states, episode_starts, self.lstm_critic)
                elif self.shared_lstm:
                    latent_pi, _ = self._process_sequence(features, lstm_states, episode_starts, self.lstm_actor)
                    latent_vf = latent_pi.detach()
                else:
                    latent_vf = self.critic(features)
                cat_vf = [latent_vf, _obs]
        latent_vf = th.cat(cat_vf, dim=1)

        latent_vf = self.mlp_extractor.forward_critic(latent_vf)
        return self.value_net(latent_vf)
    
    def predict_values_latent(
        self,
        obs: th.Tensor,
        lstm_states: Tuple[th.Tensor, th.Tensor],
        episode_starts: th.Tensor,
    ) -> th.Tensor:
        """
        Get the estimated values according to the current policy given the observations.

        :param obs: Observation.
        :param lstm_states: The last hidden and memory states for the LSTM.
        :param episode_starts: Whether the observations correspond to new episodes
            or not (we reset the lstm states in that case).
        :return: the estimated values.
        """
        # Call the method from the parent of the parent class
        obs_depth = obs[:, 0, :(self.features_dim + self.states_dim)]
        if self.lstm_critic is not None:
            latent_vf, _ = self._process_sequence(obs_depth, lstm_states, episode_starts, self.lstm_critic)
        elif self.shared_lstm:
            latent_pi, _ = self._process_sequence(obs_depth, lstm_states, episode_starts, self.lstm_actor)
            latent_vf = latent_pi.detach()
        else:
            latent_vf = self.critic(obs_depth)
        latent_vf = th.cat([latent_vf , obs[:, 0, self.features_dim:]], dim=1)
        latent_vf = self.mlp_extractor.forward_critic(latent_vf)
        return self.value_net(latent_vf)
    
    def predict_values_latent_kl_latent(
        self,
        obs: th.Tensor,
        lstm_states_means: Tuple[th.Tensor, th.Tensor],
        lstm_states_stds: Tuple[th.Tensor, th.Tensor],
        episode_starts: th.Tensor,
    ) -> th.Tensor:
        """
        Get the estimated values according to the current policy given the observations.

        :param obs: Observation.
        :param lstm_states: The last hidden and memory states for the LSTM.
        :param episode_starts: Whether the observations correspond to new episodes
            or not (we reset the lstm states in that case).
        :return: the estimated values.
        """
        obs_mean = obs[:, 0, 0, :(self.features_dim + self.states_dim)]
        obs_std = obs[:, 0, 1, :(self.features_dim + self.states_dim)]
        latent_vf_mean, _ = self._process_sequence(obs_mean, lstm_states_means, episode_starts, self.lstm_mean)
        latent_vf_std, _ = self._process_sequence(obs_std, lstm_states_stds, episode_starts, self.lstm_std)
        latent = th.cat([latent_vf_mean.unsqueeze(-2), latent_vf_std.unsqueeze(-2)], dim=-2)
        latent_vf = DiagonalGaussianDistribution(latent).mode().squeeze(-2)
        latent_vf = th.cat([latent_vf , obs[:, 0, 0, self.features_dim:]], dim=1)
        latent_vf = self.mlp_extractor.forward_critic(latent_vf)
        return self.value_net(latent_vf)
    
    def evaluate_actions(self, latend_lstm_pi: th.Tensor, latend_lstm_vf: th.Tensor, actions: th.Tensor) -> Tuple[th.Tensor, th.Tensor, th.Tensor]:
        latent_pi = self.mlp_extractor.forward_actor(latend_lstm_pi)
        latent_vf = self.mlp_extractor.forward_critic(latend_lstm_vf)
        distribution = self._get_action_dist_from_latent(latent_pi)
        log_prob = distribution.log_prob(actions)
        # print("actions: ", actions)
        # print("log_prob: ", log_prob)
        values = self.value_net(latent_vf)
        return values, log_prob, distribution.entropy()

    def _predict(
        self,
        observation: th.Tensor,
        lstm_states: Tuple[th.Tensor, th.Tensor],
        episode_starts: th.Tensor,
        deterministic: bool = False,
    ) -> Tuple[th.Tensor, Tuple[th.Tensor, ...]]:
        """
        Get the action according to the policy for a given observation.

        :param observation:
        :param lstm_states: The last hidden and memory states for the LSTM.
        :param episode_starts: Whether the observations correspond to new episodes
            or not (we reset the lstm states in that case).
        :param deterministic: Whether to use stochastic or deterministic actions
        :return: Taken action according to the policy and hidden states of the RNN
        """
        distribution, lstm_states = self.get_distribution(observation, lstm_states, episode_starts)
        return distribution.get_actions(deterministic=deterministic), lstm_states

    def predict(
        self,
        observation: Union[np.ndarray, Dict[str, np.ndarray]],
        state: Optional[Tuple[np.ndarray, ...]] = None,
        episode_start: Optional[np.ndarray] = None,
        deterministic: bool = False,
    ) -> Tuple[np.ndarray, Optional[Tuple[np.ndarray, ...]]]:
        """
        Get the policy action from an observation (and optional hidden state).
        Includes sugar-coating to handle different observations (e.g. normalizing images).

        :param observation: the input observation
        :param lstm_states: The last hidden and memory states for the LSTM.
        :param episode_starts: Whether the observations correspond to new episodes
            or not (we reset the lstm states in that case).
        :param deterministic: Whether or not to return deterministic actions.
        :return: the model's action and the next hidden state
            (used in recurrent policies)
        """
        # Switch to eval mode (this affects batch norm / dropout)
        self.set_training_mode(False)

        observation, vectorized_env = self.obs_to_tensor(observation)

        if isinstance(observation, dict):
            n_envs = observation[list(observation.keys())[0]].shape[0]
        else:
            n_envs = observation.shape[0]
        # state : (n_layers, n_envs, dim)
        if state is None:
            # Initialize hidden states to zeros
            state = np.concatenate([np.zeros(self.lstm_hidden_state_shape) for _ in range(n_envs)], axis=1)
            state = (state, state)

        if episode_start is None:
            episode_start = np.array([False for _ in range(n_envs)])

        with th.no_grad():
            # Convert to PyTorch tensors
            states = th.tensor(state[0], dtype=th.float32, device=self.device), th.tensor(
                state[1], dtype=th.float32, device=self.device
            )
            episode_starts = th.tensor(episode_start, dtype=th.float32, device=self.device)
            actions, states = self._predict(
                observation, lstm_states=states, episode_starts=episode_starts, deterministic=deterministic
            )
            states = (states[0].cpu().numpy(), states[1].cpu().numpy())

        # Convert to numpy
        actions = actions.cpu().numpy()
        # print(actions)
        if isinstance(self.action_space, spaces.Box):
            if self.squash_output:
                # Rescale to proper domain when using squashing
                actions = self.unscale_action(actions)
            else:
                # Actions could be on arbitrary scale, so clip the actions to avoid
                # out of bound error (e.g. if sampling from a Gaussian distribution)
                actions = np.clip(actions, self.action_space.low, self.action_space.high)

        # Remove batch dimension if needed
        if not vectorized_env:
            actions = actions.squeeze(axis=0)

        return actions, states
            
    def to_latent(self, obs):
        with th.no_grad():
            if isinstance(obs, dict):
                obs_mu = self.extract_features(obs['image'])
                state_shape = obs['state'].shape
                obs_state = obs['state'].reshape([state_shape[0], state_shape[2]]).float()
            # else:
            #     obs_mu = self.extract_features(obs)
        # latent_obs, latent_next_obs = [
        #     (x_mu + x_logsigma.exp() * th.randn_like(x_mu))
        #     for x_mu, x_logsigma in [(obs_mu, obs_logsigma), (next_obs_mu, next_obs_logsigma)]]
        # latent_obs = th.cat([obs_mu, obs['state'].squeeze().float()[:, 3:]], dim=1)
        if self.states_dim > 0:
            latent_obs = th.cat([obs_mu, obs_state[:, :self.states_dim]], dim=1)
        return latent_obs
            
    def predict_img(self,
        latent_obs: th.Tensor,):
        # Switch to eval mode (this affects batch norm / dropout)
        with th.no_grad():
            # latent_obs = th.tensor(latent_obs, dtype=th.float32, device=self.device)
            latent_obs = self.mu_linear(latent_obs)
            recon_latent_size = self.features_dim
            pre_latent_obs, cur_latent_obs, next_latent_obs = th.split(latent_obs, [recon_latent_size, recon_latent_size, recon_latent_size], dim=1)
            total_laten_obs = [pre_latent_obs, cur_latent_obs, next_latent_obs]
            reconstruction = []
            for i in range(len(self.reconstruction_members)):
                if self.reconstruction_members[i]:
                    reconstruction.append(self.feature_decoder0(total_laten_obs[i]).cpu().numpy())
                else:
                    reconstruction.append(None)
        return reconstruction

    def predict_lstm(self, 
        latent_obs: th.Tensor,
        lstm_states: Tuple[th.Tensor, th.Tensor],
        episode_starts: th.Tensor,
        is_eva: bool = False,
        ) -> Tuple[th.Tensor, th.Tensor, int, Tuple[th.Tensor, th.Tensor]]:

        pre_latent_obs, lstm_state = self._process_sequence(latent_obs, lstm_states, episode_starts, self.lstm_actor)
        n_seq = lstm_states[0].shape[1]
        pre_latent_obs = pre_latent_obs.reshape([n_seq, -1, self.lstm_output_dim])
        pre_latent_obs = self.mu_linear(pre_latent_obs)
        recon_latent_size = self.features_dim
        pre_latent_obs, cur_latent_obs, next_latent_obs = th.split(pre_latent_obs, [recon_latent_size, recon_latent_size, recon_latent_size], dim=2)
        # reconstruction0 = self.feature_decoder0(th.flatten(cur_latent_obs, start_dim=0, end_dim=1))
        # print("pre_latent_obs: ", pre_latent_obs.shape)
        total_laten_obs = [pre_latent_obs, cur_latent_obs, next_latent_obs]
        reconstruction = []
        if is_eva:
            for i in range(len(self.reconstruction_members)):
                if self.reconstruction_members[0]:
                    reconstruction.append(self.feature_decoder0(th.flatten(total_laten_obs[i], start_dim=0, end_dim=1)))
                else:
                    reconstruction.append(None)
                if self.reconstruction_members[1]:
                    reconstruction.append(self.feature_decoder1(th.flatten(total_laten_obs[i], start_dim=0, end_dim=1)))
                else:
                    reconstruction.append(None)
                if self.reconstruction_members[2]:
                    reconstruction.append(self.feature_decoder2(th.flatten(total_laten_obs[i], start_dim=0, end_dim=1)))
                else:
                    reconstruction.append(None)
        else:
            if self.reconstruction_members[0]:
                reconstruction.append(self.feature_decoder0(th.flatten(total_laten_obs[0][:, self.reconstruction_steps:, :], start_dim=0, end_dim=1)))
            else:
                reconstruction.append(None)
            if self.reconstruction_members[1]:
                reconstruction.append(self.feature_decoder0(th.flatten(total_laten_obs[1], start_dim=0, end_dim=1)))
            else:
                reconstruction.append(None)
            if self.reconstruction_members[2]:
                reconstruction.append(self.feature_decoder0(th.flatten(total_laten_obs[2][:, :-self.reconstruction_steps, :], start_dim=0, end_dim=1)))
            else:
                reconstruction.append(None)

        return reconstruction, n_seq, lstm_state


class RecurrentActorCriticCnnPolicy(RecurrentActorCriticPolicy):
    """
    CNN recurrent policy class for actor-critic algorithms (has both policy and value prediction).
    Used by A2C, PPO and the likes.

    :param observation_space: Observation space
    :param action_space: Action space
    :param lr_schedule: Learning rate schedule (could be constant)
    :param net_arch: The specification of the policy and value networks.
    :param activation_fn: Activation function
    :param ortho_init: Whether to use or not orthogonal initialization
    :param use_sde: Whether to use State Dependent Exploration or not
    :param log_std_init: Initial value for the log standard deviation
    :param full_std: Whether to use (n_features x n_actions) parameters
        for the std instead of only (n_features,) when using gSDE
    :param use_expln: Use ``expln()`` function instead of ``exp()`` to ensure
        a positive standard deviation (cf paper). It allows to keep variance
        above zero and prevent it from growing too fast. In practice, ``exp()`` is usually enough.
    :param squash_output: Whether to squash the output using a tanh function,
        this allows to ensure boundaries when using gSDE.
    :param features_extractor_class: Features extractor to use.
    :param features_extractor_kwargs: Keyword arguments
        to pass to the features extractor.
    :param share_features_extractor: If True, the features extractor is shared between the policy and value networks.
    :param normalize_images: Whether to normalize images or not,
         dividing by 255.0 (True by default)
    :param optimizer_class: The optimizer to use,
        ``th.optim.Adam`` by default
    :param optimizer_kwargs: Additional keyword arguments,
        excluding the learning rate, to pass to the optimizer
    :param lstm_hidden_size: Number of hidden units for each LSTM layer.
    :param n_lstm_layers: Number of LSTM layers.
    :param shared_lstm: Whether the LSTM is shared between the actor and the critic.
        By default, only the actor has a recurrent network.
    :param enable_critic_lstm: Use a seperate LSTM for the critic.
    :param lstm_kwargs: Additional keyword arguments to pass the the LSTM
        constructor.
    """

    def __init__(
        self,
        observation_space: spaces.Space,
        action_space: spaces.Space,
        lr_schedule: Schedule,
        net_arch: Optional[Union[List[int], Dict[str, List[int]]]] = None,
        activation_fn: Type[nn.Module] = nn.Tanh,
        ortho_init: bool = True,
        use_sde: bool = False,
        log_std_init: float = 0.0,
        full_std: bool = True,
        use_expln: bool = False,
        squash_output: bool = False,
        features_extractor_class: Type[BaseFeaturesExtractor] = NatureCNN,
        features_extractor_kwargs: Optional[Dict[str, Any]] = None,
        share_features_extractor: bool = True,
        normalize_images: bool = True,
        optimizer_class: Type[th.optim.Optimizer] = th.optim.Adam,
        optimizer_kwargs: Optional[Dict[str, Any]] = None,
        lstm_hidden_size: int = 256,
        n_lstm_layers: int = 1,
        shared_lstm: bool = False,
        enable_critic_lstm: bool = True,
        states_dim: int = 0,
        features_dim: int = 32,
        only_lstm_training: bool = False,
        use_beta: bool = False,
        reconstruction_members: Optional[List[bool]] = None,
        reconstruction_steps: int = 2,
        use_kl_loss: bool = False,
        lstm_kwargs: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(
            observation_space,
            action_space,
            lr_schedule,
            net_arch,
            activation_fn,
            ortho_init,
            use_sde,
            log_std_init,
            full_std,
            use_expln,
            squash_output,
            features_extractor_class,
            features_extractor_kwargs,
            share_features_extractor,
            normalize_images,
            optimizer_class,
            optimizer_kwargs,
            lstm_hidden_size,
            n_lstm_layers,
            shared_lstm,
            enable_critic_lstm,
            states_dim,
            features_dim,
            only_lstm_training,
            use_beta,
            reconstruction_members,
            reconstruction_steps,
            use_kl_loss,
            lstm_kwargs,
        )


class RecurrentMultiInputActorCriticPolicy(RecurrentActorCriticPolicy):
    """
    MultiInputActorClass policy class for actor-critic algorithms (has both policy and value prediction).
    Used by A2C, PPO and the likes.

    :param observation_space: Observation space
    :param action_space: Action space
    :param lr_schedule: Learning rate schedule (could be constant)
    :param net_arch: The specification of the policy and value networks.
    :param activation_fn: Activation function
    :param ortho_init: Whether to use or not orthogonal initialization
    :param use_sde: Whether to use State Dependent Exploration or not
    :param log_std_init: Initial value for the log standard deviation
    :param full_std: Whether to use (n_features x n_actions) parameters
        for the std instead of only (n_features,) when using gSDE
    :param use_expln: Use ``expln()`` function instead of ``exp()`` to ensure
        a positive standard deviation (cf paper). It allows to keep variance
        above zero and prevent it from growing too fast. In practice, ``exp()`` is usually enough.
    :param squash_output: Whether to squash the output using a tanh function,
        this allows to ensure boundaries when using gSDE.
    :param features_extractor_class: Features extractor to use.
    :param features_extractor_kwargs: Keyword arguments
        to pass to the features extractor.
    :param share_features_extractor: If True, the features extractor is shared between the policy and value networks.
    :param normalize_images: Whether to normalize images or not,
         dividing by 255.0 (True by default)
    :param optimizer_class: The optimizer to use,
        ``th.optim.Adam`` by default
    :param optimizer_kwargs: Additional keyword arguments,
        excluding the learning rate, to pass to the optimizer
    :param lstm_hidden_size: Number of hidden units for each LSTM layer.
    :param n_lstm_layers: Number of LSTM layers.
    :param shared_lstm: Whether the LSTM is shared between the actor and the critic.
        By default, only the actor has a recurrent network.
    :param enable_critic_lstm: Use a seperate LSTM for the critic.
    :param lstm_kwargs: Additional keyword arguments to pass the the LSTM
        constructor.
    """

    def __init__(
        self,
        observation_space: spaces.Space,
        action_space: spaces.Space,
        lr_schedule: Schedule,
        net_arch: Optional[Union[List[int], Dict[str, List[int]]]] = None,
        activation_fn: Type[nn.Module] = nn.Tanh,
        ortho_init: bool = True,
        use_sde: bool = False,
        log_std_init: float = 0.0,
        full_std: bool = True,
        use_expln: bool = False,
        squash_output: bool = False,
        features_extractor_class: Type[BaseFeaturesExtractor] = Encoder320,
        features_extractor_kwargs: Optional[Dict[str, Any]] = None,
        share_features_extractor: bool = True,
        normalize_images: bool = True,
        optimizer_class: Type[th.optim.Optimizer] = th.optim.Adam,
        optimizer_kwargs: Optional[Dict[str, Any]] = None,
        lstm_hidden_size: int = 256,
        n_lstm_layers: int = 1,
        shared_lstm: bool = False,
        enable_critic_lstm: bool = True,
        states_dim: int = 0,
        features_dim: int = 32,
        only_lstm_training: bool = False,
        use_beta: bool = False,
        reconstruction_members: Optional[List[bool]] = None,
        reconstruction_steps: int = 2,
        use_kl_loss: bool = False,
        lstm_kwargs: Optional[Dict[str, Any]] = None,
    ):
        if features_extractor_kwargs is not None:
            features_extractor_class = EncoderResnet
        super().__init__(
            observation_space,
            action_space,
            lr_schedule,
            net_arch,
            activation_fn,
            ortho_init,
            use_sde,
            log_std_init,
            full_std,
            use_expln,
            squash_output,
            features_extractor_class,
            features_extractor_kwargs,
            share_features_extractor,
            normalize_images,
            optimizer_class,
            optimizer_kwargs,
            lstm_hidden_size,
            n_lstm_layers,
            shared_lstm,
            enable_critic_lstm,
            states_dim,
            features_dim,
            only_lstm_training,
            use_beta,
            reconstruction_members,
            reconstruction_steps,
            use_kl_loss,
            lstm_kwargs,
        )

_policy_registry = dict()  # type: Dict[Type[BasePolicy], Dict[str, Type[BasePolicy]]]


def get_policy_from_name(base_policy_type: Type[BasePolicy], name: str) -> Type[BasePolicy]:
    """
    Returns the registered policy from the base type and name.
    See `register_policy` for registering policies and explanation.

    :param base_policy_type: the base policy class
    :param name: the policy name
    :return: the policy
    """
    if base_policy_type not in _policy_registry:
        raise KeyError(f"Error: the policy type {base_policy_type} is not registered!")
    if name not in _policy_registry[base_policy_type]:
        raise KeyError(
            f"Error: unknown policy type {name},"
            f"the only registed policy type are: {list(_policy_registry[base_policy_type].keys())}!"
        )
    return _policy_registry[base_policy_type][name]



def register_policy(name: str, policy: Type[BasePolicy]) -> None:
    """
    Register a policy, so it can be called using its name.
    e.g. SAC('MlpPolicy', ...) instead of SAC(MlpPolicy, ...).

    The goal here is to standardize policy naming, e.g.
    all algorithms can call upon "MlpPolicy" or "CnnPolicy",
    and they receive respective policies that work for them.
    Consider following:

    OnlinePolicy
    -- OnlineMlpPolicy ("MlpPolicy")
    -- OnlineCnnPolicy ("CnnPolicy")
    OfflinePolicy
    -- OfflineMlpPolicy ("MlpPolicy")
    -- OfflineCnnPolicy ("CnnPolicy")

    Two policies have name "MlpPolicy" and two have "CnnPolicy".
    In `get_policy_from_name`, the parent class (e.g. OnlinePolicy)
    is given and used to select and return the correct policy.

    :param name: the policy name
    :param policy: the policy class
    """
    sub_class = None
    for cls in BasePolicy.__subclasses__():
        if issubclass(policy, cls):
            sub_class = cls
            break
    if sub_class is None:
        raise ValueError(f"Error: the policy {policy} is not of any known subclasses of BasePolicy!")

    if sub_class not in _policy_registry:
        _policy_registry[sub_class] = {}
    if name in _policy_registry[sub_class]:
        # Check if the registered policy is same
        # we try to register. If not so,
        # do not override and complain.
        if _policy_registry[sub_class][name] != policy:
            raise ValueError(f"Error: the name {name} is already registered for a different policy, will not override.")
    _policy_registry[sub_class][name] = policy
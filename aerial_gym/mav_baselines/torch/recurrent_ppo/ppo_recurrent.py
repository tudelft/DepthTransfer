import sys
import os
import time
from copy import deepcopy
from typing import Any, Dict, Optional, Type, TypeVar, Union, Tuple, List
import numpy as np
import torch as th
from torch.cuda.amp import autocast
import torch.utils.data
from torch.nn import functional as F
from torchvision.utils import save_image
from gymnasium import spaces
from stable_baselines3.common.buffers import RolloutBuffer
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.on_policy_algorithm import OnPolicyAlgorithm
from stable_baselines3.common.policies import BasePolicy
from stable_baselines3.common.type_aliases import GymEnv, MaybeCallback, Schedule
from stable_baselines3.common.utils import explained_variance, get_schedule_fn, obs_as_tensor, safe_mean, configure_logger
from stable_baselines3.common.vec_env import VecEnv
from stable_baselines3.common import utils

from aerial_gym.mav_baselines.torch.recurrent_ppo.recurrent.buffers import RecurrentRolloutBuffer, LSTMThDictRolloutBuffer, LatentRolloutBuffer, DistLatentRolloutBuffer
from aerial_gym.mav_baselines.torch.recurrent_ppo.recurrent.policies import RecurrentActorCriticPolicy
from aerial_gym.mav_baselines.torch.recurrent_ppo.recurrent.type_aliases import RNNStates
from aerial_gym.mav_baselines.torch.recurrent_ppo.policies import CnnLstmPolicy, MlpLstmPolicy, MultiInputLstmPolicy
from aerial_gym.mav_baselines.torch.common.util import traj_rollout, lstm_rollout
SelfRecurrentPPO = TypeVar("SelfRecurrentPPO", bound="RecurrentPPO")
# from torch.utils.tensorboard.writer import SummaryWriter
# import threading
from aerial_gym.data.loaders import RolloutLSTMSequenceDataset#, RosbagSequenceDataset
import cv2

class RecurrentPPO(OnPolicyAlgorithm):
    """
    Proximal Policy Optimization algorithm (PPO) (clip version)
    with support for recurrent policies (LSTM).

    Based on the original Stable Baselines 3 implementation.

    Introduction to PPO: https://spinningup.openai.com/en/latest/algorithms/ppo.html

    :param policy: The policy model to use (MlpPolicy, CnnPolicy, ...)
    :param env: The environment to learn from (if registered in Gym, can be str)
    :param learning_rate: The learning rate, it can be a function
        of the current progress remaining (from 1 to 0)
    :param n_steps: The number of steps to run for each environment per update
        (i.e. batch size is n_steps * n_env where n_env is number of environment copies running in parallel)
    :param batch_size: Minibatch size
    :param n_epochs: Number of epoch when optimizing the surrogate loss
    :param gamma: Discount factor
    :param gae_lambda: Factor for trade-off of bias vs variance for Generalized Advantage Estimator
    :param clip_range: Clipping parameter, it can be a function of the current progress
        remaining (from 1 to 0).
    :param clip_range_vf: Clipping parameter for the value function,
        it can be a function of the current progress remaining (from 1 to 0).
        This is a parameter specific to the OpenAI implementation. If None is passed (default),
        no clipping will be done on the value function.
        IMPORTANT: this clipping depends on the reward scaling.
    :param normalize_advantage: Whether to normalize or not the advantage
    :param ent_coef: Entropy coefficient for the loss calculation
    :param vf_coef: Value function coefficient for the loss calculation
    :param max_grad_norm: The maximum value for the gradient clipping
    :param target_kl: Limit the KL divergence between updates,
        because the clipping is not enough to prevent large update
        see issue #213 (cf https://github.com/hill-a/stable-baselines/issues/213)
        By default, there is no limit on the kl div.
    :param tensorboard_log: the log location for tensorboard (if None, no logging)
    :param policy_kwargs: additional arguments to be passed to the policy on creation
    :param verbose: the verbosity level: 0 no output, 1 info, 2 debug
    :param seed: Seed for the pseudo random generators
    :param device: Device (cpu, cuda, ...) on which the code should be run.
        Setting it to auto, the code will be run on the GPU if possible.
    :param _init_setup_model: Whether or not to build the network at the creation of the instance
    """

    policy_aliases: Dict[str, Type[BasePolicy]] = {
        "MlpLstmPolicy": MlpLstmPolicy,
        "CnnLstmPolicy": CnnLstmPolicy,
        "MultiInputLstmPolicy": MultiInputLstmPolicy,
    }

    def __init__(
        self,
        policy: Union[str, Type[RecurrentActorCriticPolicy]],
        env: Union[GymEnv, str] = None,
        learning_rate: Union[float, Schedule] = 1e-4,
        n_steps: int = 128,
        use_tanh_act: bool = True,
        batch_size: Optional[int] = 128,
        n_epochs: int = 10,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_range: Union[float, Schedule] = 0.2,
        clip_range_vf: Union[None, float, Schedule] = None,
        normalize_advantage: bool = True,
        ent_coef: float = 0.0,
        vf_coef: float = 0.5,
        max_grad_norm: float = 0.5,
        use_sde: bool = False,
        retrain: bool = False,
        lstm_layer = 1,
        n_seq = 1,
        sde_sample_freq: int = -1,
        target_kl: Optional[float] = None,
        tensorboard_log: Optional[str] = None,
        eval_env: Union[GymEnv, str] = None,
        policy_kwargs: Optional[Dict[str, Any]] = None,
        verbose: int = 0,
        seed: Optional[int] = None,
        device: Union[th.device, str] = "auto",
        _init_setup_model: bool = True,
        state_vae: Optional[Dict[str, Any]] = None,
        features_dim: int = 32,
        states_dim: int = 0,
        only_lstm_training: bool = False,
        if_change_maps: bool = True,
        reconstruction_members: Optional[List[bool]] = [True, False, True],
        reconstruction_steps: int = 2,
        save_lstm_dataset: bool = False,
        train_lstm_without_env: bool = False,
        fine_tune_from_rosbag: bool = False,
        lstm_dataset_path: Optional[str] = None,
        lstm_weight_saved_path: Optional[str] = 'LSTM_weights',
        control_policy: Optional[Dict[str, Any]] = None,
        save_encoder: bool = False,
        use_kl_latent_loss: bool = False,
        observation_space: Optional[spaces.Space] = None,
        action_space: Optional[spaces.Space] = None,
    ):
        super().__init__(
            policy,
            env,
            learning_rate=learning_rate,
            n_steps=n_steps,
            gamma=gamma,
            gae_lambda=gae_lambda,
            ent_coef=ent_coef,
            vf_coef=vf_coef,
            max_grad_norm=max_grad_norm,
            use_sde=use_sde,
            sde_sample_freq=sde_sample_freq,
            tensorboard_log=tensorboard_log,
            policy_kwargs=policy_kwargs,
            verbose=verbose,
            seed=seed,
            device=device,
            _init_setup_model=False,
            supported_action_spaces=(
                spaces.Box,
                spaces.Discrete,
                spaces.MultiDiscrete,
                spaces.MultiBinary,
            ),
        )
        if env is None and observation_space is not None:
            self.observation_space = observation_space
        if env is None and action_space is not None:
            self.action_space = action_space

        self.use_tanh_act = use_tanh_act
        self.batch_size = batch_size
        self.n_epochs = n_epochs
        self.clip_range = clip_range
        self.clip_range_vf = clip_range_vf
        self.normalize_advantage = normalize_advantage
        self.retrain = retrain
        self.target_kl = target_kl
        self._last_lstm_states = None
        self.eval_env = eval_env
        self.lstm_layer = lstm_layer
        self.n_seq = n_seq
        self.state_vae = state_vae
        self.features_dim = features_dim
        self.states_dim = states_dim
        self.only_lstm_training = only_lstm_training
        self.finished_save_pc = True
        self.if_change_maps = if_change_maps
        self.reconstruction_members = reconstruction_members
        self.reconstruction_steps = reconstruction_steps
        self.save_lstm_dataset = save_lstm_dataset
        self.train_lstm_without_env = train_lstm_without_env
        self.lstm_dataset_path = lstm_dataset_path
        self.fine_tune_from_rosbag = fine_tune_from_rosbag
        self.control_policy = control_policy
        self.save_encoder = save_encoder
        self.use_kl_latent_loss = use_kl_latent_loss
        self.lstm_weight_saved_path = lstm_weight_saved_path

        if self.retrain:
            self.policy = policy
            if self.control_policy is not None:
                self.policy.load_state_dict(self.control_policy, strict=False)
            self.policy = self.policy.to(self.device)
        if _init_setup_model:
            self._setup_model()

    def _setup_model(self) -> None:
        self._setup_lr_schedule()
        self.set_random_seed(self.seed)

        if not self.retrain:
            self.policy = self.policy_class(
                self.observation_space,
                self.action_space,
                self.lr_schedule,
                use_sde=self.use_sde,
                n_lstm_layers=self.lstm_layer,
                lstm_hidden_size=256,
                shared_lstm = True,
                enable_critic_lstm = False,
                states_dim = self.states_dim,
                features_dim = self.features_dim,
                only_lstm_training = self.only_lstm_training,
                reconstruction_members = self.reconstruction_members,
                reconstruction_steps = self.reconstruction_steps,
                use_kl_loss = self.use_kl_latent_loss,
                **self.policy_kwargs,  # pytype:disable=not-instantiable
            )

            # 1) Add Tanh activation to Policy Net
            if self.use_tanh_act:
                self.policy.action_net = th.nn.Sequential(
                    self.policy.action_net, th.nn.Tanh()
                )
            if self.state_vae is not None:
                pretrained_cnn = {
                    'features_extractor.conv1.weight': self.state_vae['state_dict']['encoder.conv1.weight'],
                    'features_extractor.conv1.bias': self.state_vae['state_dict']['encoder.conv1.bias'],
                    'features_extractor.conv2.weight': self.state_vae['state_dict']['encoder.conv2.weight'],
                    'features_extractor.conv2.bias': self.state_vae['state_dict']['encoder.conv2.bias'],
                    'features_extractor.conv3.weight': self.state_vae['state_dict']['encoder.conv3.weight'],
                    'features_extractor.conv3.bias': self.state_vae['state_dict']['encoder.conv3.bias'],
                    'features_extractor.conv4.weight': self.state_vae['state_dict']['encoder.conv4.weight'],
                    'features_extractor.conv4.bias': self.state_vae['state_dict']['encoder.conv4.bias'],
                    'features_extractor.conv5.weight': self.state_vae['state_dict']['encoder.conv5.weight'],
                    'features_extractor.conv5.bias': self.state_vae['state_dict']['encoder.conv5.bias'],
                    'features_extractor.conv6.weight': self.state_vae['state_dict']['encoder.conv6.weight'],
                    'features_extractor.conv6.bias': self.state_vae['state_dict']['encoder.conv6.bias'],
                    'features_extractor.linear.weight': self.state_vae['state_dict']['encoder.fc_mu.weight'],
                    'features_extractor.linear.bias': self.state_vae['state_dict']['encoder.fc_mu.bias'],
                    'features_extractor.fc_logsigma.weight': self.state_vae['state_dict']['encoder.fc_logsigma.weight'],
                    'features_extractor.fc_logsigma.bias': self.state_vae['state_dict']['encoder.fc_logsigma.bias'],
                }
                self.policy.load_state_dict(pretrained_cnn, strict=False)

            if self.control_policy is not None:
                self.policy.load_state_dict(self.control_policy, strict=False)

            self.policy = self.policy.to(self.device)

        if not isinstance(self.policy, RecurrentActorCriticPolicy):
            raise ValueError("Policy must subclass RecurrentActorCriticPolicy")  
                  
        # We assume that LSTM for the actor and the critic
        # have the same architecture
        if not self.use_kl_latent_loss:
            lstm = self.policy.lstm_actor
        else:
            lstm = self.policy.lstm_mean

        if self.train_lstm_without_env:
            self.dataset_train = RolloutLSTMSequenceDataset(self.lstm_dataset_path, self.device, train=True)
            self.dataset_test = RolloutLSTMSequenceDataset(self.lstm_dataset_path, self.device, train=False)
            self.train_loader = torch.utils.data.DataLoader(
                self.dataset_train, batch_size=1, shuffle=False, num_workers=0)
            self.test_loader = torch.utils.data.DataLoader(
                self.dataset_test, batch_size=1, shuffle=False, num_workers=0)
            self.n_envs = 1
            lstm_logger = utils.configure_logger(self.verbose, self.tensorboard_log, self.lstm_weight_saved_path, False)
            self.set_logger(lstm_logger)
        # elif self.fine_tune_from_rosbag:
        #     self.dataset_train = RosbagSequenceDataset('real_imgs', '/camera/depth/image_rect_raw', transform=None, train=True)
        #     self.dataset_test  = RosbagSequenceDataset('real_imgs', '/camera/depth/image_rect_raw', transform=None, train=False)
        #     self.train_loader = torch.utils.data.DataLoader(
        #         self.dataset_train, batch_size=1, shuffle=False, num_workers=0)
        #     self.test_loader = torch.utils.data.DataLoader(
        #         self.dataset_test, batch_size=1, shuffle=False, num_workers=0)
        #     self.n_envs = 1
        elif self.only_lstm_training:
            hidden_state_buffer_shape = (self.n_steps, lstm.num_layers, self.n_envs, lstm.hidden_size)
            buffer_cls = LSTMThDictRolloutBuffer
            self.rollout_buffer = buffer_cls(
                self.n_steps,
                self.observation_space,
                self.action_space,
                hidden_state_buffer_shape,
                self.device,
                gamma=self.gamma,
                gae_lambda=self.gae_lambda,
                n_envs=self.n_envs,
                n_seq=self.n_seq,
            )
        else:
            hidden_state_buffer_shape = (self.n_steps, lstm.num_layers, self.n_envs, lstm.hidden_size)
            buffer_cls = LatentRolloutBuffer
            self.rollout_buffer = buffer_cls(
                self.n_steps,
                self.observation_space,
                self.action_space,
                hidden_state_buffer_shape,
                self.device,
                gamma=self.gamma,
                gae_lambda=self.gae_lambda,
                n_envs=self.n_envs,
                n_seq=self.n_seq,
                ppo_input_size=lstm.hidden_size + 14,
            )

        single_hidden_state_shape = (lstm.num_layers, self.n_envs, lstm.hidden_size)
        # hidden and cell states for actor and critic
        self._last_lstm_states = RNNStates(
            (
                th.zeros(single_hidden_state_shape, device=self.device),
                th.zeros(single_hidden_state_shape, device=self.device),
            ),
            (
                th.zeros(single_hidden_state_shape, device=self.device),
                th.zeros(single_hidden_state_shape, device=self.device),
            ),
        )
        if self.use_kl_latent_loss:
            self._last_lstm_states_std = RNNStates(
                (
                    th.zeros(single_hidden_state_shape, device=self.device),
                    th.zeros(single_hidden_state_shape, device=self.device),
                ),
                (
                    th.zeros(single_hidden_state_shape, device=self.device),
                    th.zeros(single_hidden_state_shape, device=self.device),
                ),
            )


        # Initialize schedules for policy/value clipping
        self.clip_range = get_schedule_fn(self.clip_range)
        if self.clip_range_vf is not None:
            if isinstance(self.clip_range_vf, (float, int)):
                assert self.clip_range_vf > 0, "`clip_range_vf` must be positive, pass `None` to deactivate vf clipping"

            self.clip_range_vf = get_schedule_fn(self.clip_range_vf)

    def kl_latent_collect_rollouts(
        self,
        env: VecEnv,
        callback: BaseCallback,
        rollout_buffer: RolloutBuffer,
        n_rollout_steps: int,
        iteration: int,
        if_easy_start: bool = True,
        deterministic: bool = False,
    ) -> bool:
        """
        Collect experiences using the current policy and fill a ``RolloutBuffer``.
        The term rollout here refers to the model-free notion and should not
        be used with the concept of rollout used in model-based RL or planning.

        :param env: The training environment
        :param callback: Callback that will be called at each step
            (and at the beginning and end of the rollout)
        :param rollout_buffer: Buffer to fill with rollouts
        :param n_steps: Number of experiences to collect per environment
        :return: True if function returned with at least `n_rollout_steps`
            collected, False if callback terminated rollout prematurely.
        """
        assert isinstance(
            rollout_buffer, (RecurrentRolloutBuffer, LatentRolloutBuffer, LSTMThDictRolloutBuffer)
        ), f"{rollout_buffer} doesn't support recurrent policy"

        assert self._last_obs is not None, "No previous observation was provided"
        # Switch to eval mode (this affects batch norm / dropout)
        self.policy.set_training_mode(False)

        n_steps = 0
        rollout_buffer.reset()
        # Sample new weights for the state dependent exploration
        if self.use_sde:
            self.policy.reset_noise(env.num_envs)

        callback.on_rollout_start()

        lstm_states = deepcopy(self._last_lstm_states)
        lstm_states_std = deepcopy(self._last_lstm_states_std)

        if self.if_change_maps and iteration % 2 == 0:
            if self.num_timesteps<1.6e7:
                self.env.reset(if_easy_start=if_easy_start)
            else:
                self.env.reset()
        # if self.num_timesteps==6.4e6:
        #     self.env.reset_reward_coeffs()

        while n_steps < n_rollout_steps:

            if self.use_sde and self.sde_sample_freq > 0 and n_steps % self.sde_sample_freq == 0:
                # Sample a new noise matrix
                self.policy.reset_noise(env.num_envs)
            with th.inference_mode():
                # Convert to pytorch tensor or to TensorDict
                obs_tensor = self._last_obs
                episode_starts = self._last_episode_starts
                latent_pi, latent_vf, lstm_states, lstm_states_std = self.policy.forward_rnn_latent_kl_latent(obs_tensor, lstm_states, lstm_states_std, episode_starts)
                actions, values, log_probs = self.policy.forward(latent_pi, latent_vf, deterministic=deterministic)
            # Rescale and perform action
            # Clip the actions to avoid out of bound error
            actions_np = actions.cpu().numpy()
            if isinstance(self.action_space, spaces.Box):
                actions_np = np.clip(actions_np, self.action_space.low, self.action_space.high)
            clipped_actions = th.tensor(actions_np, device=self.device, dtype=th.float32)
            new_obs, rewards, dones, infos = env.step(clipped_actions)
            self.num_timesteps += env.num_envs
            # Give access to local variables
            callback.update_locals(locals())
            if callback.on_step() is False:
                return False

            self._update_info_buffer(infos)
            n_steps += 1

            if isinstance(self.action_space, spaces.Discrete):
                # Reshape in case of discrete action
                actions = actions.reshape(-1, 1)

            rollout_buffer.add(
                self._last_obs,
                actions,
                rewards,
                self._last_episode_starts,
                values,
                log_probs,
                lstm_states=self._last_lstm_states,
                latent_lstm_pi=latent_pi,
                latent_lstm_vf=latent_vf,
            )

            self._last_obs = new_obs.clone()
            self._last_episode_starts = dones.clone()
            self._last_lstm_states = lstm_states
            self._last_lstm_states_std = lstm_states_std

        with th.no_grad():
            # Compute value for the last timestep
            episode_starts = dones
            values = self.policy.predict_values_latent_kl_latent(new_obs, lstm_states.vf, lstm_states_std.vf, episode_starts)
        rollout_buffer.compute_returns_and_advantage(last_values=values.flatten(), dones=dones)

        callback.on_rollout_end()

        return True

    def collect_rollouts(
        self,
        env: VecEnv,
        callback: BaseCallback,
        rollout_buffer: RolloutBuffer,
        n_rollout_steps: int,
        iteration: int,
        if_easy_start: bool = True,
        deterministic: bool = False,
    ) -> bool:
        """
        Collect experiences using the current policy and fill a ``RolloutBuffer``.
        The term rollout here refers to the model-free notion and should not
        be used with the concept of rollout used in model-based RL or planning.

        :param env: The training environment
        :param callback: Callback that will be called at each step
            (and at the beginning and end of the rollout)
        :param rollout_buffer: Buffer to fill with rollouts
        :param n_steps: Number of experiences to collect per environment
        :return: True if function returned with at least `n_rollout_steps`
            collected, False if callback terminated rollout prematurely.
        """
        assert isinstance(
            rollout_buffer, (RecurrentRolloutBuffer, LatentRolloutBuffer, LSTMThDictRolloutBuffer)
        ), f"{rollout_buffer} doesn't support recurrent policy"

        assert self._last_obs is not None, "No previous observation was provided"
        # Switch to eval mode (this affects batch norm / dropout)
        self.policy.set_training_mode(False)

        n_steps = 0
        rollout_buffer.reset()
        # Sample new weights for the state dependent exploration
        if self.use_sde:
            self.policy.reset_noise(env.num_envs)

        callback.on_rollout_start()

        lstm_states = deepcopy(self._last_lstm_states)

        if self.if_change_maps and iteration % 2 == 0:
            if self.num_timesteps<2.0e7:
                self.env.reset(if_easy_start=if_easy_start)
            else:
                self.env.reset()
        # if self.num_timesteps==6.4e6:
        #     self.env.reset_reward_coeffs()

        while n_steps < n_rollout_steps:

            if self.use_sde and self.sde_sample_freq > 0 and n_steps % self.sde_sample_freq == 0:
                # Sample a new noise matrix
                self.policy.reset_noise(env.num_envs)
            with th.inference_mode():
                # Convert to pytorch tensor or to TensorDict
                obs_tensor = self._last_obs
                episode_starts = self._last_episode_starts
                latent_pi, latent_vf, lstm_states = self.policy.forward_rnn_latent(obs_tensor, lstm_states, episode_starts)

                actions, values, log_probs = self.policy.forward(latent_pi, latent_vf, deterministic=deterministic)
            # Rescale and perform action
            # Clip the actions to avoid out of bound error
            actions_np = actions.cpu().numpy()
            if isinstance(self.action_space, spaces.Box):
                actions_np = np.clip(actions_np, self.action_space.low, self.action_space.high)
            clipped_actions = th.tensor(actions_np, device=self.device, dtype=th.float32)
            new_obs, rewards, dones, infos = env.step(clipped_actions)
            self.num_timesteps += env.num_envs
            # Give access to local variables
            callback.update_locals(locals())
            if callback.on_step() is False:
                return False

            self._update_info_buffer(infos)
            n_steps += 1

            if isinstance(self.action_space, spaces.Discrete):
                # Reshape in case of discrete action
                actions = actions.reshape(-1, 1)

            rollout_buffer.add(
                self._last_obs,
                actions,
                rewards,
                self._last_episode_starts,
                values,
                log_probs,
                lstm_states=self._last_lstm_states,
                latent_lstm_pi=latent_pi,
                latent_lstm_vf=latent_vf,
            )

            self._last_obs = new_obs.clone()
            self._last_episode_starts = dones.clone()
            self._last_lstm_states = lstm_states

        with th.no_grad():
            # Compute value for the last timestep
            episode_starts = dones
            values = self.policy.predict_values_latent(new_obs, lstm_states.vf, episode_starts)
        rollout_buffer.compute_returns_and_advantage(last_values=values.flatten(), dones=dones)

        callback.on_rollout_end()

        return True
    
    def collect_lstm_rollouts(
        self,
        env: VecEnv,
        callback: BaseCallback,
        rollout_buffer: RolloutBuffer,
        n_rollout_steps: int,
        deterministic: bool = False,
    ) -> bool:
        """
        Collect experiences using the current policy and fill a ``RolloutBuffer``.
        The term rollout here refers to the model-free notion and should not
        be used with the concept of rollout used in model-based RL or planning.

        :param env: The training environment
        :param callback: Callback that will be called at each step
            (and at the beginning and end of the rollout)
        :param rollout_buffer: Buffer to fill with rollouts
        :param n_steps: Number of experiences to collect per environment
        :return: True if function returned with at least `n_rollout_steps`
            collected, False if callback terminated rollout prematurely.
        """
        assert isinstance(
            rollout_buffer, (RecurrentRolloutBuffer, LatentRolloutBuffer, LSTMThDictRolloutBuffer)
        ), f"{rollout_buffer} doesn't support recurrent policy"

        assert self._last_obs is not None, "No previous observation was provided"
        # Switch to eval mode (this affects batch norm / dropout)
        self.policy.set_training_mode(False)

        n_steps = 0
        rollout_buffer.reset()
        # Sample new weights for the state dependent exploration
        if self.use_sde:
            self.policy.reset_noise(env.num_envs)

        callback.on_rollout_start()

        lstm_states = deepcopy(self._last_lstm_states)

        if self.if_change_maps:
            self.env.reset()
        
        while n_steps < n_rollout_steps:
            if self.use_sde and self.sde_sample_freq > 0 and n_steps % self.sde_sample_freq == 0:
                # Sample a new noise matrix
                self.policy.reset_noise(env.num_envs)

            with th.no_grad():
                # Convert to pytorch tensor or to TensorDict
                obs_tensor = self._last_obs
                episode_starts = self._last_episode_starts
                latent_pi, latent_vf, lstm_states = self.policy.forward_rnn(obs_tensor, lstm_states, episode_starts)
                actions, values, log_probs = self.policy.forward(latent_pi, latent_vf, deterministic=deterministic)

            actions_np = actions.cpu().numpy()
            # Clip the actions to avoid out of bound error
            if isinstance(self.action_space, spaces.Box):
                actions_np = np.clip(actions_np, self.action_space.low, self.action_space.high)
            clipped_actions = th.tensor(actions_np, device=self.device, dtype=th.float32)
            new_obs, rewards, dones, infos = env.step(clipped_actions)
            self.num_timesteps += env.num_envs
            # Give access to local variables
            callback.update_locals(locals())
            if callback.on_step() is False:
                return False

            self._update_info_buffer(infos)
            n_steps += 1

            if isinstance(self.action_space, spaces.Discrete):
                # Reshape in case of discrete action
                actions = actions.reshape(-1, 1)

            rollout_buffer.add(
                self._last_obs,
                actions,
                rewards,
                self._last_episode_starts,
                values,
                log_probs,
                lstm_states=self._last_lstm_states,
            )

            self._last_obs = new_obs
            self._last_episode_starts = dones.clone()
            self._last_lstm_states = lstm_states

        with th.no_grad():
            # Compute value for the last timestep
            episode_starts = dones
            values = self.policy.predict_values(new_obs, lstm_states.vf, episode_starts)
        rollout_buffer.compute_returns_and_advantage(values.flatten(), dones=dones)

        callback.on_rollout_end()

        return True

    def train(self) -> None:
        """
        Update policy using the currently gathered rollout buffer.
        """
        # Switch to train mode (this affects batch norm / dropout)
        self.policy.set_training_mode(True)
        # Update optimizer learning rate
        self._update_learning_rate(self.policy.optimizer)
        # Compute current clip range
        clip_range = self.clip_range(self._current_progress_remaining)
        # Optional: clip range for the value function
        if self.clip_range_vf is not None:
            clip_range_vf = self.clip_range_vf(self._current_progress_remaining)

        entropy_losses = []
        pg_losses, value_losses = [], []
        clip_fractions = []

        continue_training = True
        # train for n_epochs epochs
        for epoch in range(self.n_epochs):
            approx_kl_divs = []
            # Do a complete pass on the rollout buffer
            for rollout_data in self.rollout_buffer.get_ppo_need(self.batch_size):
                actions = rollout_data.actions
                if isinstance(self.action_space, spaces.Discrete):
                    # Convert discrete action from float to long
                    actions = rollout_data.actions.long().flatten()
                # Convert mask from float to bool
                # mask = rollout_data.mask > 1e-8
                # Re-sample the noise matrix because the log_std has changed
                if self.use_sde:
                    self.policy.reset_noise(self.batch_size)

                values, log_prob, entropy = self.policy.evaluate_actions(
                    rollout_data.latent_lstm_pi,
                    rollout_data.latent_lstm_vf,
                    actions,
                )
                values = values.flatten()
                # Normalize advantage
                advantages = rollout_data.advantages
                if self.normalize_advantage:
                    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

                # ratio between old and new policy, should be one at the first iteration
                ratio = th.exp(log_prob - rollout_data.old_log_prob)
                # clipped surrogate loss
                policy_loss_1 = advantages * ratio
                policy_loss_2 = advantages * th.clamp(ratio, 1 - clip_range, 1 + clip_range)
                policy_loss = -th.mean(th.min(policy_loss_1, policy_loss_2))

                # Logging
                pg_losses.append(policy_loss.item())
                clip_fraction = th.mean((th.abs(ratio - 1) > clip_range).float()).item()
                clip_fractions.append(clip_fraction)

                if self.clip_range_vf is None:
                    # No clipping
                    values_pred = values
                else:
                    # Clip the different between old and new value
                    # NOTE: this depends on the reward scaling
                    values_pred = rollout_data.old_values + th.clamp(
                        values - rollout_data.old_values, -clip_range_vf, clip_range_vf
                    )
                # Value loss using the TD(gae_lambda) target
                # Mask padded sequences
                value_loss = th.mean(((rollout_data.returns - values_pred) ** 2))

                value_losses.append(value_loss.item())

                # Entropy loss favor exploration
                if entropy is None:
                    # Approximate entropy when no analytical form
                    entropy_loss = -th.mean(-log_prob)
                else:
                    entropy_loss = -th.mean(entropy)

                entropy_losses.append(entropy_loss.item())

                loss = policy_loss + self.ent_coef * entropy_loss + self.vf_coef * value_loss
                # print("loss: ", loss)
                # Calculate approximate form of reverse KL Divergence for early stopping
                # see issue #417: https://github.com/DLR-RM/stable-baselines3/issues/417
                # and discussion in PR #419: https://github.com/DLR-RM/stable-baselines3/pull/419
                # and Schulman blog: http://joschu.net/blog/kl-approx.html
                with th.no_grad():
                    log_ratio = log_prob - rollout_data.old_log_prob
                    approx_kl_div = th.mean(((th.exp(log_ratio) - 1) - log_ratio)).cpu().numpy()
                    approx_kl_divs.append(approx_kl_div)

                if self.target_kl is not None and approx_kl_div > 1.5 * self.target_kl:
                    continue_training = False
                    if self.verbose >= 1:
                        print(f"Early stopping at step {epoch} due to reaching max kl: {approx_kl_div:.2f}")
                    break
                
                # Optimization step
                self.policy.optimizer.zero_grad()
                loss.backward()
                # Clip grad norm
                th.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
                self.policy.optimizer.step()

            if not continue_training:
                break

        self._n_updates += self.n_epochs
        explained_var = explained_variance(self.rollout_buffer.values.flatten().cpu().numpy(), self.rollout_buffer.returns.flatten().cpu().numpy())

        # Logs
        self.logger.record("train/entropy_loss", np.mean(entropy_losses))
        self.logger.record("train/policy_gradient_loss", np.mean(pg_losses))
        self.logger.record("train/value_loss", np.mean(value_losses))
        self.logger.record("train/approx_kl", np.mean(approx_kl_divs))
        self.logger.record("train/clip_fraction", np.mean(clip_fractions))
        self.logger.record("train/loss", loss.item())
        self.logger.record("train/explained_variance", explained_var)
        if hasattr(self.policy, "log_std"):
            self.logger.record("train/std", th.exp(self.policy.log_std).mean().item())

        self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
        self.logger.record("train/clip_range", clip_range)
        if self.clip_range_vf is not None:
            self.logger.record("train/clip_range_vf", clip_range_vf)

    def eval(self, iteration, if_eval=True, max_ep_length=1000) -> None:
        save_path = self.logger.get_dir() + "/TestTraj"
        save_vis_path = self._logger.get_dir() + "/TSNE/" + "TSNE_{0:05d}".format(iteration)
        os.makedirs(save_path, exist_ok=True)
        os.makedirs(save_vis_path, exist_ok=True)
        #
        self.policy.eval()
        # rollout trajectory and save the trajectory
        if self.is_forest_env:
            easy_r = 5.5
            medium_r = 4.0
            hard_r = 3.2
        else:
            easy_r = 3.2
            medium_r = 2.5
            hard_r = 1.8
        # easy map
        # self.change_maps(env=self.eval_env, seed=10, radius=easy_r, if_eval=if_eval)
        # traj_df1, ave_reward1, success_rate1, trial_numbers1 = traj_rollout(self.eval_env, self.policy, max_ep_length=max_ep_length)
        # traj_df1.to_csv(save_path + "/test_traj_{0:05d}_easy.csv".format(iteration))
        # medium map
        self.change_maps(env=self.eval_env, seed=20, radius=medium_r, if_eval=if_eval)
        traj_df0, ave_reward0, success_rate0, trial_numbers0 = traj_rollout(self.eval_env, self.policy, max_ep_length=max_ep_length)
        traj_df0.to_csv(save_path + "/test_traj_{0:05d}_medium1.csv".format(iteration))
        # hard map
        self.change_maps(env=self.eval_env, seed=20, radius=hard_r, if_eval=if_eval)
        traj_df1, ave_reward1, success_rate1, trial_numbers1 = traj_rollout(self.eval_env, self.policy, max_ep_length=max_ep_length)
        traj_df1.to_csv(save_path + "/test_traj_{0:05d}_hard1.csv".format(iteration))

        self.change_maps(env=self.eval_env, seed=10, radius=medium_r, if_eval=if_eval)
        traj_df2, ave_reward2, success_rate2, trial_numbers2 = traj_rollout(self.eval_env, self.policy, max_ep_length=max_ep_length)
        traj_df2.to_csv(save_path + "/test_traj_{0:05d}_medium2.csv".format(iteration))
        # hard map
        self.change_maps(env=self.eval_env, seed=10, radius=hard_r, if_eval=if_eval)
        traj_df3, ave_reward3, success_rate3, trial_numbers3 = traj_rollout(self.eval_env, self.policy, max_ep_length=max_ep_length)
        traj_df3.to_csv(save_path + "/test_traj_{0:05d}_hard2.csv".format(iteration))
        self.finished_save_pc = True

        # self.logger.record("test/ave_reward_easy", ave_reward1)
        # self.logger.record("test/success_rate_easy", success_rate1)
        # self.logger.record("test/trial_numbers_easy", trial_numbers1)
        self.logger.record("test/ave_reward_medium1", ave_reward0)
        self.logger.record("test/success_rate_medium1", success_rate0)
        self.logger.record("test/trial_numbers_medium1", trial_numbers0)
        self.logger.record("test/ave_reward_hard1", ave_reward1)
        self.logger.record("test/success_rate_hard1", success_rate1)
        self.logger.record("test/trial_numbers_hard1", trial_numbers1)

        self.logger.record("test/ave_reward_medium2", ave_reward2)
        self.logger.record("test/success_rate_medium2", success_rate2)
        self.logger.record("test/trial_numbers_medium2", trial_numbers2)
        self.logger.record("test/ave_reward_hard2", ave_reward3)
        self.logger.record("test/success_rate_hard2", success_rate3)
        self.logger.record("test/trial_numbers_hard2", trial_numbers3)
        self.logger.dump(step=iteration)

    def learn(
        self: SelfRecurrentPPO,
        total_timesteps: int,
        callback: MaybeCallback = None,
        log_interval: Tuple = (10, 100),
        progress_bar: bool = True,
        tb_log_name: str = "RecurrentPPO",
        reset_num_timesteps: bool = True,
        normailize_state: bool = True,
        if_easy_start: bool = True,
        deterministic: bool = False,
    ) -> SelfRecurrentPPO:
        iteration = 0

        total_timesteps, callback = self._setup_learn(
            total_timesteps,
            callback,
            reset_num_timesteps,
            tb_log_name,
            progress_bar
        )
        self._last_episode_starts = th.ones((self.env.num_envs,), dtype=th.float32, device=self.device)

        callback.on_training_start(locals(), globals())

        while self.num_timesteps < total_timesteps:
            if not self.use_kl_latent_loss:
                continue_training = self.collect_rollouts(self.env, callback, self.rollout_buffer, n_rollout_steps=self.n_steps, if_easy_start=if_easy_start, iteration=iteration, deterministic=deterministic)
            else:
                continue_training = self.kl_latent_collect_rollouts(self.env, callback, self.rollout_buffer, n_rollout_steps=self.n_steps, iteration=iteration, if_easy_start=if_easy_start, deterministic=deterministic)
            if continue_training is False:
                break

            iteration += 1
            self._update_current_progress_remaining(self.num_timesteps, total_timesteps)

            # Display training infos
            if log_interval is not None and iteration % log_interval[0] == 0:
                time_elapsed = max((time.time_ns() - self.start_time) / 1e9, sys.float_info.epsilon)
                fps = int((self.num_timesteps - self._num_timesteps_at_start) / time_elapsed)
                self.logger.record("time/iterations", iteration, exclude="tensorboard")
                if len(self.ep_info_buffer) > 0 and len(self.ep_info_buffer[0]) > 0:
                    self.logger.record("rollout/ep_rew_mean", safe_mean([ep_info["r"] for ep_info in self.ep_info_buffer]))
                    self.logger.record("rollout/ep_len_mean", safe_mean([ep_info["l"] for ep_info in self.ep_info_buffer]))
                self.logger.record("time/fps", fps)
                self.logger.record("time/time_elapsed", int(time_elapsed), exclude="tensorboard")
                self.logger.record("time/total_timesteps", self.num_timesteps, exclude="tensorboard")
                self.logger.record("rollout/success_rate", self.env.cal_success_rate().clone().detach().item())

                for i in range(self.env.rew_dim - 1):
                    self.logger.record(
                        "rewards/{0}".format(self.env.reward_names[i]),
                        safe_mean(
                            [
                                ep_info[self.env.reward_names[i]]
                                for ep_info in self.ep_info_buffer
                            ]
                        ),
                    )
                self.logger.dump(step=self.num_timesteps)

            self.train()


            if iteration % 10 == 0 and iteration <= 1000 and normailize_state:
                # update running mean and standard deivation for state normalization
                self.env.update_rms()
                
            if log_interval is not None and iteration % log_interval[1] == 0:

                policy_path = self.logger.get_dir() + "/Policy"
                os.makedirs(policy_path, exist_ok=True)
                if normailize_state:
                    self.env.save_rms(
                        save_dir=self.logger.get_dir() + "/RMS", n_iter=iteration
                    )

                self.policy.save(policy_path + "/iter_{0:05d}.pth".format(iteration))
                if self.save_encoder:
                    encoder_path = self.logger.get_dir() + "/Encoder"
                    os.makedirs(encoder_path, exist_ok=True)
                    # self.env.get_encoder().save(encoder_path + "/iter_{0:05d}.pth".format(iteration))
                    th.save(self.env.get_encoder().state_dict(), encoder_path + "/iter_{0:05d}.pth".format(iteration))

                # self.eval(iteration)
        callback.on_training_end()

        return self
    
    def setup_eval(self) -> None:
        self._setup_learn(total_timesteps=0,
                    tb_log_name="RecurrentPPO_EVAL")
    
    def eval_from_outer(self, iteration) -> None:
        self.eval(iteration, if_eval=False, max_ep_length=10000)

    def change_maps(self, env, seed=-1, radius=-1.0, if_eval=False):
        self.finished_save_pc = False
        self.env.spawnObstacles(change_obs=True, seed=seed, radius=radius)
        while not self.env.ifSceneChanged():
            self.env.spawnObstacles(change_obs=False)
            time.sleep(0.02)
        self.env.getPointClouds('', 0, True)
        time.sleep(0.2)
        while(not self.env.getSavingState()):
            time.sleep(0.02)
        if self.is_forest_env:
            time.sleep(12.0)
        else:
            time.sleep(2.0)
        env.readPointClouds(0)
        while(not env.getReadingState()):
            time.sleep(0.02)
        time.sleep(1.0)
        if not if_eval:
            self.finished_save_pc = True

    def change_policy(self, weight):
        self.policy.load_state_dict(weight, strict=False)
    
    def eval_lstm(self, iteration) -> None:
        save_path = self.logger.get_dir() + "/Reconstruction"
        os.makedirs(save_path, exist_ok=True)
        #
        self.policy.eval()
        lstm_rollout(self.eval_env, self.policy, self.device, save_path, iteration)
        # rollout trajectory and save the trajectory
        # traj_df, features, labels = traj_rollout(self.eval_env, self.policy)

    def train_lstm(self):
        # Switch to train mode (this affects batch norm / dropout)
        self.policy.set_training_mode(True)
        # Update optimizer learning rate
        self._update_learning_rate(self.policy.optimizer)
        total_loss = 0
        record_loss = 0
        # train for n_epochs epochs
        for epoch in range(self.n_epochs):
            # Do a complete pass on the rollout buffer
            for rollout_data in self.rollout_buffer.get(self.batch_size):
                latent_obs = self.policy.to_latent(rollout_data.observations)
                # recon_current, recon_previous, n_seq, _ = self.policy.predict_lstm(latent_obs, rollout_data.lstm_states.pi, rollout_data.episode_starts)
                # loss = self.lstm_loss_function(rollout_data.observations, recon_current, recon_previous, n_seq)
                recon, n_seq, _ = self.policy.predict_lstm(latent_obs, rollout_data.lstm_states.pi, rollout_data.episode_starts)
                loss, record = self.lstm_loss_function(rollout_data.observations, recon, n_seq, epoch)
                print("epoch: ", epoch, "  --loss: ", loss.item())
                # print("pre_next_obs: ", pre_next_obs[:, :35])
                # print("next_obs: ", latent_obs[:, :35])
                self.policy.optimizer.zero_grad()
                loss.backward()
                self.policy.optimizer.step()
                total_loss += loss
                record_loss += record
        return total_loss / self.n_epochs, record_loss / self.n_epochs
    
    def fine_tune_lstm_from_rosbag(self):
        for epoch in range(self.n_epochs):
            self.policy.set_training_mode(True)
            # Update optimizer learning rate
            self._update_learning_rate(self.policy.optimizer)
            train_loss = 0
            future_loss = 0
            self.dataset_train.load_next_buffer()
            for batch_idx, data in enumerate(self.train_loader):
                obs_th = data.squeeze().unsqueeze(1).to(self.device)
                latent_obs = self.policy.to_latent(obs_th)
                single_hidden_state_shape = self.policy.lstm_hidden_state_shape
                lstm_states = (
                    th.zeros(single_hidden_state_shape,  device=self.device),
                    th.zeros(single_hidden_state_shape,  device=self.device),
                )
                episode_starts = th.zeros((1,), dtype=th.float32, device=self.device)
                recon, n_seq, _ = self.policy.predict_lstm(latent_obs, lstm_states, episode_starts)
                loss, record = self.lstm_loss_function(obs_th, recon, n_seq, epoch)
                self.policy.optimizer.zero_grad()
                train_loss += loss.item()
                future_loss += record
                loss.backward()
                self.policy.optimizer.step()
                if batch_idx % 5 == 0:
                    print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}, Loss for Future: {:.6f}'.format(
                        epoch, batch_idx * len(data), len(self.train_loader.dataset),
                        100. * batch_idx / len(self.train_loader),
                        loss.item() / len(data), record / len(data)))
            print('====> Epoch: {} Average loss: {:.4f}, Future loss: {:.4}'.format(
                epoch, train_loss / len(self.train_loader.dataset), future_loss / len(self.train_loader.dataset)))
            self.logger.record("train/loss", train_loss / len(self.train_loader.dataset))
            self.logger.record("train/future_loss", future_loss / len(self.train_loader.dataset))
            self.logger.dump(step=epoch)
            if epoch % 10 == 0:
                self.test_lstm_from_dataset(epoch)
            if epoch % 20 == 0:
                policy_path =self.logger.get_dir() + "/Policy"
                os.makedirs(policy_path, exist_ok=True)
                self.policy.save(self.logger.get_dir() + "/Policy" + "/iter_{0:05d}.pth".format(epoch))
    
    def log_depth(self, x: torch.Tensor, bound_min=False):
        """ Log depth transformation """
        if bound_min:
            x = x * 10.0
            x = torch.clamp(x, min=0.1)
        else:
            x = x * (10.0 - 0.1) + 0.1
        return (1.0 + torch.log10(x)) / 2.0

    def train_lstm_from_dataset(self, use_log_depth=False):
        # train for n_epochs epochs
        for epoch in range(self.n_epochs):
            self.policy.set_training_mode(True)
            # Update optimizer learning rate
            self._update_learning_rate(self.policy.optimizer)
            train_loss = 0
            future_loss = 0
            self.dataset_train.load_next_buffer()
            for batch_idx, data in enumerate(self.train_loader):

                n_seq = data[1][0].shape[1]
                observations = {key: obs[0] for (key, obs) in data[0].items()}

                lstm_states = data[1]
                episode_starts = data[2]
                latent_obs = self.policy.to_latent(observations)

                recon, n_seq, _ = self.policy.predict_lstm(latent_obs, lstm_states, episode_starts)
                loss, record = self.lstm_loss_function(observations, recon, n_seq, epoch, use_log_depth)
                self.policy.optimizer.zero_grad()
                train_loss += loss.item()
                future_loss += record
                loss.backward()
                self.policy.optimizer.step()
                if batch_idx % 20 == 0:
                    print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}, Loss for Future: {:.6f}'.format(
                        epoch, batch_idx * len(data), len(self.train_loader.dataset),
                        100. * batch_idx / len(self.train_loader),
                        loss.item() / len(data), record / len(data)))
            print('====> Epoch: {} Average loss: {:.4f}, Future loss: {:.4}'.format(
                epoch, train_loss / len(self.train_loader.dataset), future_loss / len(self.train_loader.dataset)))
            self.logger.record("train/loss", train_loss / len(self.train_loader.dataset))
            self.logger.record("train/future_loss", future_loss / len(self.train_loader.dataset))
            self.logger.dump(step=epoch)
            # test the model and save the model each 50 epochs
            if epoch % 10 == 0:
                self.test_lstm_from_dataset(epoch, use_log_depth)
            if epoch % 50 == 0:
                policy_path =self.logger.get_dir() + "/Policy"
                os.makedirs(policy_path, exist_ok=True)
                self.policy.save(self.logger.get_dir() + "/Policy" + "/iter_{0:05d}.pth".format(epoch))
            
    def test_lstm_from_dataset(self, epoch, use_log_depth=False):
        self.policy.eval()
        self.dataset_test.load_next_buffer()
        test_loss = 0
        future_loss = 0
        with th.no_grad():
            for data in self.test_loader:
                if not self.fine_tune_from_rosbag:
                    observations = {key: obs[0] for (key, obs) in data[0].items()}

                    # only test the first n_seq images if your pc don't have enough memory
                    # n_seq = data[1][0][0].shape[1]
                    # img_num = observations['image'].shape[0]
                    # observations = {key: obs[0 : int(img_num/n_seq)] for (key, obs) in observations.items()}

                    lstm_states = data[1]
                    episode_starts = data[2]
                else:
                    observations = data.squeeze().unsqueeze(1).to(self.device)
                    single_hidden_state_shape = self.policy.lstm_hidden_state_shape
                    lstm_states = (
                        th.zeros(single_hidden_state_shape,  device=self.device),
                        th.zeros(single_hidden_state_shape,  device=self.device),
                    )
                    episode_starts = th.zeros((1,), dtype=th.float32, device=self.device)
                latent_obs = self.policy.to_latent(observations)
                recon, n_seq, _ = self.policy.predict_lstm(latent_obs, lstm_states, episode_starts)
                loss, record = self.lstm_loss_function(observations, recon, n_seq, 0, use_log_depth)
                test_loss += loss.item()
                future_loss += record
        test_loss /= len(self.test_loader.dataset)
        future_loss /= len(self.test_loader.dataset)
        print('====> Test set loss: {:.4f}, Future loss: {:.4}'.format(test_loss, future_loss))
        self.logger.record("test/loss", test_loss)
        self.logger.record("test/future_loss", future_loss)
        self.logger.dump(step=epoch)
        save_path = self.logger.get_dir() + "/Reconstruction"
        os.makedirs(save_path, exist_ok=True)
        if self.fine_tune_from_rosbag:
            self.plot_depth_image(observations, recon, n_seq, epoch)
        else:
            self.plot_test_image(observations, recon, n_seq, epoch)

    def test_lstm_seperate(self):
        self.policy.eval()
        self.dataset_test.load_next_buffer()
        test_loss = 0
        for data in self.test_loader:
            observations = {key: obs[0] for (key, obs) in data[0].items()}
            lstm_states = (data[1][0][0], data[1][1][0])
            episode_starts = data[2][0]
            latent_obs = self.policy.to_latent(observations)
            recon, n_seq, _ = self.policy.predict_lstm(latent_obs, lstm_states, episode_starts)
            mae, std = self.lstm_test_loss_std(observations, recon, n_seq)
            print('====> Test set loss: {:.4f}, std: {:.4}'.format(mae, std))

    def plot_test_image(self, obs, recon, n_seq, epoch):
        if isinstance(obs, dict):
            obs = obs['image']
        shape = obs.shape
        recon_next_plot = None
        recon_previous_plot = None
        recon_current_plot = None
        obs = obs[0 : int(shape[0]/n_seq), :, :, :].float() / 255.0
        if recon[0] is not None:
            recon_previous_plot = recon[0][0 : int(shape[0]/n_seq), :, :, :]
        if recon[1] is not None:
            recon_current_plot = recon[1][0 : int(shape[0]/n_seq), :, :, :]
        if recon[2] is not None:
            recon_next_plot = recon[2][0 : int(shape[0]/n_seq), :, :, :]
        print("recon_current_plot: ", recon_current_plot.max(), recon_current_plot.min())
        saved_images = []
        # save the plot each 20 timesteps
        for i in range(15, int(shape[0]/n_seq), 25):
            plot = []
            if recon_previous_plot is not None:
                plot.append(obs[i-self.reconstruction_steps])
                plot.append(recon_previous_plot[i-self.reconstruction_steps])
            if recon_current_plot is not None:
                plot.append(obs[i])
                plot.append(recon_current_plot[i])
            if recon_next_plot is not None:
                plot.append(obs[i+self.reconstruction_steps])
                plot.append(recon_next_plot[i])
            saved_images.append(th.stack(plot, dim=0))
        save_image(th.cat(saved_images), self.logger.get_dir() + "/Reconstruction" + "/recon_{0:05d}.png".format(epoch))

    def plot_depth_image(self, obs, recon, n_seq, seq_num):
        if isinstance(obs, dict):
            obs = obs['image']
        shape = obs.shape
        recon_next_plot = None
        recon_previous_plot = None
        recon_current_plot = None
        obs = obs[0 : int(shape[0]/n_seq), :, :, :].float() / 255.0
        if recon[0] is not None:
            recon_previous_plot = recon[0][0 : int(shape[0]/n_seq), :, :, :]
        if recon[1] is not None:
            recon_current_plot = recon[1][0 : int(shape[0]/n_seq), :, :, :]
        if recon[2] is not None:
            recon_next_plot = recon[2][0 : int(shape[0]/n_seq), :, :, :]
        save_path = self.logger.get_dir() + "/Reconstruction/Sequence_{0}".format(seq_num)
        save_path3 = self.logger.get_dir() + "/Reconstruction/Sequence_{0}/recon_future".format(seq_num)
        save_path2 = self.logger.get_dir() + "/Reconstruction/Sequence_{0}/recon_current".format(seq_num)
        save_path1 = self.logger.get_dir() + "/Reconstruction/Sequence_{0}/recon_past".format(seq_num)
        save_path0 = self.logger.get_dir() + "/Reconstruction/Sequence_{0}/obs".format(seq_num)
        os.makedirs(save_path, exist_ok=True)
        os.makedirs(save_path3, exist_ok=True)
        os.makedirs(save_path2, exist_ok=True)
        os.makedirs(save_path1, exist_ok=True)
        os.makedirs(save_path0, exist_ok=True)
        for i in range(10, int(shape[0]/n_seq)-10):
            # save each sequence images to a seperate folder
            save_image(obs[i], save_path0 + "/obs_{0:05d}.png".format(i))
            save_image(recon_current_plot[i], save_path2 + "/recon_current_{0:05d}.png".format(i))
            # save_image(recon_next_plot[i-self.reconstruction_steps], save_path3 + "/recon_future_{0:05d}.png".format(i))
            save_image(recon_previous_plot[i+self.reconstruction_steps], save_path1 + "/recon_past_{0:05d}.png".format(i))

    
    def save_lstm_rollout(self, iteration):
        self.policy.set_training_mode(False)
        for rollout_data in self.rollout_buffer.get(self.batch_size):
            save_path = self.logger.get_dir()
            os.makedirs(save_path, exist_ok=True)
            th.save(rollout_data, save_path + "/rollout_{0:05d}.pth".format(iteration))

    def lstm_loss_function(self, obs, obs_recon, n_seq, epoch, use_log_depth=False):
        if isinstance(obs, dict):
            obs = obs['image'].float() / 255.0
        else:
            obs = obs.float() / 255.0
        obs_shape = obs.shape
        BCE = 0
        future_loss = 0
        if use_log_depth:
            obs = self.log_depth(obs, bound_min=True)
        if self.reconstruction_members[0]:
            if use_log_depth:
                obs_recon[0] = self.log_depth(obs_recon[0], bound_min=True)
            BCE += F.mse_loss(obs_recon[0], th.flatten(obs.reshape((n_seq, -1) + obs_shape[1:])[:, :-self.reconstruction_steps, :], 
                                                       start_dim=0, end_dim=1), reduction='sum')
        if self.reconstruction_members[1]:
            if use_log_depth:
                obs_recon[1] = self.log_depth(obs_recon[1], bound_min=True)
            BCE += F.mse_loss(obs_recon[1], obs, reduction='sum')

        if self.reconstruction_members[2]:
            if use_log_depth:
                obs_recon[2] = self.log_depth(obs_recon[2], bound_min=True)
            future_loss = F.mse_loss(obs_recon[2], th.flatten(obs.reshape((n_seq, -1) + obs_shape[1:])[:, self.reconstruction_steps:, :], 
                                                       start_dim=0, end_dim=1), reduction='sum')
            BCE += future_loss
            future_loss = future_loss.item()
        return BCE, future_loss
    
    def lstm_test_loss_std(self, obs, obs_recon, n_seq):
        if isinstance(obs, dict):
            obs = obs['image'].float() / 255.0
        else:
            obs = obs.float() / 255.0
        obs_shape = obs.shape
        print(obs_recon[0].shape)
        BCE = 0
        if self.reconstruction_members[0]:
            diff_0 = th.abs(obs_recon[0] - th.flatten(obs.reshape((n_seq, -1) + obs_shape[1:])[:, :-self.reconstruction_steps, :], 
                                                       start_dim=0, end_dim=1)) * 255.0
            diff_0 = th.sum(diff_0, dim=(1, 2, 3)) / (obs_shape[-2] * obs_shape[-1])
            BCE += th.mean(diff_0)
            std = th.std(diff_0)
        if self.reconstruction_members[1]:
            diff_1 = th.abs(obs_recon[1] - obs) * 255.0
            diff_1 = th.sum(diff_1, dim=(1, 2, 3)) / (obs_shape[-2] * obs_shape[-1])
            BCE += th.mean(diff_1)
            std = th.std(diff_1)
        if self.reconstruction_members[2]:
            diff_2 = th.abs(obs_recon[2] - th.flatten(obs.reshape((n_seq, -1) + obs_shape[1:])[:, self.reconstruction_steps:, :], 
                                                       start_dim=0, end_dim=1)) * 255.0
            diff_2 = th.sum(diff_2, dim=(1, 2, 3)) / (obs_shape[-2] * obs_shape[-1])
            BCE += th.mean(diff_2)
            std = th.std(diff_2)
        return BCE.item(), std.item()

    def learn_lstm(
        self: SelfRecurrentPPO,
        total_timesteps: int,
        callback: MaybeCallback = None,
        log_interval: Tuple = (10, 10),
        eval_env: Optional[GymEnv] = None,
        eval_freq: int = -1,
        progress_bar: bool = True,
        n_eval_episodes: int = 5,
        tb_log_name: str = "RecurrentPPO",
        eval_log_path: Optional[str] = None,
        reset_num_timesteps: bool = True,
    ):
        iteration = 0
        total_timesteps, callback = self._setup_learn(
            total_timesteps,
            callback,
            reset_num_timesteps,
            tb_log_name,
            progress_bar
        )
        self._last_episode_starts = th.ones((self.env.num_envs,), dtype=th.float32, device=self.device)

        callback.on_training_start(locals(), globals())

        while self.num_timesteps < total_timesteps:

            continue_training = self.collect_lstm_rollouts(self.env, callback, self.rollout_buffer, n_rollout_steps=self.n_steps, deterministic=True)

            if continue_training is False:
                break
            if not self.save_lstm_dataset:
                iteration += 1
                self._update_current_progress_remaining(self.num_timesteps, total_timesteps)
                ave_loss, record_loss = self.train_lstm()
                print("average loss: ", ave_loss)
                self.logger.record("train/future_loss", record_loss)
                self.logger.dump(step=self.num_timesteps)
                if log_interval is not None and iteration % log_interval[1] == 0:
                    policy_path = self.logger.get_dir() + "/Policy"
                    os.makedirs(policy_path, exist_ok=True)
                    self.policy.save(policy_path + "/iter_{0:05d}.pth".format(iteration))
                    self.eval_lstm(iteration)

            else:
                iteration += 1
                self.save_lstm_rollout(iteration)

            callback.on_training_end()


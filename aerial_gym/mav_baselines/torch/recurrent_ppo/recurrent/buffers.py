from functools import partial
from typing import Callable, Generator, Optional, Tuple, Union, Dict

import numpy as np
import torch as th
from gym import spaces
from stable_baselines3.common.buffers import DictRolloutBuffer, RolloutBuffer, BaseBuffer
from stable_baselines3.common.vec_env import VecNormalize
from stable_baselines3.common.utils import obs_as_tensor
from aerial_gym.mav_baselines.torch.recurrent_ppo.recurrent.type_aliases import (
    RecurrentDictRolloutBufferSamples,
    LSTMDictRolloutBufferSamples,
    InputLSTMRolloutBufferSamples,
    RecurrentRolloutBufferSamples,
    RNNStates,
    LatentLSTMRolloutBufferSamples,
    DistLatentLSTMRolloutBufferSamples,
)
from stable_baselines3.common.type_aliases import RolloutBufferSamples

def pad(
    seq_start_indices: np.ndarray,
    seq_end_indices: np.ndarray,
    device: th.device,
    tensor: np.ndarray,
    padding_value: float = 0.0,
) -> th.Tensor:
    """
    Chunk sequences and pad them to have constant dimensions.

    :param seq_start_indices: Indices of the transitions that start a sequence
    :param seq_end_indices: Indices of the transitions that end a sequence
    :param device: PyTorch device
    :param tensor: Tensor of shape (batch_size, *tensor_shape)
    :param padding_value: Value used to pad sequence to the same length
        (zero padding by default)
    :return: (n_seq, max_length, *tensor_shape)
    """
    # Create sequences given start and end
    seq = [th.tensor(tensor[start : end + 1], device=device) for start, end in zip(seq_start_indices, seq_end_indices)]
    if(len(seq) == 0):
        print(seq_start_indices)
        print(seq_end_indices)
    return th.nn.utils.rnn.pad_sequence(seq, batch_first=True, padding_value=padding_value)

def pad_th(
    seq_start_indices: th.Tensor,
    seq_end_indices: th.Tensor,
    device: th.device,
    tensor: th.Tensor,
    padding_value: float = 0.0,
) -> th.Tensor:
    """
    Chunk sequences and pad them to have constant dimensions.

    :param seq_start_indices: Indices of the transitions that start a sequence
    :param seq_end_indices: Indices of the transitions that end a sequence
    :param device: PyTorch device
    :param tensor: Tensor of shape (batch_size, *tensor_shape)
    :param padding_value: Value used to pad sequence to the same length
        (zero padding by default)
    :return: (n_seq, max_length, *tensor_shape)
    """
    # Create sequences given start and end
    seq = [tensor[start : end + 1] for start, end in zip(seq_start_indices, seq_end_indices)]
    return th.nn.utils.rnn.pad_sequence(seq, batch_first=True, padding_value=padding_value).requires_grad_(True)


def pad_and_flatten(
    seq_start_indices: np.ndarray,
    seq_end_indices: np.ndarray,
    device: th.device,
    tensor: np.ndarray,
    padding_value: float = 0.0,
) -> th.Tensor:
    """
    Pad and flatten the sequences of scalar values,
    while keeping the sequence order.
    From (batch_size, 1) to (n_seq, max_length, 1) -> (n_seq * max_length,)

    :param seq_start_indices: Indices of the transitions that start a sequence
    :param seq_end_indices: Indices of the transitions that end a sequence
    :param device: PyTorch device (cpu, gpu, ...)
    :param tensor: Tensor of shape (max_length, n_seq, 1)
    :param padding_value: Value used to pad sequence to the same length
        (zero padding by default)
    :return: (n_seq * max_length,) aka (padded_batch_size,)
    """
    return pad(seq_start_indices, seq_end_indices, device, tensor, padding_value).flatten()

def pad_and_flatten_th(
    seq_start_indices: th.Tensor,
    seq_end_indices: th.Tensor,
    device: th.device,
    tensor: th.Tensor,
    padding_value: float = 0.0,
) -> th.Tensor:
    """
    Pad and flatten the sequences of scalar values,
    while keeping the sequence order.
    From (batch_size, 1) to (n_seq, max_length, 1) -> (n_seq * max_length,)

    :param seq_start_indices: Indices of the transitions that start a sequence
    :param seq_end_indices: Indices of the transitions that end a sequence
    :param device: PyTorch device (cpu, gpu, ...)
    :param tensor: Tensor of shape (max_length, n_seq, 1)
    :param padding_value: Value used to pad sequence to the same length
        (zero padding by default)
    :return: (n_seq * max_length,) aka (padded_batch_size,)
    """
    return pad_th(seq_start_indices, seq_end_indices, device, tensor, padding_value).flatten()


def create_sequencers(
    episode_starts: np.ndarray,
    env_change: np.ndarray,
    device: th.device,
) -> Tuple[np.ndarray, Callable, Callable]:
    """
    Create the utility function to chunk data into
    sequences and pad them to create fixed size tensors.

    :param episode_starts: Indices where an episode starts
    :param env_change: Indices where the data collected
        come from a different env (when using multiple env for data collection)
    :param device: PyTorch device
    :return: Indices of the transitions that start a sequence,
        pad and pad_and_flatten utilities tailored for this batch
        (sequence starts and ends indices are fixed)
    """
    # Create sequence if env changes too
    seq_start = np.logical_or(episode_starts, env_change).flatten()
    # First index is always the beginning of a sequence
    seq_start[0] = True
    # Retrieve indices of sequence starts
    seq_start_indices = np.where(seq_start == True)[0]  # noqa: E712
    # End of sequence are just before sequence starts
    # Last index is also always end of a sequence
    seq_end_indices = np.concatenate([(seq_start_indices - 1)[1:], np.array([len(episode_starts)])])
    # Create padding method for this minibatch
    # to avoid repeating arguments (seq_start_indices, seq_end_indices)
    local_pad = partial(pad, seq_start_indices, seq_end_indices, device)
    local_pad_and_flatten = partial(pad_and_flatten, seq_start_indices, seq_end_indices, device)
    return seq_start_indices, local_pad, local_pad_and_flatten

def create_sequencers_th(
        episode_starts: th.Tensor,
        env_change: th.Tensor,
        device: th.device,
) -> Tuple[th.Tensor, Callable, Callable]:
    """
    Create the utility function to chunk data into
    sequences and pad them to create fixed size tensors.

    :param episode_starts: Indices where an episode starts
    :param env_change: Indices where the data collected
        come from a different env (when using multiple env for data collection)
    :param device: PyTorch device
    :return: Indices of the transitions that start a sequence,
        pad and pad_and_flatten utilities tailored for this batch
        (sequence starts and ends indices are fixed)
    """
    # Create sequence if env changes too
    seq_start = th.logical_or(episode_starts, env_change).flatten()
    # First index is always the beginning of a sequence
    seq_start[0] = True
    # Retrieve indices of sequence starts
    seq_start_indices = th.where(seq_start == True)[0]  # noqa: E712
    # End of sequence are just before sequence starts
    # Last index is also always end of a sequence
    seq_end_indices = th.cat([(seq_start_indices - 1)[1:], th.tensor([len(episode_starts)], device=device)])
    # Create padding method for this minibatch
    # to avoid repeating arguments (seq_start_indices, seq_end_indices)
    local_pad = partial(pad_th, seq_start_indices, seq_end_indices, device)
    local_pad_and_flatten = partial(pad_and_flatten_th, seq_start_indices, seq_end_indices, device)
    return seq_start_indices, local_pad, local_pad_and_flatten

def creare_sequencers_th_with_minLength(
    episode_starts: th.Tensor,
    env_change: th.Tensor,
    device: th.device,
    minLength: int,
) -> Tuple[th.Tensor, Callable, Callable]:
    # Create sequence if env changes too
    # seq_start = np.logical_or(episode_starts, env_change).flatten()
    seq_start = th.logical_or(episode_starts, env_change).flatten()
    # First index is always the beginning of a sequence
    seq_start[0] = True
    # Retrieve indices of sequence starts
    # seq_start_indices = np.where(seq_start == True)[0]  # noqa: E712
    seq_start_indices = th.where(seq_start == True)[0]  # noqa: E712
    # End of sequence are just before sequence starts
    # Last index is also always end of a sequence
    # seq_end_indices = np.concatenate([(seq_start_indices - 1)[1:], np.array([len(episode_starts)-1])])
    seq_end_indices = th.cat([(seq_start_indices - 1)[1:], th.tensor([len(episode_starts)-1], device=device)])
    seq_start_indices_temp = []
    seq_end_indices_temp = []
    for i in range(seq_end_indices.shape[0]):
        if(seq_end_indices[i] - seq_start_indices[i] + 1 > minLength):
            seq_start_indices_temp.append(seq_start_indices[i])
            seq_end_indices_temp.append(seq_end_indices[i])
    seq_start_indices_temp = th.tensor(seq_start_indices_temp, device=device)
    seq_end_indices_temp = th.tensor(seq_end_indices_temp, device=device)
    local_pad = partial(pad_th, seq_start_indices_temp, seq_end_indices_temp, device)
    local_pad_and_flatten = partial(pad_and_flatten_th, seq_start_indices_temp, seq_end_indices_temp, device)
    return seq_start_indices_temp, local_pad, local_pad_and_flatten

def creare_sequencers_with_minLength(
    episode_starts: np.ndarray,
    env_change: np.ndarray,
    device: th.device,
    minLength: int,
) -> Tuple[np.ndarray, Callable, Callable]:
    # Create sequence if env changes too
    seq_start = np.logical_or(episode_starts, env_change).flatten()
    # First index is always the beginning of a sequence
    seq_start[0] = True
    # Retrieve indices of sequence starts
    seq_start_indices = np.where(seq_start == True)[0]  # noqa: E712
    # End of sequence are just before sequence starts
    # Last index is also always end of a sequence
    seq_end_indices = np.concatenate([(seq_start_indices - 1)[1:], np.array([len(episode_starts)-1])])
    seq_start_indices_temp = []
    seq_end_indices_temp = []
    for i in range(seq_end_indices.shape[0]):
        if(seq_end_indices[i] - seq_start_indices[i] + 1 > minLength):
            seq_start_indices_temp.append(seq_start_indices[i])
            seq_end_indices_temp.append(seq_end_indices[i])
    seq_start_indices_temp = np.array(seq_start_indices_temp)
    seq_end_indices_temp = np.array(seq_end_indices_temp)
    local_pad = partial(pad, seq_start_indices_temp, seq_end_indices_temp, device)
    local_pad_and_flatten = partial(pad_and_flatten, seq_start_indices_temp, seq_end_indices_temp, device)
    return seq_start_indices_temp, local_pad, local_pad_and_flatten

def create_double_sequencers(
    episode_starts: np.ndarray,
    env_change: np.ndarray,
    device: th.device,
) -> Tuple[np.ndarray, Callable, Callable]:
    """
    Create the utility function to chunk data into
    sequences and pad them to create fixed size tensors.

    :param episode_starts: Indices where an episode starts
    :param env_change: Indices where the data collected
        come from a different env (when using multiple env for data collection)
    :param device: PyTorch device
    :return: Indices of the transitions that start a sequence,
        pad and pad_and_flatten utilities tailored for this batch
        (sequence starts and ends indices are fixed)
    """
    # Create sequence if env changes too
    seq_start = np.logical_or(episode_starts, env_change).flatten()
    # First index is always the beginning of a sequence
    seq_start[0] = True
    # Retrieve indices of sequence starts
    seq_start_indices = np.where(seq_start == True)[0]  # noqa: E712
    next_seq_start_indices = seq_start_indices+1
    # End of sequence are just before sequence starts
    # Last index is also always end of a sequence
    next_seq_end_indices = np.concatenate([(seq_start_indices - 1)[1:], np.array([len(episode_starts)-1])])
    seq_end_indices = next_seq_end_indices-1
    # Create padding method for this minibatch
    # to avoid repeating arguments (seq_start_indices, seq_end_indices)
    local_pad = partial(pad, seq_start_indices, seq_end_indices, device)
    local_pad_and_flatten = partial(pad_and_flatten, seq_start_indices, seq_end_indices, device)
    next_local_pad = partial(pad, next_seq_start_indices, next_seq_end_indices, device)
    next_local_pad_and_flatten = partial(pad_and_flatten, next_seq_start_indices, next_seq_end_indices, device)
    return seq_start_indices, local_pad, local_pad_and_flatten, next_seq_start_indices, next_local_pad, next_local_pad_and_flatten


class RecurrentRolloutBuffer(RolloutBuffer):
    """
    Rollout buffer that also stores the LSTM cell and hidden states.

    :param buffer_size: Max number of element in the buffer
    :param observation_space: Observation space
    :param action_space: Action space
    :param hidden_state_shape: Shape of the buffer that will collect lstm states
        (n_steps, lstm.num_layers, n_envs, lstm.hidden_size)
    :param device: PyTorch device
    :param gae_lambda: Factor for trade-off of bias vs variance for Generalized Advantage Estimator
        Equivalent to classic advantage when set to 1.
    :param gamma: Discount factor
    :param n_envs: Number of parallel environments
    """

    def __init__(
        self,
        buffer_size: int,
        observation_space: spaces.Space,
        action_space: spaces.Space,
        hidden_state_shape: Tuple[int, int, int, int],
        device: Union[th.device, str] = "auto",
        gae_lambda: float = 1,
        gamma: float = 0.99,
        n_envs: int = 1,
    ):
        self.hidden_state_shape = hidden_state_shape
        self.seq_start_indices, self.seq_end_indices = None, None
        super().__init__(buffer_size, observation_space, action_space, device, gae_lambda, gamma, n_envs)

    def reset(self):
        super().reset()
        self.hidden_states_pi = np.zeros(self.hidden_state_shape, dtype=np.float32)
        self.cell_states_pi = np.zeros(self.hidden_state_shape, dtype=np.float32)
        self.hidden_states_vf = np.zeros(self.hidden_state_shape, dtype=np.float32)
        self.cell_states_vf = np.zeros(self.hidden_state_shape, dtype=np.float32)

    def add(self, *args, lstm_states: RNNStates, **kwargs) -> None:
        """
        :param hidden_states: LSTM cell and hidden state
        """
        self.hidden_states_pi[self.pos] = np.array(lstm_states.pi[0].cpu().numpy())
        self.cell_states_pi[self.pos] = np.array(lstm_states.pi[1].cpu().numpy())
        self.hidden_states_vf[self.pos] = np.array(lstm_states.vf[0].cpu().numpy())
        self.cell_states_vf[self.pos] = np.array(lstm_states.vf[1].cpu().numpy())

        super().add(*args, **kwargs)

    def get(self, batch_size: Optional[int] = None) -> Generator[RecurrentRolloutBufferSamples, None, None]:
        assert self.full, "Rollout buffer must be full before sampling from it"

        # Prepare the data
        if not self.generator_ready:
            # hidden_state_shape = (self.n_steps, lstm.num_layers, self.n_envs, lstm.hidden_size)
            # swap first to (self.n_steps, self.n_envs, lstm.num_layers, lstm.hidden_size)
            for tensor in ["hidden_states_pi", "cell_states_pi", "hidden_states_vf", "cell_states_vf"]:
                self.__dict__[tensor] = self.__dict__[tensor].swapaxes(1, 2)

            # flatten but keep the sequence order
            # 1. (n_steps, n_envs, *tensor_shape) -> (n_envs, n_steps, *tensor_shape)
            # 2. (n_envs, n_steps, *tensor_shape) -> (n_envs * n_steps, *tensor_shape)
            for tensor in [
                "observations",
                "actions",
                "values",
                "log_probs",
                "advantages",
                "returns",
                "hidden_states_pi",
                "cell_states_pi",
                "hidden_states_vf",
                "cell_states_vf",
                "episode_starts",
            ]:
                self.__dict__[tensor] = self.swap_and_flatten(self.__dict__[tensor])
            self.generator_ready = True

        # Return everything, don't create minibatches
        if batch_size is None:
            batch_size = self.buffer_size * self.n_envs

        # Sampling strategy that allows any mini batch size but requires
        # more complexity and use of padding
        # Trick to shuffle a bit: keep the sequence order
        # but split the indices in two
        split_index = np.random.randint(self.buffer_size * self.n_envs)
        indices = np.arange(self.buffer_size * self.n_envs)
        indices = np.concatenate((indices[split_index:], indices[:split_index]))

        env_change = np.zeros(self.buffer_size * self.n_envs).reshape(self.buffer_size, self.n_envs)
        # Flag first timestep as change of environment
        env_change[0, :] = 1.0
        env_change = self.swap_and_flatten(env_change)

        start_idx = 0
        while start_idx < self.buffer_size * self.n_envs:
            batch_inds = indices[start_idx : start_idx + batch_size]
            yield self._get_samples(batch_inds, env_change)
            start_idx += batch_size

    def _get_samples(
        self,
        batch_inds: np.ndarray,
        env_change: np.ndarray,
        env: Optional[VecNormalize] = None,
    ) -> RecurrentRolloutBufferSamples:
        # Retrieve sequence starts and utility function
        self.seq_start_indices, self.pad, self.pad_and_flatten = create_sequencers(
            self.episode_starts[batch_inds], env_change[batch_inds], self.device
        )

        # Number of sequences
        n_seq = len(self.seq_start_indices)
        max_length = self.pad(self.actions[batch_inds]).shape[1]
        padded_batch_size = n_seq * max_length
        # We retrieve the lstm hidden states that will allow
        # to properly initialize the LSTM at the beginning of each sequence
        lstm_states_pi = (
            # 1. (n_envs * n_steps, n_layers, dim) -> (batch_size, n_layers, dim)
            # 2. (batch_size, n_layers, dim)  -> (n_seq, n_layers, dim)
            # 3. (n_seq, n_layers, dim) -> (n_layers, n_seq, dim)
            self.hidden_states_pi[batch_inds][self.seq_start_indices].swapaxes(0, 1),
            self.cell_states_pi[batch_inds][self.seq_start_indices].swapaxes(0, 1),
        )
        lstm_states_vf = (
            # (n_envs * n_steps, n_layers, dim) -> (n_layers, n_seq, dim)
            self.hidden_states_vf[batch_inds][self.seq_start_indices].swapaxes(0, 1),
            self.cell_states_vf[batch_inds][self.seq_start_indices].swapaxes(0, 1),
        )
        lstm_states_pi = (self.to_torch(lstm_states_pi[0]).contiguous(), self.to_torch(lstm_states_pi[1]).contiguous())
        lstm_states_vf = (self.to_torch(lstm_states_vf[0]).contiguous(), self.to_torch(lstm_states_vf[1]).contiguous())

        return RecurrentRolloutBufferSamples(
            # (batch_size, obs_dim) -> (n_seq, max_length, obs_dim) -> (n_seq * max_length, obs_dim)
            observations=self.pad(self.observations[batch_inds]).reshape((padded_batch_size,) + self.obs_shape),
            actions=self.pad(self.actions[batch_inds]).reshape((padded_batch_size,) + self.actions.shape[1:]),
            old_values=self.pad_and_flatten(self.values[batch_inds]),
            old_log_prob=self.pad_and_flatten(self.log_probs[batch_inds]),
            advantages=self.pad_and_flatten(self.advantages[batch_inds]),
            returns=self.pad_and_flatten(self.returns[batch_inds]),
            lstm_states=RNNStates(lstm_states_pi, lstm_states_vf),
            episode_starts=self.pad_and_flatten(self.episode_starts[batch_inds]),
            mask=self.pad_and_flatten(np.ones_like(self.returns[batch_inds])),
        )


class LatentRolloutBuffer(RolloutBuffer):
    """
    Dict Rollout buffer used in on-policy algorithms like A2C/PPO.
    Extends the RolloutBuffer to use dictionary observations

    :param buffer_size: Max number of element in the buffer
    :param observation_space: Observation space
    :param action_space: Action space
    :param device: PyTorch device
    :param gae_lambda: Factor for trade-off of bias vs variance for Generalized Advantage Estimator
        Equivalent to classic advantage when set to 1.
    :param gamma: Discount factor
    :param n_envs: Number of parallel environments
    """

    def __init__(
        self,
        buffer_size: int,
        observation_space: spaces.Space,
        action_space: spaces.Space,
        hidden_state_shape: Tuple[int, int, int, int],
        device: Union[th.device, str] = "auto",
        gae_lambda: float = 1,
        gamma: float = 0.99,
        n_envs: int = 1,
        n_seq: int = 1,
        ppo_input_size: int = 256,
    ):
        self.hidden_state_shape = hidden_state_shape
        self.seq_start_indices, self.seq_end_indices = None, None
        self.n_seq = n_seq
        self.ppo_input_size = ppo_input_size
        super().__init__(buffer_size, observation_space, action_space, device, gae_lambda, gamma, n_envs=n_envs)
    
    def reset(self):
        self.observations = th.zeros((self.buffer_size, self.n_envs) + self.obs_shape, device=self.device)
        self.actions = th.zeros((self.buffer_size, self.n_envs, self.action_dim), device=self.device)
        self.rewards = th.zeros((self.buffer_size, self.n_envs), device=self.device)
        self.returns = th.zeros((self.buffer_size, self.n_envs), device=self.device)
        self.episode_starts = th.zeros((self.buffer_size, self.n_envs), device=self.device)
        self.values = th.zeros((self.buffer_size, self.n_envs), device=self.device)
        self.log_probs = th.zeros((self.buffer_size, self.n_envs), device=self.device)
        self.advantages = th.zeros((self.buffer_size, self.n_envs), device=self.device)
        self.generator_ready = False
        self.pos = 0
        self.full = False

        self.hidden_states_pi = th.zeros(self.hidden_state_shape, device=self.device)
        self.cell_states_pi = th.zeros(self.hidden_state_shape, device=self.device)
        self.hidden_states_vf = th.zeros(self.hidden_state_shape, device=self.device)
        self.cell_states_vf = th.zeros(self.hidden_state_shape, device=self.device)
        
        self.latent_lstm_pi = th.zeros((self.buffer_size, self.n_envs, self.ppo_input_size), device=self.device)
        self.latent_lstm_vf = th.zeros((self.buffer_size, self.n_envs, self.ppo_input_size), device=self.device)

    def add(self,
        obs: th.Tensor,
        action: th.Tensor,
        reward: th.Tensor,
        episode_start: th.Tensor,
        value: th.Tensor,
        log_prob: th.Tensor, lstm_states: RNNStates, latent_lstm_pi: th.Tensor, latent_lstm_vf: th.Tensor, **kwargs) -> None:
        """
        :param hidden_states: LSTM cell and hidden state
        """
        self.hidden_states_pi[self.pos] = lstm_states.pi[0].clone().detach()
        self.cell_states_pi[self.pos] = lstm_states.pi[1].clone().detach()
        self.hidden_states_vf[self.pos] = lstm_states.vf[0].clone().detach()
        self.cell_states_vf[self.pos] = lstm_states.vf[1].clone().detach()

        self.latent_lstm_pi[self.pos] = latent_lstm_pi.clone().detach()
        self.latent_lstm_vf[self.pos] = latent_lstm_vf.clone().detach()

        if len(log_prob.shape) == 0:
            # Reshape 0-d tensor to avoid error
            log_prob = log_prob.reshape(-1, 1)

        # Reshape needed when using multiple envs with discrete observations
        # as numpy cannot broadcast (n_discrete,) to (n_discrete, 1)
        if isinstance(self.observation_space, spaces.Discrete):
            obs = obs.reshape((self.n_envs,) + self.obs_shape)

        self.observations[self.pos] = obs.clone().detach()
        self.actions[self.pos] = action.clone().detach()
        self.rewards[self.pos] = reward.clone().detach()
        self.episode_starts[self.pos] = episode_start.clone().detach()
        self.values[self.pos] = value.clone().detach().flatten()
        self.log_probs[self.pos] = log_prob.clone().detach()
        self.pos += 1
        if self.pos == self.buffer_size:
            self.full = True

    def get(self, batch_size: Optional[int] = None) -> Generator[LatentLSTMRolloutBufferSamples, None, None]:
        assert self.full, "Rollout buffer must be full before sampling from it"

        # Prepare the data
        if not self.generator_ready:
            for tensor in ["hidden_states_pi", "cell_states_pi", "hidden_states_vf", "cell_states_vf"]:
                self.__dict__[tensor] = self.__dict__[tensor].swapaxes(1, 2)

            for tensor in [
                "observations",
                "latent_lstm_pi",
                "latent_lstm_vf",
                "actions",
                "values",
                "log_probs",
                "advantages",
                "returns",
                "hidden_states_pi",
                "cell_states_pi",
                "hidden_states_vf",
                "cell_states_vf",
                "episode_starts",
            ]:
                self.__dict__[tensor] = self.swap_and_flatten_th(self.__dict__[tensor])
            self.generator_ready = True

        # Return everything, don't create minibatches
        if batch_size is None:
            batch_size = self.buffer_size * self.n_envs

        # Trick to shuffle a bit: keep the sequence order
        # but split the indices in two
        split_index = th.randint(0, self.buffer_size * self.n_envs, (1,), device=self.device, requires_grad=False).item()
        indices = th.arange(self.buffer_size * self.n_envs, device=self.device, requires_grad=False)
        indices = th.cat((indices[split_index:], indices[:split_index]))

        env_change = th.zeros(self.buffer_size * self.n_envs, device=self.device).reshape(self.buffer_size, self.n_envs)
        # Flag first timestep as change of environment
        env_change[0, :] = 1.0
        env_change = self.swap_and_flatten_th(env_change)

        start_idx = 0
        while start_idx < self.buffer_size * self.n_envs:
            batch_inds = indices[start_idx : start_idx + batch_size]
            yield self._get_samples(batch_inds, env_change)
            start_idx += batch_size

    def _get_samples(
        self,
        batch_inds: th.Tensor,
        env_change: th.Tensor,
        env: Optional[VecNormalize] = None,
    ) -> LatentLSTMRolloutBufferSamples:
        # Retrieve sequence starts and utility function
        self.seq_start_indices, self.pad_th, self.pad_and_flatten_th = create_sequencers_th(
            self.episode_starts[batch_inds], env_change[batch_inds], self.device
        )

        # Number of sequences
        n_seq = len(self.seq_start_indices)
        max_length = self.pad_th(self.actions[batch_inds]).shape[1]
        padded_batch_size = n_seq * max_length
        # We retrieve the lstm hidden states that will allow
        # to properly initialize the LSTM at the beginning of each sequence
        lstm_states_pi = (
            # 1. (n_envs * n_steps, n_layers, dim) -> (batch_size, n_layers, dim)
            # 2. (batch_size, n_layers, dim)  -> (n_seq, n_layers, dim)
            # 3. (n_seq, n_layers, dim) -> (n_layers, n_seq, dim)
            self.hidden_states_pi[batch_inds][self.seq_start_indices].swapaxes(0, 1),
            self.cell_states_pi[batch_inds][self.seq_start_indices].swapaxes(0, 1),
        )
        lstm_states_vf = (
            # (n_envs * n_steps, n_layers, dim) -> (n_layers, n_seq, dim)
            self.hidden_states_vf[batch_inds][self.seq_start_indices].swapaxes(0, 1),
            self.cell_states_vf[batch_inds][self.seq_start_indices].swapaxes(0, 1),
        )
        lstm_states_pi = (lstm_states_pi[0].contiguous(), lstm_states_pi[1].contiguous())
        lstm_states_vf = (lstm_states_vf[0].contiguous(), lstm_states_vf[1].contiguous())

        return LatentLSTMRolloutBufferSamples(
            # (batch_size, obs_dim) -> (n_seq, max_length, obs_dim) -> (n_seq * max_length, obs_dim)
            observations=self.pad_th(self.observations[batch_inds]).reshape((padded_batch_size,) + self.obs_shape),
            latent_lstm_pi=self.pad_th(self.latent_lstm_pi[batch_inds]).reshape((padded_batch_size, self.ppo_input_size)),
            latent_lstm_vf=self.pad_th(self.latent_lstm_vf[batch_inds]).reshape((padded_batch_size, self.ppo_input_size)),
            actions=self.pad_th(self.actions[batch_inds]).reshape((padded_batch_size,) + self.actions.shape[1:]),
            old_values=self.pad_and_flatten_th(self.values[batch_inds]),
            old_log_prob=self.pad_and_flatten_th(self.log_probs[batch_inds]),
            advantages=self.pad_and_flatten_th(self.advantages[batch_inds]),
            returns=self.pad_and_flatten_th(self.returns[batch_inds]),
            lstm_states=RNNStates(lstm_states_pi, lstm_states_vf),
            episode_starts=self.pad_and_flatten_th(self.episode_starts[batch_inds]),
            mask=self.pad_and_flatten_th(th.ones_like(self.returns[batch_inds])),
        )

    def get_ppo_need(self, batch_size: Optional[int] = None) -> Generator[InputLSTMRolloutBufferSamples, None, None]:
        assert self.full, ""
        indices = np.random.permutation(self.buffer_size * self.n_envs)
        indices = th.tensor(indices, device=self.device, requires_grad=False)
        # Prepare the data
        if not self.generator_ready:

            _tensor_names = [
                "latent_lstm_pi",
                "latent_lstm_vf",
                "actions",
                "values",
                "log_probs",
                "advantages",
                "returns",
            ]
            for tensor in _tensor_names:
                self.__dict__[tensor] = self.swap_and_flatten_th(self.__dict__[tensor])
            self.generator_ready = True

        # Return everything, don't create minibatches
        if batch_size is None:
            batch_size = self.buffer_size * self.n_envs

        start_idx = 0
        while start_idx < self.buffer_size * self.n_envs:
            yield self._get_ppo_need_samples(indices[start_idx : start_idx + batch_size])
            start_idx += batch_size

    def _get_ppo_need_samples(self, batch_inds: th.Tensor, env: Optional[VecNormalize] = None) -> InputLSTMRolloutBufferSamples:
        return InputLSTMRolloutBufferSamples(
            self.latent_lstm_pi[batch_inds],
            self.latent_lstm_vf[batch_inds],
            self.actions[batch_inds],
            self.values[batch_inds].flatten(),
            self.log_probs[batch_inds].flatten(),
            self.advantages[batch_inds].flatten(),
            self.returns[batch_inds].flatten(),
        )
    
    def compute_returns_and_advantage(self, last_values: th.Tensor, dones: th.Tensor) -> None:
        """
        Post-processing step: compute the lambda-return (TD(lambda) estimate)
        and GAE(lambda) advantage.

        Uses Generalized Advantage Estimation (https://arxiv.org/abs/1506.02438)
        to compute the advantage. To obtain Monte-Carlo advantage estimate (A(s) = R - V(S))
        where R is the sum of discounted reward with value bootstrap
        (because we don't always have full episode), set ``gae_lambda=1.0`` during initialization.

        The TD(lambda) estimator has also two special cases:
        - TD(1) is Monte-Carlo estimate (sum of discounted rewards)
        - TD(0) is one-step estimate with bootstrapping (r_t + gamma * v(s_{t+1}))

        For more information, see discussion in https://github.com/DLR-RM/stable-baselines3/pull/375.

        :param last_values: state value estimation for the last step (one for each env)
        :param dones: if the last step was a terminal step (one bool for each env).
        """

        last_gae_lam = 0
        for step in reversed(range(self.buffer_size)):
            if step == self.buffer_size - 1:
                next_non_terminal = 1.0 - dones
                next_values = last_values
            else:
                next_non_terminal = 1.0 - self.episode_starts[step + 1]
                next_values = self.values[step + 1]
            delta = self.rewards[step] + self.gamma * next_values * next_non_terminal - self.values[step]
            last_gae_lam = delta + self.gamma * self.gae_lambda * next_non_terminal * last_gae_lam
            self.advantages[step] = last_gae_lam
        # TD(lambda) estimator, see Github PR #375 or "Telescoping in TD(lambda)"
        # in David Silver Lecture 4: https://www.youtube.com/watch?v=PnHCvfgC_ZA
        self.returns = self.advantages + self.values

    def swap_and_flatten_th(self, arr: th.Tensor) -> th.Tensor:
        """
        Swap and then flatten axes 0 (buffer_size) and 1 (n_envs)
        to convert shape from [n_steps, n_envs, ...] (when ... is the shape of the features)
        to [n_steps * n_envs, ...] (which maintain the order)

        :param arr:
        :return:
        """
        shape = arr.shape
        if len(shape) < 3:
            shape = shape + (1,)
        return arr.swapaxes(0, 1).reshape(shape[0] * shape[1], *shape[2:])
    
class TorchRolloutBuffer(BaseBuffer):
    """
    Rollout buffer used in on-policy algorithms like A2C/PPO.
    It corresponds to ``buffer_size`` transitions collected
    using the current policy.
    This experience will be discarded after the policy update.
    In order to use PPO objective, we also store the current value of each state
    and the log probability of each taken action.

    The term rollout here refers to the model-free notion and should not
    be used with the concept of rollout used in model-based RL or planning.
    Hence, it is only involved in policy and value function training but not action selection.

    :param buffer_size: Max number of element in the buffer
    :param observation_space: Observation space
    :param action_space: Action space
    :param device: PyTorch device
    :param gae_lambda: Factor for trade-off of bias vs variance for Generalized Advantage Estimator
        Equivalent to classic advantage when set to 1.
    :param gamma: Discount factor
    :param n_envs: Number of parallel environments
    """

    observations: th.Tensor
    actions: th.Tensor
    rewards: th.Tensor
    advantages: th.Tensor
    returns: th.Tensor
    episode_starts: th.Tensor
    log_probs: th.Tensor
    values: th.Tensor

    def __init__(
        self,
        buffer_size: int,
        observation_space: spaces.Space,
        action_space: spaces.Space,
        device: Union[th.device, str] = "auto",
        gae_lambda: float = 1,
        gamma: float = 0.99,
        n_envs: int = 1,
    ):
        super(TorchRolloutBuffer, self).__init__(
            buffer_size, observation_space, action_space, device, n_envs=n_envs
        )
        self.gae_lambda = gae_lambda
        self.gamma = gamma
        self.generator_ready = False
        self.reset()

    def reset(self) -> None:
        self.observations = th.zeros((self.buffer_size, self.n_envs) + self.obs_shape, device=self.device)
        self.actions = th.zeros((self.buffer_size, self.n_envs, self.action_dim), device=self.device)
        self.rewards = th.zeros((self.buffer_size, self.n_envs), device=self.device)
        self.returns = th.zeros((self.buffer_size, self.n_envs), device=self.device)
        self.episode_starts = th.zeros((self.buffer_size, self.n_envs), device=self.device)
        self.values = th.zeros((self.buffer_size, self.n_envs), device=self.device)
        self.log_probs = th.zeros((self.buffer_size, self.n_envs), device=self.device)
        self.advantages = th.zeros((self.buffer_size, self.n_envs), device=self.device)
        self.generator_ready = False
        super().reset()

    
    def compute_returns_and_advantage(self, last_values: th.Tensor, dones: th.Tensor) -> None:
        """
        Post-processing step: compute the lambda-return (TD(lambda) estimate)
        and GAE(lambda) advantage.

        Uses Generalized Advantage Estimation (https://arxiv.org/abs/1506.02438)
        to compute the advantage. To obtain Monte-Carlo advantage estimate (A(s) = R - V(S))
        where R is the sum of discounted reward with value bootstrap
        (because we don't always have full episode), set ``gae_lambda=1.0`` during initialization.

        The TD(lambda) estimator has also two special cases:
        - TD(1) is Monte-Carlo estimate (sum of discounted rewards)
        - TD(0) is one-step estimate with bootstrapping (r_t + gamma * v(s_{t+1}))

        For more information, see discussion in https://github.com/DLR-RM/stable-baselines3/pull/375.

        :param last_values: state value estimation for the last step (one for each env)
        :param dones: if the last step was a terminal step (one bool for each env).
        """

        last_gae_lam = 0
        for step in reversed(range(self.buffer_size)):
            if step == self.buffer_size - 1:
                next_non_terminal = 1.0 - dones
                next_values = last_values
            else:
                next_non_terminal = 1.0 - self.episode_starts[step + 1]
                next_values = self.values[step + 1]
            delta = self.rewards[step] + self.gamma * next_values * next_non_terminal - self.values[step]
            last_gae_lam = delta + self.gamma * self.gae_lambda * next_non_terminal * last_gae_lam
            self.advantages[step] = last_gae_lam
        # TD(lambda) estimator, see Github PR #375 or "Telescoping in TD(lambda)"
        # in David Silver Lecture 4: https://www.youtube.com/watch?v=PnHCvfgC_ZA
        self.returns = self.advantages + self.values

    def add(
        self,
        obs: th.Tensor,
        action: th.Tensor,
        reward: th.Tensor,
        episode_start: th.Tensor,
        value: th.Tensor,
        log_prob: th.Tensor,
    ) -> None:
        """
        :param obs: Observation
        :param action: Action
        :param reward:
        :param episode_start: Start of episode signal.
        :param value: estimated value of the current state
            following the current policy.
        :param log_prob: log probability of the action
            following the current policy.
        """
        if len(log_prob.shape) == 0:
            # Reshape 0-d tensor to avoid error
            log_prob = log_prob.reshape(-1, 1)

        # Reshape needed when using multiple envs with discrete observations
        # as numpy cannot broadcast (n_discrete,) to (n_discrete, 1)
        if isinstance(self.observation_space, spaces.Discrete):
            obs = obs.reshape((self.n_envs, *self.obs_shape))

        # Reshape to handle multi-dim and discrete action spaces, see GH #970 #1392
        action = action.reshape((self.n_envs, self.action_dim))

        self.observations[self.pos] = obs.clone().detach()
        self.actions[self.pos] = action.clone().detach()
        self.rewards[self.pos] = reward.clone().detach()
        self.episode_starts[self.pos] = episode_start.clone().detach()
        self.values[self.pos] = value.clone().detach().flatten()
        self.log_probs[self.pos] = log_prob.clone().detach()
        self.pos += 1
        if self.pos == self.buffer_size:
            self.full = True
    
    def get(self, batch_size: Optional[int] = None) -> Generator[RolloutBufferSamples, None, None]:
        assert self.full, "Rollout buffer must be full before sampling from it"
        indices = np.random.permutation(self.buffer_size * self.n_envs)
        indices = th.tensor(indices, device=self.device, requires_grad=False)
        # Prepare the data
        if not self.generator_ready:
            for tensor in [
                "observations",
                "actions",
                "values",
                "log_probs",
                "advantages",
                "returns",
            ]:
                self.__dict__[tensor] = self.swap_and_flatten_th(self.__dict__[tensor])
            self.generator_ready = True

        # Return everything, don't create minibatches
        if batch_size is None:
            batch_size = self.buffer_size * self.n_envs

        start_idx = 0
        while start_idx < self.buffer_size * self.n_envs:
            yield self._get_samples(indices[start_idx : start_idx + batch_size])
            start_idx += batch_size
            
    def swap_and_flatten_th(self, arr: th.Tensor) -> th.Tensor:
        """
        Swap and then flatten axes 0 (buffer_size) and 1 (n_envs)
        to convert shape from [n_steps, n_envs, ...] (when ... is the shape of the features)
        to [n_steps * n_envs, ...] (which maintain the order)

        :param arr:
        :return:
        """
        shape = arr.shape
        if len(shape) < 3:
            shape = shape + (1,)
        return arr.swapaxes(0, 1).reshape(shape[0] * shape[1], *shape[2:])
    
    def _get_samples(self, batch_inds: th.Tensor, env: Optional[VecNormalize] = None) -> RolloutBufferSamples:
        return RolloutBufferSamples(
            self.observations[batch_inds],
            self.actions[batch_inds],
            self.values[batch_inds].flatten(),
            self.log_probs[batch_inds].flatten(),
            self.advantages[batch_inds].flatten(),
            self.returns[batch_inds].flatten(),
        )
    
class DistLatentRolloutBuffer(LatentRolloutBuffer):
    def reset(self):
        self.hidden_states_pi_std = th.zeros(self.hidden_state_shape, device=self.device)
        self.cell_states_pi_std = th.zeros(self.hidden_state_shape, device=self.device)
        self.hidden_states_vf_std = th.zeros(self.hidden_state_shape, device=self.device)
        self.cell_states_vf_std = th.zeros(self.hidden_state_shape, device=self.device)
        super().reset()

    def add(self,
        obs: th.Tensor,
        action: th.Tensor,
        reward: th.Tensor,
        episode_start: th.Tensor,
        value: th.Tensor,
        log_prob: th.Tensor, lstm_states: RNNStates, lstm_states_std: RNNStates,
        latent_lstm_pi: th.Tensor, latent_lstm_vf: th.Tensor, **kwargs) -> None:
        """
        :param hidden_states: LSTM cell and hidden state
        """
        self.hidden_states_pi_std[self.pos] = lstm_states_std.pi[0].clone().detach()
        self.cell_states_pi_std[self.pos] = lstm_states_std.pi[1].clone().detach()
        self.hidden_states_vf_std[self.pos] = lstm_states_std.vf[0].clone().detach()
        self.cell_states_vf_std[self.pos] = lstm_states_std.vf[1].clone().detach()
        super().add(obs, action, reward, episode_start, value, log_prob, lstm_states, latent_lstm_pi, latent_lstm_vf, **kwargs)

class RecurrentDictRolloutBuffer(DictRolloutBuffer):
    """
    Dict Rollout buffer used in on-policy algorithms like A2C/PPO.
    Extends the RecurrentRolloutBuffer to use dictionary observations

    :param buffer_size: Max number of element in the buffer
    :param observation_space: Observation space
    :param action_space: Action space
    :param hidden_state_shape: Shape of the buffer that will collect lstm states
    :param device: PyTorch device
    :param gae_lambda: Factor for trade-off of bias vs variance for Generalized Advantage Estimator
        Equivalent to classic advantage when set to 1.
    :param gamma: Discount factor
    :param n_envs: Number of parallel environments
    """

    def __init__(
        self,
        buffer_size: int,
        observation_space: spaces.Space,
        action_space: spaces.Space,
        hidden_state_shape: Tuple[int, int, int, int],
        device: Union[th.device, str] = "auto",
        gae_lambda: float = 1,
        gamma: float = 0.99,
        n_envs: int = 1,
        n_seq: int = 1,
        ppo_input_size: int = 256,
    ):
        self.hidden_state_shape = hidden_state_shape
        self.seq_start_indices, self.seq_end_indices = None, None
        self.n_seq = n_seq
        self.ppo_input_size = ppo_input_size
        super().__init__(buffer_size, observation_space, action_space, device, gae_lambda, gamma, n_envs=n_envs)

    def reset(self):
        assert isinstance(self.obs_shape, dict), "DictRolloutBuffer must be used with Dict obs space only"
        self.observations = {}
        for key, obs_input_shape in self.obs_shape.items():
            self.observations[key] = np.zeros((self.buffer_size, self.n_envs) + obs_input_shape, dtype=np.float32)
        self.actions = np.zeros((self.buffer_size, self.n_envs, self.action_dim), dtype=np.float32)
        self.rewards = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
        self.returns = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
        self.episode_starts = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
        self.values = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
        self.log_probs = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
        self.advantages = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
        self.generator_ready = False
        self.pos = 0
        self.full = False

        self.hidden_states_pi = np.zeros(self.hidden_state_shape, dtype=np.float32)
        self.cell_states_pi = np.zeros(self.hidden_state_shape, dtype=np.float32)
        self.hidden_states_vf = np.zeros(self.hidden_state_shape, dtype=np.float32)
        self.cell_states_vf = np.zeros(self.hidden_state_shape, dtype=np.float32)
        
        self.latent_lstm_pi = np.zeros((self.buffer_size, self.n_envs, self.ppo_input_size), dtype=np.float32)
        self.latent_lstm_vf = np.zeros((self.buffer_size, self.n_envs, self.ppo_input_size), dtype=np.float32)

    def add(self, *args, lstm_states: RNNStates, latent_lstm_pi: th.Tensor, latent_lstm_vf: th.Tensor, **kwargs) -> None:
        """
        :param hidden_states: LSTM cell and hidden state
        """
        self.hidden_states_pi[self.pos] = np.array(lstm_states.pi[0].cpu().numpy())
        self.cell_states_pi[self.pos] = np.array(lstm_states.pi[1].cpu().numpy())
        self.hidden_states_vf[self.pos] = np.array(lstm_states.vf[0].cpu().numpy())
        self.cell_states_vf[self.pos] = np.array(lstm_states.vf[1].cpu().numpy())

        self.latent_lstm_pi[self.pos] = np.array(latent_lstm_pi.cpu().numpy())
        self.latent_lstm_vf[self.pos] = np.array(latent_lstm_vf.cpu().numpy())

        super().add(*args, **kwargs)

    def get(self, batch_size: Optional[int] = None) -> Generator[RecurrentDictRolloutBufferSamples, None, None]:
        assert self.full, "Rollout buffer must be full before sampling from it"

        # Prepare the data
        if not self.generator_ready:
            # hidden_state_shape = (self.n_steps, lstm.num_layers, self.n_envs, lstm.hidden_size)
            # swap first to (self.n_steps, self.n_envs, lstm.num_layers, lstm.hidden_size)
            for tensor in ["hidden_states_pi", "cell_states_pi", "hidden_states_vf", "cell_states_vf"]:
                self.__dict__[tensor] = self.__dict__[tensor].swapaxes(1, 2)

            for key, obs in self.observations.items():
                self.observations[key] = self.swap_and_flatten(obs)

            for tensor in [
                "actions",
                "values",
                "log_probs",
                "advantages",
                "returns",
                "hidden_states_pi",
                "cell_states_pi",
                "hidden_states_vf",
                "cell_states_vf",
                "episode_starts",
            ]:
                self.__dict__[tensor] = self.swap_and_flatten(self.__dict__[tensor])
            self.generator_ready = True

        # Return everything, don't create minibatches
        if batch_size is None:
            batch_size = self.buffer_size * self.n_envs

        # Trick to shuffle a bit: keep the sequence order
        # but split the indices in two
        split_index = np.random.randint(self.buffer_size * self.n_envs)
        indices = np.arange(self.buffer_size * self.n_envs)
        indices = np.concatenate((indices[split_index:], indices[:split_index]))

        env_change = np.zeros(self.buffer_size * self.n_envs).reshape(self.buffer_size, self.n_envs)
        # Flag first timestep as change of environment
        env_change[0, :] = 1.0
        env_change = self.swap_and_flatten(env_change)

        start_idx = 0
        while start_idx < self.buffer_size * self.n_envs:
            batch_inds = indices[start_idx : start_idx + batch_size]
            yield self._get_samples(batch_inds, env_change)
            start_idx += batch_size

    def _get_samples(
        self,
        batch_inds: np.ndarray,
        env_change: np.ndarray,
        env: Optional[VecNormalize] = None,
    ) -> RecurrentDictRolloutBufferSamples:
        self.seq_start_indices, self.pad, self.pad_and_flatten = create_sequencers(
            self.episode_starts[batch_inds], env_change[batch_inds], self.device
        )

        n_seq = len(self.seq_start_indices)
        max_length = self.pad(self.actions[batch_inds]).shape[1]
        padded_batch_size = n_seq * max_length
        # We retrieve the lstm hidden states that will allow
        # to properly initialize the LSTM at the beginning of each sequence
        lstm_states_pi = (
            # (n_envs * n_steps, n_layers, dim) -> (n_layers, n_seq, dim)
            self.hidden_states_pi[batch_inds][self.seq_start_indices].swapaxes(0, 1),
            self.cell_states_pi[batch_inds][self.seq_start_indices].swapaxes(0, 1),
        )

        lstm_states_vf = (
            # (n_envs * n_steps, n_layers, dim) -> (n_layers, n_seq, dim)
            self.hidden_states_vf[batch_inds][self.seq_start_indices].swapaxes(0, 1),
            self.cell_states_vf[batch_inds][self.seq_start_indices].swapaxes(0, 1),
        )
        lstm_states_pi = (self.to_torch(lstm_states_pi[0]).contiguous(), self.to_torch(lstm_states_pi[1]).contiguous())
        lstm_states_vf = (self.to_torch(lstm_states_vf[0]).contiguous(), self.to_torch(lstm_states_vf[1]).contiguous())

        observations = {key: self.pad(obs[batch_inds]) for (key, obs) in self.observations.items()}
        observations = {key: obs.reshape((padded_batch_size,) + self.obs_shape[key]) for (key, obs) in observations.items()}
        # print("observations: ", observations['image'].shape)
        return RecurrentDictRolloutBufferSamples(
            observations=observations,
            actions=self.pad(self.actions[batch_inds]).reshape((padded_batch_size,) + self.actions.shape[1:]),
            old_values=self.pad_and_flatten(self.values[batch_inds]),
            old_log_prob=self.pad_and_flatten(self.log_probs[batch_inds]),
            advantages=self.pad_and_flatten(self.advantages[batch_inds]),
            returns=self.pad_and_flatten(self.returns[batch_inds]),
            lstm_states=RNNStates(lstm_states_pi, lstm_states_vf),
            episode_starts=self.pad_and_flatten(self.episode_starts[batch_inds]),
            mask=self.pad_and_flatten(np.ones_like(self.returns[batch_inds])),
        )

    def get_ppo_need(self, batch_size: Optional[int] = None) -> Generator[InputLSTMRolloutBufferSamples, None, None]:
        assert self.full, ""
        indices = np.random.permutation(self.buffer_size * self.n_envs)
        # Prepare the data
        if not self.generator_ready:

            _tensor_names = [
                "latent_lstm_pi",
                "latent_lstm_vf",
                "actions",
                "values",
                "log_probs",
                "advantages",
                "returns",
            ]
            for tensor in _tensor_names:
                self.__dict__[tensor] = self.swap_and_flatten(self.__dict__[tensor])
            self.generator_ready = True

        # Return everything, don't create minibatches
        if batch_size is None:
            batch_size = self.buffer_size * self.n_envs

        start_idx = 0
        while start_idx < self.buffer_size * self.n_envs:
            yield self._get_ppo_need_samples(indices[start_idx : start_idx + batch_size])
            start_idx += batch_size

    def _get_ppo_need_samples(self, batch_inds: np.ndarray, env: Optional[VecNormalize] = None) -> InputLSTMRolloutBufferSamples:
        data = (
            self.latent_lstm_pi[batch_inds],
            self.latent_lstm_vf[batch_inds],
            self.actions[batch_inds],
            self.values[batch_inds].flatten(),
            self.log_probs[batch_inds].flatten(),
            self.advantages[batch_inds].flatten(),
            self.returns[batch_inds].flatten(),
        )
        return InputLSTMRolloutBufferSamples(*tuple(map(self.to_torch, data)))

class LSTMThDictRolloutBuffer(DictRolloutBuffer):
    
    def __init__(
        self,
        buffer_size: int,
        observation_space: spaces.Space,
        action_space: spaces.Space,
        hidden_state_shape: Tuple[int, int, int, int],
        device: Union[th.device, str] = "auto",
        gae_lambda: float = 1,
        gamma: float = 0.99,
        n_envs: int = 1,
        n_seq: int = 1,
    ):
        self.hidden_state_shape = hidden_state_shape
        self.seq_start_indices, self.seq_end_indices = None, None
        self.next_seq_start_indices, self.next_seq_end_indices = None, None
        self.n_seq = n_seq
        super().__init__(buffer_size, observation_space, action_space, device, gae_lambda, gamma, n_envs=n_envs)

    def reset(self):
        assert isinstance(self.obs_shape, dict), "DictRolloutBuffer must be used with Dict obs space only"
        self.observations = {}
        for key, obs_input_shape in self.obs_shape.items():
            self.observations[key] = th.zeros((self.buffer_size, self.n_envs) + obs_input_shape, device=self.device)
        self.actions = th.zeros((self.buffer_size, self.n_envs, self.action_dim), device=self.device)
        self.rewards = th.zeros((self.buffer_size, self.n_envs), device=self.device)
        self.returns = th.zeros((self.buffer_size, self.n_envs), device=self.device)
        self.episode_starts = th.zeros((self.buffer_size, self.n_envs), device=self.device)
        self.values = th.zeros((self.buffer_size, self.n_envs), device=self.device)
        self.log_probs = th.zeros((self.buffer_size, self.n_envs), device=self.device)
        self.advantages = th.zeros((self.buffer_size, self.n_envs), device=self.device)
        self.generator_ready = False
        self.pos = 0
        self.full = False

        self.hidden_states_pi = th.zeros(self.hidden_state_shape, device=self.device)
        self.cell_states_pi = th.zeros(self.hidden_state_shape, device=self.device)
        self.hidden_states_vf = th.zeros(self.hidden_state_shape, device=self.device)
        self.cell_states_vf = th.zeros(self.hidden_state_shape, device=self.device)
        

    def add(self,
        obs: Dict[str, th.Tensor],
        action: th.Tensor,
        reward: th.Tensor,
        episode_start: th.Tensor,
        value: th.Tensor,
        log_prob: th.Tensor, lstm_states: RNNStates, **kwargs) -> None:
        """
        :param hidden_states: LSTM cell and hidden state
        """
        self.hidden_states_pi[self.pos] = lstm_states.pi[0].clone().detach()
        self.cell_states_pi[self.pos] = lstm_states.pi[1].clone().detach()
        self.hidden_states_vf[self.pos] = lstm_states.vf[0].clone().detach()
        self.cell_states_vf[self.pos] = lstm_states.vf[1].clone().detach()

        if len(log_prob.shape) == 0:
            # Reshape 0-d tensor to avoid error
            log_prob = log_prob.reshape(-1, 1)

        for key in self.observations.keys():
            # Reshape needed when using multiple envs with discrete observations
            # as numpy cannot broadcast (n_discrete,) to (n_discrete, 1)
            # if isinstance(self.observation_space.spaces[key], spaces.Discrete):
            #     obs_ = obs_.reshape((self.n_envs,) + self.obs_shape[key])
            self.observations[key][self.pos] = obs[key].clone().detach()

        self.actions[self.pos] = action.clone().detach()
        self.rewards[self.pos] = reward.clone().detach()
        self.episode_starts[self.pos] = episode_start.clone().detach()
        self.values[self.pos] = value.clone().detach().flatten()

        self.log_probs[self.pos] = log_prob.clone().detach()
        self.pos += 1
        if self.pos == self.buffer_size:
            self.full = True

    def get(self, batch_size: Optional[int] = None) -> Generator[LSTMDictRolloutBufferSamples, None, None]:
        assert self.full, "Rollout buffer must be full before sampling from it"

        # Prepare the data
        if not self.generator_ready:
            # hidden_state_shape = (self.n_steps, lstm.num_layers, self.n_envs, lstm.hidden_size)
            # swap first to (self.n_steps, self.n_envs, lstm.num_layers, lstm.hidden_size)
            for tensor in ["hidden_states_pi", "cell_states_pi", "hidden_states_vf", "cell_states_vf"]:
                self.__dict__[tensor] = self.__dict__[tensor].swapaxes(1, 2)

            for key, obs in self.observations.items():
                self.observations[key] = self.swap_and_flatten_th(obs)

            for tensor in [
                "actions",
                "values",
                "log_probs",
                "advantages",
                "returns",
                "hidden_states_pi",
                "cell_states_pi",
                "hidden_states_vf",
                "cell_states_vf",
                "episode_starts",
            ]:
                self.__dict__[tensor] = self.swap_and_flatten_th(self.__dict__[tensor])
            self.generator_ready = True

        # Return everything, don't create minibatches
        if batch_size is None:
            batch_size = self.buffer_size * self.n_envs

        # Trick to shuffle a bit: keep the sequence order
        # but split the indices in two
        # split_index = np.random.randint(self.buffer_size * self.n_envs)
        split_index = 0
        # indices = np.arange(self.buffer_size * self.n_envs)
        # indices = np.concatenate((indices[split_index:], indices[:split_index]))
        indices = th.arange(self.buffer_size * self.n_envs, device=self.device, requires_grad=False)
        indices = th.cat((indices[split_index:], indices[:split_index]))

        # env_change = np.zeros(self.buffer_size * self.n_envs).reshape(self.buffer_size, self.n_envs)
        env_change = th.zeros(self.buffer_size * self.n_envs, device=self.device).reshape(self.buffer_size, self.n_envs)
        # Flag first timestep as change of environment
        env_change[0, :] = 1.0
        env_change = self.swap_and_flatten_th(env_change)

        start_idx = 0
        while start_idx < self.buffer_size * self.n_envs:
            batch_inds = indices[start_idx : start_idx + batch_size]
            yield self._get_samples(batch_inds, env_change)
            start_idx += batch_size

    def _get_samples(
        self,
        batch_inds: th.Tensor,
        env_change: th.Tensor,
        env: Optional[VecNormalize] = None,
    ) -> LSTMDictRolloutBufferSamples:
        self.seq_start_indices, self.pad_th, self.pad_and_flatten_th = creare_sequencers_th_with_minLength(
            self.episode_starts[batch_inds], env_change[batch_inds], self.device, 30
        )
        # print("batch_inds: ", batch_inds)
        # print("self.pad: ", self.pad)

        n_seq = len(self.seq_start_indices)
        max_length = self.pad_th(self.actions[batch_inds]).shape[1]
        padded_batch_size = n_seq * max_length
        # print(n_seq, max_length)
        # We retrieve the lstm hidden states that will allow
        # to properly initialize the LSTM at the beginning of each sequence
        lstm_states_pi = (
            # (n_envs * n_steps, n_layers, dim) -> (n_layers, n_seq, dim)
            self.hidden_states_pi[batch_inds][self.seq_start_indices].swapaxes(0, 1),
            self.cell_states_pi[batch_inds][self.seq_start_indices].swapaxes(0, 1),
        )
        lstm_states_vf = (
            # (n_envs * n_steps, n_layers, dim) -> (n_layers, n_seq, dim)
            self.hidden_states_vf[batch_inds][self.seq_start_indices].swapaxes(0, 1),
            self.cell_states_vf[batch_inds][self.seq_start_indices].swapaxes(0, 1),
        )
        # lstm_states_pi = (self.to_torch(lstm_states_pi[0]).contiguous(), self.to_torch(lstm_states_pi[1]).contiguous())
        # lstm_states_vf = (self.to_torch(lstm_states_vf[0]).contiguous(), self.to_torch(lstm_states_vf[1]).contiguous())
        lstm_states_pi = (lstm_states_pi[0].contiguous(), lstm_states_pi[1].contiguous())
        lstm_states_vf = (lstm_states_vf[0].contiguous(), lstm_states_vf[1].contiguous())

        # observations = {key: self.pad(obs[batch_inds]) for (key, obs) in self.observations.items()}
        # observations = {key: obs.reshape((padded_batch_size,) + self.obs_shape[key]) for (key, obs) in observations.items()}
        observations = {key: self.pad_th(obs[batch_inds]).reshape((padded_batch_size,) + self.obs_shape[key]) for (key, obs) in self.observations.items()}

        return LSTMDictRolloutBufferSamples(
            obs_vae=self.observations['image'][batch_inds],
            observations=observations,
            returns=self.pad_and_flatten_th(self.returns[batch_inds]),
            lstm_states=RNNStates(lstm_states_pi, lstm_states_vf),
            episode_starts=self.pad_and_flatten_th(self.episode_starts[batch_inds]),
        )
    
    def _get_samples_with_pre(
        self,
        batch_inds: np.ndarray,
        env_change: np.ndarray,
        env: Optional[VecNormalize] = None,
    ) -> LSTMDictRolloutBufferSamples:
        self.seq_start_indices, self.pad, _, self.next_seq_start_indices, self.next_pad, self.next_pad_and_flatten = create_double_sequencers(
            self.episode_starts[batch_inds], env_change[batch_inds], self.device
        )
        # print("self.pad: ", self.pad)
        # print("self.next_pad: ", self.next_pad)

        n_seq = len(self.next_seq_start_indices)
        max_length = self.pad(self.actions[batch_inds]).shape[1]
        max_length_next = self.next_pad(self.actions[batch_inds]).shape[1]
        # print(max_length, max_length_next)
        padded_batch_size = n_seq * max_length_next
        # We retrieve the lstm hidden states that will allow
        # to properly initialize the LSTM at the beginning of each sequence
        # print(self.seq_start_indices)
        # print(self.next_seq_start_indices)
        lstm_states_pi = (
            # (n_envs * n_steps, n_layers, dim) -> (n_layers, n_seq, dim)
            self.hidden_states_pi[batch_inds][self.next_seq_start_indices].swapaxes(0, 1),
            self.cell_states_pi[batch_inds][self.next_seq_start_indices].swapaxes(0, 1),
        )
        lstm_states_vf = (
            # (n_envs * n_steps, n_layers, dim) -> (n_layers, n_seq, dim)
            self.hidden_states_vf[batch_inds][self.next_seq_start_indices].swapaxes(0, 1),
            self.cell_states_vf[batch_inds][self.next_seq_start_indices].swapaxes(0, 1),
        )
        lstm_states_pi = (self.to_torch(lstm_states_pi[0]).contiguous(), self.to_torch(lstm_states_pi[1]).contiguous())
        lstm_states_vf = (self.to_torch(lstm_states_vf[0]).contiguous(), self.to_torch(lstm_states_vf[1]).contiguous())

        observations = {key: self.pad(obs[batch_inds]) for (key, obs) in self.observations.items()}
        observations = {key: obs.reshape((padded_batch_size,) + self.obs_shape[key]) for (key, obs) in observations.items()}
        next_observations = {key: self.next_pad(obs[batch_inds]) for (key, obs) in self.observations.items()}
        next_observations = {key: obs.reshape((padded_batch_size,) + self.obs_shape[key]) for (key, obs) in next_observations.items()}

        return LSTMDictRolloutBufferSamples(
            observations=observations,
            observations_next=next_observations,
            returns=self.next_pad_and_flatten(self.returns[batch_inds]),
            lstm_states=RNNStates(lstm_states_pi, lstm_states_vf),
            episode_starts=self.next_pad_and_flatten(self.episode_starts[batch_inds]),
        )
    
    def compute_returns_and_advantage(self, last_values: th.Tensor, dones: th.Tensor) -> None:
        """
        Post-processing step: compute the lambda-return (TD(lambda) estimate)
        and GAE(lambda) advantage.

        Uses Generalized Advantage Estimation (https://arxiv.org/abs/1506.02438)
        to compute the advantage. To obtain Monte-Carlo advantage estimate (A(s) = R - V(S))
        where R is the sum of discounted reward with value bootstrap
        (because we don't always have full episode), set ``gae_lambda=1.0`` during initialization.

        The TD(lambda) estimator has also two special cases:
        - TD(1) is Monte-Carlo estimate (sum of discounted rewards)
        - TD(0) is one-step estimate with bootstrapping (r_t + gamma * v(s_{t+1}))

        For more information, see discussion in https://github.com/DLR-RM/stable-baselines3/pull/375.

        :param last_values: state value estimation for the last step (one for each env)
        :param dones: if the last step was a terminal step (one bool for each env).
        """

        last_gae_lam = 0
        for step in reversed(range(self.buffer_size)):
            if step == self.buffer_size - 1:
                next_non_terminal = 1.0 - dones
                next_values = last_values
            else:
                next_non_terminal = 1.0 - self.episode_starts[step+1]
                next_values = self.values[step+1]
            delta = self.rewards[step] + self.gamma * next_values * next_non_terminal - self.values[step]
            last_gae_lam = delta + self.gamma * self.gae_lambda * next_non_terminal * last_gae_lam
            self.advantages[step] = last_gae_lam
        # TD(lambda) estimator, see Github PR #375 or "Telescoping in TD(lambda)"
        # in David Silver Lecture 4: https://www.youtube.com/watch?v=PnHCvfgC_ZA
        self.returns = self.advantages + self.values

    def swap_and_flatten_th(self, arr: th.Tensor) -> th.Tensor:
        """
        Swap and then flatten axes 0 (buffer_size) and 1 (n_envs)
        to convert shape from [n_steps, n_envs, ...] (when ... is the shape of the features)
        to [n_steps * n_envs, ...] (which maintain the order)

        :param arr:
        :return:
        """
        shape = arr.shape
        if len(shape) < 3:
            shape = shape + (1,)
        return arr.swapaxes(0, 1).reshape(shape[0] * shape[1], *shape[2:])

class LSTMDictRolloutBuffer(DictRolloutBuffer):
    """
    Dict Rollout buffer used in on-policy algorithms like A2C/PPO.
    Extends the RecurrentRolloutBuffer to use dictionary observations

    :param buffer_size: Max number of element in the buffer
    :param observation_space: Observation space
    :param action_space: Action space
    :param hidden_state_shape: Shape of the buffer that will collect lstm states
    :param device: PyTorch device
    :param gae_lambda: Factor for trade-off of bias vs variance for Generalized Advantage Estimator
        Equivalent to classic advantage when set to 1.
    :param gamma: Discount factor
    :param n_envs: Number of parallel environments
    """

    def __init__(
        self,
        buffer_size: int,
        observation_space: spaces.Space,
        action_space: spaces.Space,
        hidden_state_shape: Tuple[int, int, int, int],
        device: Union[th.device, str] = "auto",
        gae_lambda: float = 1,
        gamma: float = 0.99,
        n_envs: int = 1,
        n_seq: int = 1,
    ):
        self.hidden_state_shape = hidden_state_shape
        self.seq_start_indices, self.seq_end_indices = None, None
        self.next_seq_start_indices, self.next_seq_end_indices = None, None
        self.n_seq = n_seq
        super().__init__(buffer_size, observation_space, action_space, device, gae_lambda, gamma, n_envs=n_envs)

    def reset(self):
        assert isinstance(self.obs_shape, dict), "DictRolloutBuffer must be used with Dict obs space only"
        self.observations = {}
        for key, obs_input_shape in self.obs_shape.items():
            self.observations[key] = np.zeros((self.buffer_size, self.n_envs) + obs_input_shape, dtype=np.float32)
        self.actions = np.zeros((self.buffer_size, self.n_envs, self.action_dim), dtype=np.float32)
        self.rewards = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
        self.returns = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
        self.episode_starts = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
        self.values = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
        self.log_probs = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
        self.advantages = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
        self.generator_ready = False
        self.pos = 0
        self.full = False

        self.hidden_states_pi = np.zeros(self.hidden_state_shape, dtype=np.float32)
        self.cell_states_pi = np.zeros(self.hidden_state_shape, dtype=np.float32)
        self.hidden_states_vf = np.zeros(self.hidden_state_shape, dtype=np.float32)
        self.cell_states_vf = np.zeros(self.hidden_state_shape, dtype=np.float32)
        

    def add(self, *args, lstm_states: RNNStates, **kwargs) -> None:
        """
        :param hidden_states: LSTM cell and hidden state
        """
        self.hidden_states_pi[self.pos] = np.array(lstm_states.pi[0].cpu().numpy())
        self.cell_states_pi[self.pos] = np.array(lstm_states.pi[1].cpu().numpy())
        self.hidden_states_vf[self.pos] = np.array(lstm_states.vf[0].cpu().numpy())
        self.cell_states_vf[self.pos] = np.array(lstm_states.vf[1].cpu().numpy())

        super().add(*args, **kwargs)

    def get(self, batch_size: Optional[int] = None) -> Generator[LSTMDictRolloutBufferSamples, None, None]:
        assert self.full, "Rollout buffer must be full before sampling from it"

        # Prepare the data
        if not self.generator_ready:
            # hidden_state_shape = (self.n_steps, lstm.num_layers, self.n_envs, lstm.hidden_size)
            # swap first to (self.n_steps, self.n_envs, lstm.num_layers, lstm.hidden_size)
            for tensor in ["hidden_states_pi", "cell_states_pi", "hidden_states_vf", "cell_states_vf"]:
                self.__dict__[tensor] = self.__dict__[tensor].swapaxes(1, 2)

            for key, obs in self.observations.items():
                self.observations[key] = self.swap_and_flatten(obs)

            for tensor in [
                "actions",
                "values",
                "log_probs",
                "advantages",
                "returns",
                "hidden_states_pi",
                "cell_states_pi",
                "hidden_states_vf",
                "cell_states_vf",
                "episode_starts",
            ]:
                self.__dict__[tensor] = self.swap_and_flatten(self.__dict__[tensor])
            self.generator_ready = True

        # Return everything, don't create minibatches
        if batch_size is None:
            batch_size = self.buffer_size * self.n_envs

        # Trick to shuffle a bit: keep the sequence order
        # but split the indices in two
        # split_index = np.random.randint(self.buffer_size * self.n_envs)
        split_index = 0
        indices = np.arange(self.buffer_size * self.n_envs)
        indices = np.concatenate((indices[split_index:], indices[:split_index]))

        env_change = np.zeros(self.buffer_size * self.n_envs).reshape(self.buffer_size, self.n_envs)
        # Flag first timestep as change of environment
        env_change[0, :] = 1.0
        env_change = self.swap_and_flatten(env_change)

        start_idx = 0
        while start_idx < self.buffer_size * self.n_envs:
            batch_inds = indices[start_idx : start_idx + batch_size]
            yield self._get_samples(batch_inds, env_change)
            start_idx += batch_size

    def _get_samples(
        self,
        batch_inds: np.ndarray,
        env_change: np.ndarray,
        env: Optional[VecNormalize] = None,
    ) -> LSTMDictRolloutBufferSamples:
        self.seq_start_indices, self.pad, self.pad_and_flatten = creare_sequencers_with_minLength(
            self.episode_starts[batch_inds], env_change[batch_inds], self.device, 30
        )
        # print("batch_inds: ", batch_inds)
        # print("self.pad: ", self.pad)

        n_seq = len(self.seq_start_indices)
        max_length = self.pad(self.actions[batch_inds]).shape[1]
        padded_batch_size = n_seq * max_length
        # print(n_seq, max_length)
        # We retrieve the lstm hidden states that will allow
        # to properly initialize the LSTM at the beginning of each sequence
        lstm_states_pi = (
            # (n_envs * n_steps, n_layers, dim) -> (n_layers, n_seq, dim)
            self.hidden_states_pi[batch_inds][self.seq_start_indices].swapaxes(0, 1),
            self.cell_states_pi[batch_inds][self.seq_start_indices].swapaxes(0, 1),
        )
        lstm_states_vf = (
            # (n_envs * n_steps, n_layers, dim) -> (n_layers, n_seq, dim)
            self.hidden_states_vf[batch_inds][self.seq_start_indices].swapaxes(0, 1),
            self.cell_states_vf[batch_inds][self.seq_start_indices].swapaxes(0, 1),
        )
        lstm_states_pi = (self.to_torch(lstm_states_pi[0]).contiguous(), self.to_torch(lstm_states_pi[1]).contiguous())
        lstm_states_vf = (self.to_torch(lstm_states_vf[0]).contiguous(), self.to_torch(lstm_states_vf[1]).contiguous())

        observations = {key: self.pad(obs[batch_inds]) for (key, obs) in self.observations.items()}
        observations = {key: obs.reshape((padded_batch_size,) + self.obs_shape[key]) for (key, obs) in observations.items()}

        return LSTMDictRolloutBufferSamples(
            observations=observations,
            returns=self.pad_and_flatten(self.returns[batch_inds]),
            lstm_states=RNNStates(lstm_states_pi, lstm_states_vf),
            episode_starts=self.pad_and_flatten(self.episode_starts[batch_inds]),
        )
    
    def _get_samples_with_pre(
        self,
        batch_inds: np.ndarray,
        env_change: np.ndarray,
        env: Optional[VecNormalize] = None,
    ) -> LSTMDictRolloutBufferSamples:
        self.seq_start_indices, self.pad, _, self.next_seq_start_indices, self.next_pad, self.next_pad_and_flatten = create_double_sequencers(
            self.episode_starts[batch_inds], env_change[batch_inds], self.device
        )
        # print("self.pad: ", self.pad)
        # print("self.next_pad: ", self.next_pad)

        n_seq = len(self.next_seq_start_indices)
        max_length = self.pad(self.actions[batch_inds]).shape[1]
        max_length_next = self.next_pad(self.actions[batch_inds]).shape[1]
        # print(max_length, max_length_next)
        padded_batch_size = n_seq * max_length_next
        # We retrieve the lstm hidden states that will allow
        # to properly initialize the LSTM at the beginning of each sequence
        # print(self.seq_start_indices)
        # print(self.next_seq_start_indices)
        lstm_states_pi = (
            # (n_envs * n_steps, n_layers, dim) -> (n_layers, n_seq, dim)
            self.hidden_states_pi[batch_inds][self.next_seq_start_indices].swapaxes(0, 1),
            self.cell_states_pi[batch_inds][self.next_seq_start_indices].swapaxes(0, 1),
        )
        lstm_states_vf = (
            # (n_envs * n_steps, n_layers, dim) -> (n_layers, n_seq, dim)
            self.hidden_states_vf[batch_inds][self.next_seq_start_indices].swapaxes(0, 1),
            self.cell_states_vf[batch_inds][self.next_seq_start_indices].swapaxes(0, 1),
        )
        lstm_states_pi = (self.to_torch(lstm_states_pi[0]).contiguous(), self.to_torch(lstm_states_pi[1]).contiguous())
        lstm_states_vf = (self.to_torch(lstm_states_vf[0]).contiguous(), self.to_torch(lstm_states_vf[1]).contiguous())

        observations = {key: self.pad(obs[batch_inds]) for (key, obs) in self.observations.items()}
        observations = {key: obs.reshape((padded_batch_size,) + self.obs_shape[key]) for (key, obs) in observations.items()}
        next_observations = {key: self.next_pad(obs[batch_inds]) for (key, obs) in self.observations.items()}
        next_observations = {key: obs.reshape((padded_batch_size,) + self.obs_shape[key]) for (key, obs) in next_observations.items()}

        return LSTMDictRolloutBufferSamples(
            observations=observations,
            observations_next=next_observations,
            returns=self.next_pad_and_flatten(self.returns[batch_inds]),
            lstm_states=RNNStates(lstm_states_pi, lstm_states_vf),
            episode_starts=self.next_pad_and_flatten(self.episode_starts[batch_inds]),
        )
        
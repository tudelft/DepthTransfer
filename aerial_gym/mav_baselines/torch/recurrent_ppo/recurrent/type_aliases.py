from typing import NamedTuple, Tuple

import torch as th
from stable_baselines3.common.type_aliases import TensorDict


class RNNStates(NamedTuple):
    pi: Tuple[th.Tensor, ...]
    vf: Tuple[th.Tensor, ...]


class RecurrentRolloutBufferSamples(NamedTuple):
    observations: th.Tensor
    actions: th.Tensor
    old_values: th.Tensor
    old_log_prob: th.Tensor
    advantages: th.Tensor
    returns: th.Tensor
    lstm_states: RNNStates
    episode_starts: th.Tensor
    mask: th.Tensor

class LatentLSTMRolloutBufferSamples(NamedTuple):
    observations: th.Tensor
    latent_lstm_pi: th.Tensor
    latent_lstm_vf: th.Tensor
    actions: th.Tensor
    old_values: th.Tensor
    old_log_prob: th.Tensor
    advantages: th.Tensor
    returns: th.Tensor
    lstm_states: RNNStates
    episode_starts: th.Tensor
    mask: th.Tensor

class DistLatentLSTMRolloutBufferSamples(NamedTuple):
    observations: th.Tensor
    latent_lstm_pi: th.Tensor
    latent_lstm_vf: th.Tensor
    actions: th.Tensor
    old_values: th.Tensor
    old_log_prob: th.Tensor
    advantages: th.Tensor
    returns: th.Tensor
    lstm_states_means: RNNStates
    lstm_states_stds: RNNStates
    episode_starts: th.Tensor
    mask: th.Tensor

class InputLSTMRolloutBufferSamples(NamedTuple):
    latent_lstm_pi: th.Tensor
    latent_lstm_vf: th.Tensor
    actions: th.Tensor
    old_values: th.Tensor
    old_log_prob: th.Tensor
    advantages: th.Tensor
    returns: th.Tensor    

class RecurrentDictRolloutBufferSamples(NamedTuple):
    observations: TensorDict
    actions: th.Tensor
    old_values: th.Tensor
    old_log_prob: th.Tensor
    advantages: th.Tensor
    returns: th.Tensor
    lstm_states: RNNStates
    episode_starts: th.Tensor
    mask: th.Tensor

class LSTMDictRolloutBufferSamples(NamedTuple):
    obs_vae: th.Tensor
    observations: TensorDict
    returns: th.Tensor
    lstm_states: RNNStates
    episode_starts: th.Tensor

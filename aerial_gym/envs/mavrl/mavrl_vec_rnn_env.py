import os
from os.path import join, exists
from typing import Tuple, Any, List, Optional, Type, Dict
import numpy as np
import torch
from torch import nn
from torchvision import transforms
from stable_baselines3.common.vec_env.base_vec_env import (VecEnv, VecEnvIndices)
from stable_baselines3.common.preprocessing import preprocess_obs
from stable_baselines3.common.utils import zip_strict
from aerial_gym.mav_baselines.torch.recurrent_ppo.recurrent.rnn_extractor import Encoder320, EncoderResnet, DecoderResnet, Decoder320
from aerial_gym.mav_baselines.torch.common.running_mean_std import RunningMeanStd
import gym
from gymnasium import spaces
from aerial_gym.envs import MAVRLTask
from omegaconf import OmegaConf
from aerial_gym.mav_baselines.torch.models.vae_320 import min_pool2d
import cv2  # Ensure cv2 is imported

image_width = 320
image_height = 224

class MavrlEnvVecRNN(VecEnv):
    def __init__(self, task: MAVRLTask, vae_dir: str = None, lstm_hidden_size: int = 256, n_lstm_layers: int = 1, reconstruction_members: Optional[List[bool]] = None):
        self.env = task
        self.num_seq = 1
        self.num_envs = task.num_envs
        self.num_actions = task.num_actions
        self.device = task.device
        
        # Config shortcuts
        latent_cfg = self.env.cfg.LatentSpaceCfg
        self.use_resnet_vae = latent_cfg.use_resnet_vae
        self.use_min_pooling = latent_cfg.use_min_pooling
        self.use_kl_latent_loss = latent_cfg.use_kl_latent_loss
        self.normalize_obs_flag = latent_cfg.normalize_obs
        
        self.state_dim = latent_cfg.vae_dims + latent_cfg.state_dims
        self.lstm_hidden_size = lstm_hidden_size
        self.n_lstm_layers = n_lstm_layers
        self.reconstruction_members = reconstruction_members

        # Initialize Encoder
        if not self.use_resnet_vae:
            self.features_extractor = Encoder320(self.env.observation_space, latent_cfg.vae_dims, ngf=64)
            if self.reconstruction_members is not None:
                self.feature_decoder0 = Decoder320(self.env.observation_space, latent_cfg.vae_dims)
                self.feature_decoder0.to(self.device)
        else:
            config_path = '../mav_baselines/torch/controlNet/models/encoder.yaml'
            ddconfig = OmegaConf.load(config_path)['ddconfig']
            self.features_extractor = EncoderResnet(self.env.observation_space, latent_cfg.vae_dims, ddconfig)
            if self.reconstruction_members is not None:
                self.feature_decoder0 = DecoderResnet(ddconfig)
                self.feature_decoder0.to(self.device)
        
        # Observation Space
        state_total_dim = lstm_hidden_size + latent_cfg.state_dims
        obs_shape = [self.num_seq, 2, state_total_dim] if self.use_kl_latent_loss else [self.num_seq, state_total_dim]
        
        self._observation_space = spaces.Box(
            np.ones(obs_shape) * -np.inf,
            np.ones(obs_shape) * np.inf,
            dtype=np.float64,
        )

        # Reward Tracking (Vectorized)
        self.rew_dim = len(self.env.cfg.RLParamsCfg.names)
        self.cur_episode_rewards = torch.zeros(self.num_envs, device=self.device)
        self.cur_episode_lengths = torch.zeros(self.num_envs, device=self.device, dtype=torch.int64)
        self.sum_reward_components = torch.zeros((self.num_envs, self.rew_dim - 1), device=self.device)

        # Load VAE
        if vae_dir is not None:
            vae_file = join(vae_dir, 'vae_64', 'best.tar')
            assert exists(vae_file), "No trained VAE in the logdir..."
            state_vae = torch.load(vae_file)
            print(f"Loading VAE at epoch {state_vae['epoch']} with test error {state_vae['precision']}")
            self.features_extractor.load_state_dict(state_vae['state_dict'])
        
        for param in self.features_extractor.parameters():
            param.requires_grad = False
        self.features_extractor.to(self.device)

        # LSTM Actor
        self.lstm_actor = nn.LSTM(
            self.state_dim,
            lstm_hidden_size,
            num_layers=n_lstm_layers,
        )
        self.lstm_actor.to(self.device)
        self.mu_linear = nn.Linear(lstm_hidden_size, 3 * latent_cfg.vae_dims)
        self.mu_linear.to(self.device)
        # LSTM States
        self.last_lstm_states = (torch.zeros(self.n_lstm_layers, self.num_envs, self.lstm_hidden_size, device=self.device),
                         torch.zeros(self.n_lstm_layers, self.num_envs, self.lstm_hidden_size, device=self.device))
        self.last_episode_starts = torch.ones(self.num_envs, device=self.device)
        
        # Normalization
        if self.normalize_obs_flag:
            self.obs_rms = RunningMeanStd(shape=[self.num_envs, state_total_dim], device=self.device)
            self.obs_rms_new = RunningMeanStd(shape=[self.num_envs, state_total_dim], device=self.device)

        # Data collection
        self.stereo_gt_enabled = self.env.cfg.camera_params.stereo_ground_truth
        if self.stereo_gt_enabled:
            self.resize_transform = transforms.Compose([transforms.Resize((image_height, image_width))])
            self.index = 0
            self.images_data = []
            self.ground_truth = []
            self.dataset = {}

    @staticmethod
    def _process_sequence(
        features: torch.Tensor,
        lstm_states: Tuple[torch.Tensor, torch.Tensor],
        episode_starts: torch.Tensor,
        lstm: nn.LSTM,
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        n_seq = lstm_states[0].shape[1]
        features_sequence = features.reshape((n_seq, -1, lstm.input_size)).swapaxes(0, 1)
        episode_starts = episode_starts.reshape((n_seq, -1)).swapaxes(0, 1)

        if torch.all(episode_starts == 0.0):
            lstm_output, lstm_states = lstm(features_sequence, lstm_states)
            lstm_output = torch.flatten(lstm_output.transpose(0, 1), start_dim=0, end_dim=1)
            return lstm_output, lstm_states

        lstm_output = []
        for feat, episode_start in zip_strict(features_sequence, episode_starts):
            # Dynamic state reset logic
            hidden, lstm_states = lstm(
                feat.unsqueeze(dim=0),
                (
                    (1.0 - episode_start).view(1, n_seq, 1) * lstm_states[0],
                    (1.0 - episode_start).view(1, n_seq, 1) * lstm_states[1],
                ),
            )
            lstm_output.append(hidden)
            
        lstm_output = torch.flatten(torch.cat(lstm_output).transpose(0, 1), start_dim=0, end_dim=1)
        return lstm_output, lstm_states

    def load_features_extractor(self, features_extractor: Optional[Dict[str, Any]], lstm: Optional[Dict[str, Any]] = None, 
                                mu_linear: Optional[Dict[str, Any]] = None, decoder: Optional[Dict[str, Any]] = None):
        self.features_extractor.load_state_dict(features_extractor, strict=True)
        if lstm: self.lstm_actor.load_state_dict(lstm, strict=True)
        if decoder:
            self.feature_decoder0.load_state_dict(decoder, strict=True)
            self.feature_decoder0.to(self.device)
        if mu_linear:
            self.mu_linear.load_state_dict(mu_linear, strict=True)
            self.mu_linear.to(self.device)

    def reset(self, if_easy_start: bool = False, save_pcd: bool = False, if_set_seed: bool = False):
        self.env.reset_idx(torch.arange(self.num_envs, device=self.device), if_easy_start=if_easy_start, if_set_seed=if_set_seed)
        if save_pcd:
            self.load_pcd()
            print("PCD loaded")
            self.env.reset_idx(torch.arange(self.num_envs, device=self.device), if_reset_obstacles=False, if_easy_start=if_easy_start)

        obs_dict, *_ = self.env.step(torch.zeros(self.num_envs, self.num_actions, device=self.device, requires_grad=False))

        if self.stereo_gt_enabled:
            obs_dict['image'] = self.resize_transform(obs_dict['image'])

        # Corrected here: unpack the tuple and return only latent_pi
        latent_pi, lstm_states_pi = self._process_obs(obs_dict, is_reset=True)
        
        # Update internal LSTM states (important for RNN consistency)
        self.last_lstm_states = lstm_states_pi
        
        # Return only the observation Tensor
        return latent_pi

    def step(self, actions):
        obs_dict, _, _rew_buf, _reset_buf, _extras = self.env.step(actions)

        # --- Data Collection ---
        if self.stereo_gt_enabled:
            self._collect_stereo_data(obs_dict)

        # --- Vectorized Reward Tracking ---
        self.cur_episode_rewards += _rew_buf[:, -1]
        self.cur_episode_lengths += 1
        self.sum_reward_components += _rew_buf[:, :self.rew_dim - 1]

        # --- Info Construction (Sparse) ---
        info = [{} for _ in range(self.num_envs)]
        reset_indices = torch.nonzero(_reset_buf).squeeze(-1)

        if len(reset_indices) > 0:
            reset_indices_cpu = reset_indices.cpu().numpy()
            ep_rewards = self.cur_episode_rewards[reset_indices].cpu().numpy()
            ep_lengths = self.cur_episode_lengths[reset_indices].cpu().numpy()
            ep_components = self.sum_reward_components[reset_indices].cpu().numpy()
            
            reward_names = self.reward_names
            
            for idx, env_idx in enumerate(reset_indices_cpu):
                epinfo = {"r": ep_rewards[idx], "l": ep_lengths[idx]}
                for j in range(self.rew_dim - 1):
                    epinfo[reward_names[j]] = ep_components[idx, j]
                info[env_idx]["episode"] = epinfo
            
            # Reset buffers
            self.cur_episode_rewards[reset_indices] = 0
            self.cur_episode_lengths[reset_indices] = 0
            self.sum_reward_components[reset_indices] = 0

        # --- Observation Processing with LSTM ---
        latent_pi, lstm_states_pi = self._process_obs(obs_dict, is_reset=False)
        
        # Update internal states
        self.last_lstm_states = lstm_states_pi
        self.last_episode_starts = _reset_buf

        return latent_pi, _rew_buf[:, -1], _reset_buf, info

    def _process_obs(self, obs_dict, is_reset=False):
        features = None
        states = None
        
        with torch.inference_mode():
            if 'image' in obs_dict:
                preprocessed_obs = preprocess_obs(obs_dict['image'], self.env.observation_space['image'], normalize_images=True)
                if self.use_min_pooling:
                    preprocessed_obs = min_pool2d(preprocessed_obs, kernel_size=7, stride=1, padding=3)
                features = self.features_extractor(preprocessed_obs).unsqueeze(1)
            
            if 'state' in obs_dict:
                states = obs_dict['state']

            if not self.use_kl_latent_loss:
                _obs = torch.cat([features, states], dim=2)
                
                if is_reset:
                    current_starts = torch.ones(self.num_envs, device=self.device)
                else:
                    current_starts = self.last_episode_starts

                latent_pi, lstm_states_pi = self._process_sequence(
                    _obs, self.last_lstm_states, current_starts, self.lstm_actor
                )
            else:
                latent_pi = features # Placeholder
                lstm_states_pi = self.last_lstm_states

            # --- CV2 Visualization (Restored) ---
            if self.reconstruction_members is not None:
                recons = self.predict_img(latent_pi[:1]) # Take first env
                
                # Logic from original code:
                if (len(recons) > 1) and (recons[0] is not None) and (recons[1] is not None):
                    img1 = np.clip((recons[0].reshape([224, 320, 1]) * 255), 0, 255).astype(np.uint8)
                    img2 = np.clip((recons[1].reshape([224, 320, 1]) * 255), 0, 255).astype(np.uint8)
                    imgs = np.hstack([img1, img2])
                    cv2.imshow("recon", imgs)
                    cv2.waitKey(1)
                elif (len(recons) > 0) and (recons[0] is not None):
                    imgs = (recons[0].reshape([224, 320, 1]) * 255).astype(np.uint8)
                    cv2.imshow("recon", imgs)
                    cv2.waitKey(1)

            # Concatenate Output
            latent_pi = torch.cat([latent_pi.unsqueeze(1), states], dim=2)
            
            if self.normalize_obs_flag:
                self.obs_rms_new.update(latent_pi.squeeze(1))
                latent_pi = self.normalize_obs(latent_pi.squeeze(1)).unsqueeze(1)
            
            if is_reset:
                self.last_lstm_states = lstm_states_pi 
                self.last_episode_starts = torch.zeros(self.num_envs, device=self.device) 

            return latent_pi, lstm_states_pi

    def _collect_stereo_data(self, obs_dict):
        save_dir = join('../tmp', 'images')
        if not exists(save_dir):
            os.makedirs(save_dir, exist_ok=True)
        
        if self.index % 500 < 499:
            self.images_data.append(obs_dict['image'].clone())
            self.ground_truth.append(obs_dict['stereo_ground_truth'].clone())
        elif self.index % 500 == 499:
            data_path = join(save_dir, f"stereo_depth_iter_{self.index:05d}.pth")
            self.dataset['stereo_depth'] = torch.stack(self.images_data)
            self.dataset['stereo_ground_truth'] = torch.stack(self.ground_truth)
            torch.save(self.dataset, data_path)
            self.images_data.clear()
            self.ground_truth.clear()
            self.dataset.clear()
        
        self.index += 1
        obs_dict['image'] = self.resize_transform(obs_dict['image'])

    def predict_img(self, latent_obs: torch.Tensor):
        with torch.no_grad():
            latent_obs = self.mu_linear(latent_obs)
            recon_latent_size = self.env.cfg.LatentSpaceCfg.vae_dims
            # Split latent into 3 parts (pre, cur, next)
            pre_latent_obs, cur_latent_obs, next_latent_obs = torch.split(latent_obs, [recon_latent_size, recon_latent_size, recon_latent_size], dim=1)
            total_laten_obs = [pre_latent_obs, cur_latent_obs, next_latent_obs]
            
            reconstruction = []
            if self.reconstruction_members:
                for i in range(len(self.reconstruction_members)):
                    # Guard against index out of range if members < 3
                    if i < len(total_laten_obs) and self.reconstruction_members[i]:
                        reconstruction.append(self.feature_decoder0(total_laten_obs[i]).cpu().numpy())
                    else:
                        reconstruction.append(None)
        return reconstruction

    # --- Utility Methods ---
    def cal_success_rate(self): return self.env.cal_success_rate()
    def get_trial_num(self): return self.env.get_trial_num()
    def load_pcd(self): self.env.pcd_load_step()
    def save_log_data(self, save_dir): self.env.save_log_data(save_dir)
    def save_encoder(self, save_path, n_iter): 
        torch.save(self.features_extractor.state_dict(), f"{save_path}/encoder_iter_{n_iter:05d}.pth")
        torch.save(self.lstm_actor.state_dict(), f"{save_path}/lstm_iter_{n_iter:05d}.pth")
        torch.save(self.mu_linear.state_dict(), f"{save_path}/mu_linear_iter_{n_iter:05d}.pth")
    def reset_reward_coeffs(self): self.env.reset_reward_coeffs()
    def set_seed(self, seed: int=1): self.env.set_seed(seed)
    def reset_goal_threshold(self, threshold: float): self.env.reset_goal_threshold(threshold)
    def update_rms(self): self.obs_rms = self.obs_rms_new
    def get_obs_norm(self): return self.obs_rms.mean, self.obs_rms.var

    def _normalize_obs(self, obs: torch.Tensor, obs_rms: RunningMeanStd) -> torch.Tensor:
        return (obs - obs_rms.mean) / torch.sqrt(obs_rms.var + 1e-8)
    
    def normalize_obs(self, obs: torch.Tensor) -> torch.Tensor:
        return self._normalize_obs(obs, self.obs_rms)

    def save_rms(self, save_dir, n_iter) -> None:
        if not os.path.exists(save_dir): os.mkdir(save_dir)
        data = {'mean': self.obs_rms.mean, 'var': self.obs_rms.var}
        torch.save(data, f"{save_dir}/iter_{n_iter:05d}.pth")

    def load_rms(self, data_dir, env_nums = 0) -> None:
        if os.path.exists(data_dir):
            data = torch.load(data_dir)
            self.obs_rms.mean = data['mean']
            self.obs_rms.var = data['var']
            if env_nums > 0:
                self.obs_rms.mean = self.obs_rms.mean[:env_nums]
                self.obs_rms.var = self.obs_rms.var[:env_nums]
        else:
            print("No RMS data found")

    def env_is_wrapped(self, wrapper_class: Type[gym.Wrapper], indices: VecEnvIndices = None) -> List[bool]:
        target_envs = self._get_target_envs(indices)
        from stable_baselines3.common import env_util
        return [env_util.is_wrapped(env_i, wrapper_class) for env_i in target_envs]

    def _get_target_envs(self, indices: VecEnvIndices) -> List[gym.Env]:
        indices = self._get_indices(indices)
        return [self.envs[i] for i in indices]

    # Properties & Unimplemented
    @property
    def observation_space(self): return self._observation_space
    @property
    def action_space(self): return self.env.action_space
    @property
    def reward_names(self): return self.env.getRewardNames()
    
    def env_method(self, *args, **kwargs): raise RuntimeError("Not implemented")
    def step_wait(self): raise RuntimeError("Not implemented")
    def get_attr(self, *args, **kwargs): raise RuntimeError("Not implemented")
    def set_attr(self, *args, **kwargs): raise RuntimeError("Not implemented")
    def step_async(self): raise RuntimeError("Not implemented")
    def seed(self, seed=0): raise RuntimeError("Not implemented")
    def close(self): raise RuntimeError("Not implemented")
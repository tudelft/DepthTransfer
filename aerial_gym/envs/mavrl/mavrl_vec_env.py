import os
from os.path import join, exists
import numpy as np
import torch
from torch.cuda.amp import autocast
import math
from torchvision import transforms
from typing import Any, List, Optional, Type, Dict
from stable_baselines3.common.vec_env.base_vec_env import (VecEnv,
                                                           VecEnvIndices,
                                                           VecEnvObs,
                                                           VecEnvStepReturn)
from stable_baselines3.common.preprocessing import preprocess_obs
from aerial_gym.mav_baselines.torch.recurrent_ppo.recurrent.rnn_extractor import Encoder320, EncoderResnet, DecoderResnet, Decoder320
from aerial_gym.mav_baselines.torch.common.running_mean_std import RunningMeanStd
# from stable_baselines3.common.running_mean_std import RunningMeanStd
import gym
from gymnasium import spaces
from aerial_gym.envs import MAVRLTask
from PIL import Image
import matplotlib.pyplot as plt
from omegaconf import OmegaConf
from aerial_gym.mav_baselines.torch.controlNet.ldm.util import instantiate_from_config
from aerial_gym.mav_baselines.torch.models.vae_320 import min_pool2d
import time
import cv2

image_width = 320
image_height = 224

class MavrlEnvVec(VecEnv):
    def __init__(self, task: MAVRLTask, reconstruction_members: bool = False, vae_dir: str = None):
        self.env = task
        self.num_seq = 1
        self.num_envs = task.num_envs
        self.num_actions = task.num_actions
        self.device = task.device
        self.use_resnet_vae = self.env.cfg.LatentSpaceCfg.use_resnet_vae
        self.use_min_pooling = self.env.cfg.LatentSpaceCfg.use_min_pooling
        self.use_kl_latent_loss = self.env.cfg.LatentSpaceCfg.use_kl_latent_loss
        self.state_dim = self.env.cfg.LatentSpaceCfg.vae_dims + self.env.cfg.LatentSpaceCfg.state_dims
        self.reconstruction_members = reconstruction_members
        if not self.use_resnet_vae:
            self.features_extractor = Encoder320(self.env.observation_space, self.env.cfg.LatentSpaceCfg.vae_dims, ngf=64)
            if self.reconstruction_members:
                self.feature_decoder0 = Decoder320(self.env.observation_space, self.env.cfg.LatentSpaceCfg.vae_dims)
        else:
            config_path = '../mav_baselines/torch/controlNet/models/encoder.yaml'
            self.features_extractor = EncoderResnet(self.env.observation_space, self.env.cfg.LatentSpaceCfg.vae_dims,
                                                     OmegaConf.load(config_path)['ddconfig'])
            if self.reconstruction_members:
                self.feature_decoder0 = DecoderResnet(OmegaConf.load(config_path)['ddconfig'])

        if not self.use_kl_latent_loss:
            self._observation_space = spaces.Box(
                    np.ones([self.num_seq, self.state_dim]) * -np.inf,
                    np.ones([self.num_seq, self.state_dim]) * np.inf,
                    dtype=np.float64,
                )
        else:
            self._observation_space = spaces.Box(
                    np.ones([self.num_seq, 2, self.state_dim]) * -np.inf,
                    np.ones([self.num_seq, 2, self.state_dim]) * np.inf,
                    dtype=np.float64,
                )
        self.rew_dim = len(self.env.cfg.RLParamsCfg.names)
        self.rewards = [[] for _ in range(self.num_envs)]
        self.sum_reward_components = torch.zeros((self.num_envs, self.rew_dim - 1), device=self.device)
        if vae_dir is not None:
            vae_file = join(vae_dir, 'vae_64', 'best.tar')
            assert exists(vae_file), "No trained VAE in the logdir..."
            state_vae = torch.load(vae_file)
            print("Loading VAE at epoch {} "
            "with test error {}".format(state_vae['epoch'], state_vae['precision']))
            self.features_extractor.load_state_dict(state_vae['state_dict'])
        for param in self.features_extractor.parameters():
            param.requires_grad = False
        self.features_extractor.to(self.device)
        #  state normalization
        if self.env.cfg.LatentSpaceCfg.normalize_obs:
            self.obs_rms = RunningMeanStd(shape=[self.num_envs, self.state_dim], device=self.device)
            self.obs_rms_new = RunningMeanStd(shape=[self.num_envs, self.state_dim], device=self.device)
        #save features extractor
        # torch.save(self.features_extractor.state_dict(), join(vae_dir, 'features_extractor.pth'))
        # self.reset()

        # for collecting data
        if self.env.cfg.camera_params.stereo_ground_truth:
            self.resize_transform = transforms.Compose([transforms.Resize((image_height, image_width))])
            self.index = 0
            self.images_data = []
            self.ground_truth = []
            self.dataset = {}

    def load_features_extractor(self, features_extractor: Optional[Dict[str, Any]] = None, decoder: Optional[Dict[str, Any]] = None):
        self.features_extractor.load_state_dict(features_extractor, strict=True)
        if (decoder is not None) and (decoder is not {}):
            self.feature_decoder0.load_state_dict(decoder, strict=True)
            self.feature_decoder0.to(self.device)

    def reset(self, if_easy_start: bool = False, save_pcd: bool = False, if_set_seed: bool = False):
        self.env.reset_idx(torch.arange(self.num_envs, device=self.device), if_easy_start=if_easy_start, if_set_seed = if_set_seed)
        if save_pcd:
            self.load_pcd()
            print("PCD loaded")
            self.env.reset_idx(torch.arange(self.num_envs, device=self.device), if_reset_obstacles=False, if_easy_start=if_easy_start)

        obs_dict, *_ = self.env.step(torch.zeros(self.num_envs, self.num_actions, device=self.device, requires_grad=False))

        if self.env.cfg.camera_params.stereo_ground_truth:
            obs_dict['image'] = self.resize_transform(obs_dict['image'])

        features = None
        states = None
        for key, _obs in obs_dict.items():
            if key == 'image':
                with torch.inference_mode():
                    preprocessed_obs = preprocess_obs(_obs, self.env.observation_space['image'], normalize_images=True)
                    if self.use_min_pooling:
                        preprocessed_obs = min_pool2d(preprocessed_obs, kernel_size=7, stride=1, padding=3)
                    if not self.use_resnet_vae:
                        features = self.features_extractor(preprocessed_obs).unsqueeze(1)
                    else:
                        features = self.features_extractor(preprocessed_obs)
                        features = features.unsqueeze(1)
            elif key == 'state':
                states = _obs
                # if self.env.cfg.LatentSpaceCfg.normalize_obs:
                #     self.obs_rms_new.update(states.squeeze(1))
                #     states = self.normalize_obs(states.squeeze(1)).unsqueeze(1)
        if not self.use_kl_latent_loss:
            _obs = torch.cat([features, states], dim=2)
            if self.env.cfg.LatentSpaceCfg.normalize_obs:
                self.obs_rms_new.update(_obs.squeeze(1))
                _obs = self.normalize_obs(_obs.squeeze(1)).unsqueeze(1)
        return _obs
    
    def step(self, actions):
        # t1 = time.time()
        obs_dict, _, _rew_buf, _reset_buf, _extras = self.env.step(actions)
        # save original images to .pth file every 500 steps
        if self.env.cfg.camera_params.stereo_ground_truth:
            save_dir = join('../tmp', 'images')
            if not exists(save_dir):
                    os.mkdir(save_dir)
            if self.index % 500 < 499:
                self.images_data.append(obs_dict['image'].clone())
                self.ground_truth.append(obs_dict['stereo_ground_truth'].clone())
            if self.index % 500 == 499:
                data_path = save_dir + "/stereo_depth_iter_{0:05d}".format(self.index) + ".pth"
                self.dataset['stereo_depth'] = torch.stack(self.images_data)
                self.dataset['stereo_ground_truth'] = torch.stack(self.ground_truth)
                torch.save(self.dataset, data_path)
                self.images_data.clear()
                self.ground_truth.clear()
                self.dataset.clear()
            self.index += 1
            obs_dict['image'] = self.resize_transform(obs_dict['image'])

        features = []
        states = None
        # t2 = time.time()
        for key, _obs in obs_dict.items():
            if key == 'image':
                with torch.inference_mode():
                    preprocessed_obs = preprocess_obs(_obs, self.env.observation_space['image'], normalize_images=True)
                    if self.use_min_pooling:
                        preprocessed_obs = min_pool2d(preprocessed_obs, kernel_size=7, stride=1, padding=3)
                    if not self.use_resnet_vae:
                        features = self.features_extractor(preprocessed_obs).unsqueeze(1)
                    else:
                        features = self.features_extractor(preprocessed_obs)
                        features = features.unsqueeze(1)
                    # features = torch.zeros([self.num_envs, 1, self.env.cfg.LatentSpaceCfg.vae_dims], device=self.device)
                    # # show 128 images in a 8 x 16 grid
                    # imgs = _obs[:, 0].cpu().numpy().reshape([128, 256, 256])
                    # imgs = imgs.reshape([8, 16, 256, 256])
                    # imgs = imgs.transpose(0, 2, 1, 3).reshape([8*256, 16*256, 1])
                    # cv2.imshow("real", imgs)
                    # cv2.waitKey(1)
                    # print("features: ", features[0, 0])
                    if self.reconstruction_members:
                        reconstruction = self.feature_decoder0(features[0]).cpu().numpy()
                        imgs = reconstruction.reshape([224, 320, 1])
                        cv2.imshow("recon", imgs)
                        imgs = _obs[0, 0].cpu().numpy().reshape([224, 320, 1])
                        cv2.imshow("real0", imgs)
                        cv2.waitKey(1)
            elif key == 'state':
                states = _obs
                # if self.env.cfg.LatentSpaceCfg.normalize_obs:
                #     self.obs_rms_new.update(states.squeeze(1))
                #     states = self.normalize_obs(states.squeeze(1)).unsqueeze(1)
                # states = torch.zeros_like(_obs)
        if not self.use_kl_latent_loss:
            _obs = torch.cat([features, states], dim=2)
            if self.env.cfg.LatentSpaceCfg.normalize_obs:
                self.obs_rms_new.update(_obs.squeeze(1))
                _obs = self.normalize_obs(_obs.squeeze(1)).unsqueeze(1)
        # else:
        #     _obs = torch.cat([dist, states.unsqueeze(-2).repeat(1, 1, 2, 1)], dim=3)
            
        # t3 = time.time()
        info = [{} for i in range(self.num_envs)]
        for i in range(self.num_envs):
            self.rewards[i].append(_rew_buf[i, -1])
            for j in range(self.rew_dim - 1):
                self.sum_reward_components[i, j] += _rew_buf[i, j]
            if _reset_buf[i]:
                eprew = sum(self.rewards[i]).cpu().numpy()
                eplen = len(self.rewards[i])
                epinfo = {"r": eprew, "l": eplen}
                for j in range(self.rew_dim - 1):
                    epinfo[self.reward_names[j]] = self.sum_reward_components[i, j].cpu().numpy()
                    self.sum_reward_components[i, j] = 0.0
                info[i]["episode"] = epinfo
                self.rewards[i].clear()
        # print("_rew_buf[:, -1]: ", _rew_buf[:, -1])
        # t4 = time.time()
        # print("render time: {0:.4f}, feature time: {1:.4f}, reward time: {2:.4f}".format(t2 - t1, t3 - t2, t4 - t3))
        return _obs, _rew_buf[:, -1], _reset_buf, info.copy()

    def predict_img(self,
        latent_obs: torch.Tensor,):
        # Switch to eval mode (this affects batch norm / dropout)
        with torch.no_grad():
            # latent_obs = th.tensor(latent_obs, dtype=th.float32, device=self.device)
            reconstruction = self.feature_decoder0(latent_obs).cpu().numpy()
        return reconstruction

    def cal_success_rate(self):
        return self.env.cal_success_rate()
    
    def get_trial_num(self):
        return self.env.get_trial_num()

    def env_is_wrapped(
        self, wrapper_class: Type[gym.Wrapper], indices: VecEnvIndices = None
    ) -> List[bool]:
        """Check if worker environments are wrapped with a given wrapper"""
        target_envs = self._get_target_envs(indices)
        # Import here to avoid a circular import
        from stable_baselines3.common import env_util

        return [env_util.is_wrapped(env_i, wrapper_class) for env_i in target_envs]
    
    def load_pcd(self):
        self.env.pcd_load_step()

    def save_log_data(self, save_dir):
        self.env.save_log_data(save_dir)
    
    def _get_target_envs(self, indices: VecEnvIndices) -> List[gym.Env]:
        indices = self._get_indices(indices)
        return [self.envs[i] for i in indices]
    
    def save_encoder(self, save_path, n_iter):    
        torch.save(self.features_extractor.state_dict(), save_path + "/iter_{0:05d}.pth".format(n_iter))
    
    def reset_reward_coeffs(self):
        self.env.reset_reward_coeffs()
    
    def set_seed(self, seed: int=1) -> None:
        self.env.set_seed(seed)

    def reset_goal_threshold(self, threshold: float):
        self.env.reset_goal_threshold(threshold)
    
    def update_rms(self):
        self.obs_rms = self.obs_rms_new

    def get_obs_norm(self):
        return self.obs_rms.mean, self.obs_rms.var
    
    def _normalize_obs(self, obs: torch.Tensor, obs_rms: RunningMeanStd) -> torch.Tensor:
        """
        Helper to normalize observation.
        :param obs:
        :param obs_rms: associated statistics
        :return: normalized observation
        """
        return (obs - obs_rms.mean) / torch.sqrt(obs_rms.var + 1e-8)
    
    def _unnormalize_obs(self, obs: torch.Tensor, obs_rms: RunningMeanStd) -> torch.Tensor:
        """
        Helper to unnormalize observation.
        :param obs:
        :param obs_rms: associated statistics
        :return: unnormalized observation
        """
        return (obs * torch.sqrt(obs_rms.var + 1e-8)) + obs_rms.mean
    
    def normalize_obs(self, obs: torch.Tensor) -> torch.Tensor:
        """
        Normalize observations using this VecNormalize's observations statistics.
        Calling this method does not update statistics.
        """
        # Avoid modifying by reference the original object
        # obs_ = deepcopy(obs)
        obs_ = self._normalize_obs(obs, self.obs_rms)
        return obs_

    def save_rms(self, save_dir, n_iter) -> None:
        if not os.path.exists(save_dir):
            os.mkdir(save_dir)
        data_path = save_dir + "/iter_{0:05d}".format(n_iter) + ".pth"
        data = {'mean': self.obs_rms.mean, 'var': self.obs_rms.var}
        torch.save(data, data_path)

    def load_rms(self, data_dir, env_nums = 0) -> None:
        self.mean, self.var = None, None
        if os.path.exists(data_dir):
            data_path = data_dir
            data = torch.load(data_path)
            self.obs_rms.mean = data['mean']
            self.obs_rms.var = data['var']
            if env_nums > 0:
                self.obs_rms.mean = self.obs_rms.mean[:env_nums]
                self.obs_rms.var = self.obs_rms.var[:env_nums]
        else:
            print("No data found in the directory")

    @property
    def observation_space(self):
        return self._observation_space
    
    @property
    def action_space(self):
        return self.env.action_space

    @property
    def reward_names(self):
        return self.env.getRewardNames()

    def env_method(self, method_name, *method_args, indices=None, **method_kwargs):
        """
        Call instance methods of vectorized environments.
        :param method_name: (str) The name of the environment method to invoke.
        :param indices: (list,int) Indices of envs whose method to call
        :param method_args: (tuple) Any positional arguments to provide in the call
        :param method_kwargs: (dict) Any keyword arguments to provide in the call
        :return: (list) List of items returned by the environment's method call
        """
        raise RuntimeError("This method is not implemented")
    
    def step_wait(self):
        raise RuntimeError("This method is not implemented")

    def get_attr(self, attr_name, indices=None):
        """
        Return attribute from vectorized environment.
        :param attr_name: (str) The name of the attribute whose value to return
        :param indices: (list,int) Indices of envs to get attribute from
        :return: (list) List of values of 'attr_name' in all environments
        """
        raise RuntimeError("This method is not implemented")

    def set_attr(self, attr_name, value, indices=None):
        """
        Set attribute inside vectorized environments.
        :param attr_name: (str) The name of attribute to assign new value
        :param value: (obj) Value to assign to `attr_name`
        :param indices: (list,int) Indices of envs to assign value
        :return: (NoneType)
        """
        raise RuntimeError("This method is not implemented")
    
    def step_async(self):
        raise RuntimeError("This method is not implemented")
    
    def seed(self, seed=0):
        raise RuntimeError("This method is not implemented")
    
    def close(self):
        raise RuntimeError("This method is not implemented")

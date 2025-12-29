import os
from os.path import join, exists
import numpy as np
from typing import Any, List, Optional, Type
from stable_baselines3.common.vec_env.base_vec_env import (VecEnv,
                                                           VecEnvIndices,
                                                           VecEnvObs,
                                                           VecEnvStepReturn)
from stable_baselines3.common.preprocessing import preprocess_obs
from aerial_gym.mav_baselines.torch.common.running_mean_std import RunningMeanStd
import gym
from gymnasium import spaces
from aerial_gym.envs import MAVRLTask
import torch
from torchvision import transforms
import cv2

image_width = 320
image_height = 224

class MavrlEnvVecV2(VecEnv):
    def __init__(self, task: MAVRLTask, vae_dir: str = None):
        self.env = task
        self.num_seq = 1
        self.num_envs = task.num_envs
        self.num_actions = task.num_actions
        self.device = task.device
        self.state_dim = self.env.cfg.LatentSpaceCfg.vae_dims + self.env.cfg.LatentSpaceCfg.state_dims
        self.rew_dim = len(self.env.cfg.RLParamsCfg.names)
        self.rewards = [[] for _ in range(self.num_envs)]
        self.sum_reward_components = np.zeros(
            [self.num_envs, self.rew_dim - 1], dtype=np.float64
        )

        if self.env.cfg.LatentSpaceCfg.normalize_obs:
            self.obs_rms = RunningMeanStd(shape=[self.num_envs, self.env.cfg.LatentSpaceCfg.state_dims], device=self.device)
            self.obs_rms_new = RunningMeanStd(shape=[self.num_envs, self.env.cfg.LatentSpaceCfg.state_dims], device=self.device)
        self.resize_transform = transforms.Compose([transforms.Resize((image_height, image_width))])

        # for collecting data
        self.index = 0
        self.images_data = []

    def reset(self, if_easy_start: bool = False):
        self.env.reset_idx(torch.arange(self.num_envs, device=self.device), if_easy_start=if_easy_start)
        obs_dict, *_ = self.env.step(torch.zeros(self.num_envs, self.num_actions, device=self.device, requires_grad=False))
        # obs_dict['image'] = self.resize_transform(obs_dict['image'])
        if self.env.cfg.LatentSpaceCfg.normalize_obs:
            self.obs_rms_new.update(obs_dict['state'].squeeze(1))
            obs_dict['state'] = self.normalize_obs(obs_dict['state'].squeeze(1)).unsqueeze(1)
        return obs_dict
    
    def step(self, actions):
        obs_dict, _, _rew_buf, _reset_buf, _extras = self.env.step(actions)
        # save original images to .pth file every 500 steps
        # save_dir = join('../tmp', 'images')
        # if not exists(save_dir):
        #         os.mkdir(save_dir)
        # if self.index % 500 < 499:
        #     self.images_data.append(obs_dict['image'].clone())
        # if self.index % 500 == 499:
        #     data_path = save_dir + "/iter_{0:05d}".format(self.index) + ".pth"
        #     torch.save(torch.stack(self.images_data), data_path)
        #     self.images_data.clear()
        # self.index += 1
        # obs_dict['image'] = self.resize_transform(obs_dict['image'])

        imgs = obs_dict['image'][0, 0].cpu().numpy().reshape([224, 320, 1])
        cv2.imshow("real0", imgs)
        cv2.waitKey(1)
        
        if self.env.cfg.LatentSpaceCfg.normalize_obs:
            self.obs_rms_new.update(obs_dict['state'].squeeze(1))
            obs_dict['state'] = self.normalize_obs(obs_dict['state'].squeeze(1)).unsqueeze(1)
        # transform the tensor dict observation to numpy array
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
                    epinfo[self.reward_names[j]] = self.sum_reward_components[i, j]
                    self.sum_reward_components[i, j] = 0.0
                info[i]["episode"] = epinfo
                self.rewards[i].clear()
        return obs_dict, _rew_buf[:, -1], _reset_buf, info.copy()
    
    def env_is_wrapped(
        self, wrapper_class: Type[gym.Wrapper], indices: VecEnvIndices = None
    ) -> List[bool]:
        """Check if worker environments are wrapped with a given wrapper"""
        target_envs = self._get_target_envs(indices)
        # Import here to avoid a circular import
        from stable_baselines3.common import env_util

        return [env_util.is_wrapped(env_i, wrapper_class) for env_i in target_envs]
    
    def _get_target_envs(self, indices: VecEnvIndices) -> List[gym.Env]:
        indices = self._get_indices(indices)
        return [self.envs[i] for i in indices]
    
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
        return self.env.observation_space
    
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

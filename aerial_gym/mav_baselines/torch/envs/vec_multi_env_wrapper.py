import os
import pickle
import time
from copy import deepcopy
from typing import Any, Callable, List, Optional, Sequence, Type, Union
from PIL import Image
import gym
import numpy as np
from gym import spaces
from numpy.core.fromnumeric import shape
from stable_baselines3.common.running_mean_std import RunningMeanStd
from stable_baselines3.common.vec_env.base_vec_env import (VecEnv,
                                                           VecEnvIndices,
                                                           VecEnvObs,
                                                           VecEnvStepReturn)
from stable_baselines3.common.vec_env.util import (copy_obs_dict, dict_to_obs,
                                                   obs_space_info)
from os.path import join, exists
import torch
from utils.misc import LSIZE, n_seq

class VisionEnvVec(VecEnv):
    #
    def __init__(self, impl, logdir=None):
        self.wrapper = impl
        self.act_dim = self.wrapper.getActDim()
        self.seq_dim = self.wrapper.getSeqDim()
        self.obs_dim = self.wrapper.getObsDim()
        self.state_dim = self.wrapper.getStateDim()
        self.rew_dim = self.wrapper.getRewDim()
        self.goal_obs_dim = self.wrapper.getGoalObsDim()
        self.img_width = self.wrapper.getImgWidth()
        self.img_height = self.wrapper.getImgHeight()
        self._observation_space = spaces.Dict(
            {
                'image': spaces.Box(
                    low=0,
                    high=255,
                    shape=(n_seq, 256, 256),
                    dtype='uint8',
                ),
                'state': spaces.Box(
                    np.ones([n_seq, self.goal_obs_dim]) * -np.Inf,
                    np.ones([n_seq, self.goal_obs_dim]) * np.Inf,
                    dtype=np.float64,
                )
            }
        )

        self._action_space = spaces.Box(
            low=np.ones(self.act_dim) * -1.0,
            high=np.ones(self.act_dim) * 1.0,
            dtype=np.float64,
        )
        self._observation = {'image': np.zeros([self.num_envs, n_seq, self.img_height, self.img_width], dtype=np.uint8),
                            'state': np.zeros([self.num_envs, n_seq, self.goal_obs_dim], dtype=np.float64)}
        self._state_observation = np.zeros([self.num_envs, self.goal_obs_dim], dtype=np.float64)
        self._observation_test = np.zeros([self.num_envs, self.obs_dim], dtype=np.float64)
        self._current_state = np.zeros([self.num_envs, self.state_dim], dtype=np.float64)
        self._rgb_img_obs = np.zeros(
            [self.num_envs, self.img_width * self.img_height * 3], dtype=np.uint8
        )
        self._gray_img_obs = np.zeros(
            [self.num_envs, self.img_width * self.img_height], dtype=np.uint8
        )
        self._depth_img_obs = np.zeros(
            [self.num_envs, self.img_width * self.img_height], dtype=np.float32
        )
        self.label_images = np.zeros([28, 28], dtype=np.float32)
        #
        self._reward_components = np.zeros(
            [self.num_envs, n_seq, self.rew_dim], dtype=np.float64
        )
        self._done = np.zeros((self.num_envs, n_seq), dtype=np.bool)
        self._single_reward_components = np.zeros(
            [self.num_envs, self.rew_dim], dtype=np.float64
        )
        self._single_done = np.zeros((self.num_envs), dtype=np.bool)

        self._extraInfoNames = self.wrapper.getExtraInfoNames()
        self.reward_names = self.wrapper.getRewardNames()
        self._extraInfo = np.zeros(
            [self.num_envs, len(self._extraInfoNames)], dtype=np.float64
        )
        self._extraInfo_test = np.zeros(
            [self.num_envs, len(self._extraInfoNames)], dtype=np.float64
        )

        self.rewards = [[] for _ in range(self.num_envs)]
        self.sum_reward_components = np.zeros(
            [self.num_envs, self.rew_dim - 1], dtype=np.float64
        )

        self._quadstate = np.zeros([self.num_envs, 14], dtype=np.float64)
        self._quadact = np.zeros([self.num_envs, self.act_dim], dtype=np.float64)
        self._flightmodes = np.zeros([self.num_envs, 1], dtype=np.float64)

        #  state normalization
        self.obs_rms = RunningMeanStd(shape=[self.num_envs, self.obs_dim])
        self.obs_rms_new = RunningMeanStd(shape=[self.num_envs, self.obs_dim])
        self.max_episode_steps = 1000

        self.image_memory = [[] for _ in range(self.num_envs)]
        self.state_memory = [[] for _ in range(self.num_envs)]
        self.reward_memory = [[] for _ in range(self.num_envs)]
        self.done_memory = [[] for _ in range(self.num_envs)]
        self.if_eval = False
        # VecEnv.__init__(self, self.num_envs,
        #                 self._observation_space, self._action_space)

    def seed(self, seed=0):
        self.wrapper.setSeed(seed)

    def update_rms(self):
        self.obs_rms = self.obs_rms_new

    def getLabelImg(self, depth):
        depth = (np.minimum(depth, 12.0)) / 12.0
        depth_img = Image.fromarray(depth)
        depth_img = depth_img.resize((28, 28))
        label = np.array(depth_img)
        return label

    def getLabelImage(self):
        return self.label_images

    def step(self, action):
        if action.ndim <= 1:
            action = action.reshape((-1, self.act_dim))
        self.wrapper.step(
            action,
            self._state_observation,
            self._single_reward_components,
            self._single_done,
            self._extraInfo,
        )
        # update the mean and variance of the Running Mean STD
        # self.obs_rms_new.update(self._observation)
        t0 = time.time()
        self.render(0)
        # print("render time: ", time.time() - t0)
        depth = self.getDepthImage()
        for i in range(self.num_envs):
            img = depth[i, :].reshape(self.img_height, self.img_width)
            # depth_img = Image.fromarray((np.minimum(img, 12.0)) / 12.0 * 255.0)
            # if i==0:
            #     depth_img.convert('RGB').save('step'+str(i)+str(time.time())+'.jpg')
            img = self.preprocess(img)
            del self.image_memory[i][:1]
            del self.state_memory[i][:1]

            self.image_memory[i].append(img.copy())
            self.state_memory[i].append(self._state_observation[i, :].copy())
        self._observation['image'] = np.stack(self.image_memory)
        self._observation['state'] = np.stack(self.state_memory)
        obs = self._observation

        info = [{} for i in range(self.num_envs)]

        for i in range(self.num_envs):
            self.rewards[i].append(self._single_reward_components[i, -1])
            for j in range(self.rew_dim - 1):
                self.sum_reward_components[i, j] += self._single_reward_components[i, j]
            if self._single_done[i]:
                eprew = sum(self.rewards[i])
                eplen = len(self.rewards[i])
                epinfo = {"r": eprew, "l": eplen}
                for j in range(self.rew_dim - 1):
                    epinfo[self.reward_names[j]] = self.sum_reward_components[i, j]
                    self.sum_reward_components[i, j] = 0.0
                info[i]["episode"] = epinfo
                self.rewards[i].clear()

        return (
            obs,
            self._single_reward_components[:, -1].copy(),
            self._single_done.copy(),
            info.copy(),
        )

    def sample_actions(self):
        actions = []
        for i in range(self.num_envs):
            action = self.action_space.sample().tolist()
            actions.append(action)
        return np.asarray(actions, dtype=np.float64)
    
    def preprocess(self, image):
        depth = (np.minimum(image, 12.0)) / 12.0 * 255.0
        return depth.astype('int')

    def reset(self, random=True):
        self.wrapper.reset(self._state_observation, random)
        # print(self._state_observation)
        self.render(0)
        self.render(0)
        depth = self.getDepthImage()
        for i in range(self.num_envs):
            img = depth[i, :].reshape(self.img_height, self.img_width)
            # depth_img = Image.fromarray((np.minimum(img, 12.0)) / 12.0 * 255.0)
            # if i==0:
            #     depth_img.convert('RGB').save('reset'+str(i)+str(time.time())+'.jpg')
            img = self.preprocess(img)
            del self.image_memory[i][:]
            del self.state_memory[i][:]
            self.image_memory[i] = [img.copy() for _ in range(n_seq)]
            self.state_memory[i] = [self._state_observation[i, :].copy() for _ in range(n_seq)]

        self._observation['image'] = np.stack(self.image_memory)
        self._observation['state'] = np.stack(self.state_memory)
        obs = self._observation
        return obs

    def resetRewCoeff(self):
        return self.wrapper.resetRewCoeff()

    def getObs(self):
        return self._observation

    def reset_and_update_info(self):
        return self.reset(), self._update_epi_info()

    def get_obs_norm(self):
        return self.obs_rms.mean, self.obs_rms.var

    def getProgress(self):
        return self._reward_components[:, 0]

    def getImage(self, rgb=False):
        if rgb:
            self.wrapper.getImage(self._rgb_img_obs, True)
            return self._rgb_img_obs.copy()
        else:
            self.wrapper.getImage(self._gray_img_obs, False)
            return self._gray_img_obs.copy()

    def getDepthImage(self):
        has_img = False
        # while(not has_img):
        has_img = self.wrapper.getDepthImage(self._depth_img_obs)
            # time.sleep(0.01)
        return self._depth_img_obs.copy()

    def getPointClouds(self, dir, id, save_pc):
        self.wrapper.getPointClouds(dir, id, save_pc)

    def readPointClouds(self, id):
        self.wrapper.readPointClouds(id)

    def getSavingState(self):
        return self.wrapper.getSavingState()

    def getReadingState(self):
        return self.wrapper.getReadingState()

    def stepUnity(self, action, send_id):
        receive_id = self.wrapper.stepUnity(
            action,
            self._observation,
            self._reward,
            self._done,
            self._extraInfo,
            send_id,
        )

        return receive_id

    def _normalize_obs(self, obs: np.ndarray, obs_rms: RunningMeanStd) -> np.ndarray:
        """
        Helper to normalize observation.
        :param obs:
        :param obs_rms: associated statistics
        :return: normalized observation
        """
        return (obs - obs_rms.mean) / np.sqrt(obs_rms.var + 1e-8)

    def _unnormalize_obs(self, obs: np.ndarray, obs_rms: RunningMeanStd) -> np.ndarray:
        """
        Helper to unnormalize observation.
        :param obs:
        :param obs_rms: associated statistics
        :return: unnormalized observation
        """
        return (obs * np.sqrt(obs_rms.var + 1e-8)) + obs_rms.mean

    def normalize_obs(self, obs: np.ndarray) -> np.ndarray:
        """
        Normalize observations using this VecNormalize's observations statistics.
        Calling this method does not update statistics.
        """
        # Avoid modifying by reference the original object
        # obs_ = deepcopy(obs)
        # print(self.obs_rms.var)
        obs_ = self._normalize_obs(obs, self.obs_rms).astype(np.float64)
        return obs_

    def getQuadState(self):
        self.wrapper.getQuadState(self._quadstate)
        return self._quadstate

    def getQuadAct(self):
        self.wrapper.getQuadAct(self._quadact)
        return self._quadact

    def getExtraInfo(self):
        return self._extraInfo

    def _update_epi_info(self):
        info = [{} for _ in range(self.num_envs)]
        for i in range(self.num_envs):
            eprew = sum(self.rewards[i])
            eplen = len(self.rewards[i])
            epinfo = {"r": eprew, "l": eplen}
            for j in range(self.rew_dim - 1):
                epinfo[self.reward_names[j]] = self.sum_reward_components[i, j]
                self.sum_reward_components[i, j] = 0.0
            info[i]["episode"] = epinfo
            self.rewards[i].clear()
        return info

    def close(self):
        self.wrapper.close()
        
    def render(self, frame_id, mode="human"):
        return self.wrapper.updateUnity(frame_id)

    def connectUnity(self):
        return self.wrapper.connectUnity()
    
    def initializeConnections(self):
        self.wrapper.initializeConnections()

    def disconnectUnity(self):
        self.wrapper.disconnectUnity()

    def spawnObstacles(self, change_obs, seed=-1, radius=-1.0):
        self.wrapper.spawnObstacles(change_obs, seed, radius)
    
    def ifSceneChanged(self):
        return self.wrapper.ifSceneChanged()

    def env_method(
        self,
        method_name: str,
        *method_args,
        indices: VecEnvIndices = None,
        **method_kwargs
    ) -> List[Any]:
        """Call instance methods of vectorized environments."""
        target_envs = self._get_target_envs(indices)
        return [
            getattr(env_i, method_name)(*method_args, **method_kwargs)
            for env_i in target_envs
        ]

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

    @property
    def num_envs(self):
        return self.wrapper.getNumOfEnvs()

    @property
    def observation_space(self):
        return self._observation_space

    @property
    def action_space(self):
        return self._action_space

    @property
    def extra_info_names(self):
        return self._extraInfoNames

    def start_recording_video(self, file_name):
        raise RuntimeError("This method is not implemented")

    def stop_recording_video(self):
        raise RuntimeError("This method is not implemented")

    def step_async(self):
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

    @staticmethod
    def load(load_path: str, venv: VecEnv) -> "VecNormalize":
        """
        Loads a saved VecNormalize object.

        :param load_path: the path to load from.
        :param venv: the VecEnv to wrap.
        :return:
        """
        with open(load_path, "rb") as file_handler:
            vec_normalize = pickle.load(file_handler)
        vec_normalize.set_venv(venv)
        return vec_normalize

    def save(self, save_path: str) -> None:
        """
        Save current VecNormalize object with
        all running statistics and settings (e.g. clip_obs)

        :param save_path: The path to save to
        """
        with open(save_path, "wb") as file_handler:
            pickle.dump(self, file_handler)

    def save_rms(self, save_dir, n_iter) -> None:
        if not os.path.exists(save_dir):
            os.mkdir(save_dir)
        data_path = save_dir + "/iter_{0:05d}".format(n_iter)
        np.savez(
            data_path,
            mean=np.asarray(self.obs_rms.mean),
            var=np.asarray(self.obs_rms.var),
        )

    def load_rms(self, data_dir) -> None:
        self.mean, self.var = None, None
        np_file = np.load(data_dir)
        #
        self.mean = np_file["mean"]
        self.var = np_file["var"]
        #
        self.obs_rms.mean = np.mean(self.mean, axis=0)
        self.obs_rms.var = np.mean(self.var, axis=0)
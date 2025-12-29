import numpy as np
import os
from datetime import datetime
import time
import isaacgym
from aerial_gym.envs import *
from aerial_gym.utils import get_args, task_registry
import torch

def test_control(args):
    env, env_cfg = task_registry.make_env(name=args.task, args=args)
    command_actions = torch.zeros((env_cfg.env.num_envs, 4))
    env.reset()

    for i in range(0, 50000):
        obs, priviliged_obs, rewards, resets, extras = env.step(command_actions)
        if i % 20 == 0:
            command_actions[:, 0] = 0.8
            command_actions[:, 1] = 0.5
            command_actions[:, 3] = 0.5
        # _, _quat, _vel, _angvel = env.getStates()

        if i % 1000 == 0:
            print("Resetting environment")
            if priviliged_obs is None:
                print("Privileged observation is None")
            else:
                print("Shape of privileged observation tensor", priviliged_obs.shape)
            print("------------------")
            env.reset()

if __name__ == '__main__':
    args = get_args()
    test_control(args)

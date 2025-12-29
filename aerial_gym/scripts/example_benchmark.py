import numpy as np
import os
from datetime import datetime
import time
import isaacgym
from aerial_gym.envs import *
from aerial_gym.utils import get_args, task_registry
import torch

def sample_command(args):

    env, env_cfg = task_registry.make_env(name=args.task, args=args)
    print("Number of environments", env_cfg.env.num_envs)
    command_actions = torch.zeros((env_cfg.env.num_envs, env_cfg.env.num_actions))
    command_actions[:, 0] = 1.0
    command_actions[:, 1] = 0.0
    command_actions[:, 2] = 0.0

    env.reset()
    step_ctr = 0
    start_time = time.time()
    for i in range(0, 50000):
        obs, priviliged_obs, rewards, resets, extras = env.step(command_actions)
        step_ctr += 1
        print("Done", i)
        if i % 100 == 0:
            print("Resetting environment")
            # print("Shape of observation tensor", obs.shape)
            # print("Shape of reward tensor", rewards.shape)
            # print("Shape of reset tensor", resets.shape)
            if priviliged_obs is None:
                print("Privileged observation is None")
            else:
                print("Shape of privileged observation tensor", priviliged_obs.shape)
            print("------------------")
            env.reset()
            end_time = time.time()
            print("Time taken", end_time - start_time)
            print("Env steps taken", step_ctr)
            print("Number of drone steps: ", env_cfg.env.num_envs * step_ctr)
            print("FPS", env_cfg.env.num_envs * step_ctr / (end_time - start_time))
            start_time = time.time()
            step_ctr = 0

if __name__ == '__main__':
    args = get_args()
    sample_command(args)

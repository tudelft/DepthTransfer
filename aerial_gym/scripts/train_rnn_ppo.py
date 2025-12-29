import numpy as np
import os
import argparse
from os.path import join, exists
from datetime import datetime
import time
from gymnasium import spaces
from isaacgym import gymutil
from aerial_gym.envs.mavrl import mavrl_vec_rnn_env
from aerial_gym.utils import get_args, task_registry
from aerial_gym.mav_baselines.torch.ppo.normal_ppo import PPO
import torch
from omegaconf import OmegaConf


def learning_rate_schedule(progress_remaining):
    """
    Custom learning rate schedule.
    :param progress_remaining: A float, the proportion of training remaining (1 at the beginning, 0 at the end)
    :return: The learning rate as a float.
    """
    # Example: Linearly decreasing learning rate
    return 2e-4 * progress_remaining

def main():
    add_args = [{"name": "--train", "type": bool, "default": True, "help": "Train the policy or evaluate the policy"},
        {"name": "--render", "type": bool, "default": True, "help": "Render with Unity"},
        {"name": "--trial", "type": int, "default": 1, "help": "PPO trial number"},
        {"name": "--iter", "type": int, "default": 100, "help": "PPO iter number"},
        {"name": "--retrain", "type": bool, "default": False, "help": "if retrain"},
        {"name": "--nocontrol", "type": bool, "default": False, "help": "if load action and value net parameters"},
        {"name": "--rollouts", "type": int, "default": 1000, "help": "Number of rollouts"},
        {"name": "--dir", "type": str, "default": "./datasets", "help": "Where to place rollouts"},
        {"name": "--logdir", "type": str, "default": "./exp_dir", "help": "Directory where results are logged"},
    ]
    args = get_args(add_args)
    env, env_cfg = task_registry.make_env(name=args.task, args=args)
    print("Number of environments", env_cfg.env.num_envs)
    train_env = mavrl_vec_rnn_env.MavrlEnvVecRNN(env, reconstruction_members=[True, True, False])

    rsg_root = os.path.dirname(os.path.abspath(__file__))
    log_dir = rsg_root + "/../saved"
    device = train_env.device

    vae_logdir = "../saved/LSTM_Resnet_VAE_02/"
    # vae_logdir = "../saved/LSTM_Resnet_VAE_Zoo/"
    rnn_vae_file = join(vae_logdir, 'Encoder', 'iter_03950.pth')
    assert exists(rnn_vae_file), "No trained VAE in the logdir..."
    state_vae = torch.load(rnn_vae_file, map_location=device)

    encoder = {}
    lstm_weights = {}
    decoder = {}
    mu_linear = {}
    for key in state_vae.keys():
        if key.startswith('features_encoder.'):
            new_key = key.replace('features_encoder.', '')
            encoder[new_key] = state_vae[key]
        elif key.startswith('features_decoder.'):
            new_key = key.replace('features_decoder.', '')
            decoder[new_key] = state_vae[key]
        elif key.startswith('mu_linear.'):
            new_key = key.replace('mu_linear.', '')
            mu_linear[new_key] = state_vae[key]
        elif key.startswith('lstm_actor.'):
            new_key = key.replace('lstm_actor.', '')
            lstm_weights[new_key] = state_vae[key]
    train_env.load_features_extractor(encoder, lstm_weights, mu_linear, decoder)
    
    policy = "MlpPolicy"

    if args.task == "mavrl_zoo" or args.task == "mavrl_task_agile":
        n_steps = env_cfg.env.max_episode_length
    elif args.task == "mavrl_task":
        n_steps = int(env_cfg.env.max_episode_length / 2)
    if args.train:
        model = PPO(
            tensorboard_log=log_dir,
            policy=policy,
            policy_kwargs=dict(
                activation_fn=torch.nn.ReLU,
                net_arch=dict(pi=[256, 256], vf=[512, 512]),
                log_std_init=-0.0,
                # use_beta = False,
            ),
            env=train_env,
            learning_rate=learning_rate_schedule,
            use_tanh_act=True,
            gae_lambda=0.95,
            gamma=0.99,
            n_steps=n_steps,
            ent_coef=0.0,
            vf_coef=0.2,
            max_grad_norm=0.5,
            batch_size=16000,
            clip_range=0.2,
            use_sde=False,  # don't use (gSDE), doesn't work
            device=device,
            verbose=1,
            if_change_maps=True,
            save_encoder=True,
        )
        model.learn(total_timesteps=int(8.0e7), log_interval=(10, 20), normailize_state = env_cfg.LatentSpaceCfg.normalize_obs, if_easy_start=True, 
                    curriculum_steps=8e6, reset_goal_threshold=True)

if __name__ == '__main__':
    main()
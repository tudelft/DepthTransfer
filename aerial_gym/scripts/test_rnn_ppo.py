import numpy as np
import random
import os
import argparse
from os.path import join, exists
from datetime import datetime
import time
from gymnasium import spaces
from isaacgym import gymutil
from aerial_gym.envs.base.zoo_task_config import ZooTaskCfg
from aerial_gym.envs.base.mavrl_task_config import MAVRLTaskCfg
from aerial_gym.envs.base.mavrl_task_agile_config import MAVRLTaskAgileCfg
from aerial_gym.envs.mavrl import mavrl_vec_rnn_env
from aerial_gym.utils import get_args, task_registry
from aerial_gym.mav_baselines.torch.recurrent_ppo.policies import MlPolicy
from aerial_gym.mav_baselines.torch.ppo.normal_ppo import PPO

import torch
# from omegaconf import OmegaConf

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)

def main():
    add_args = [{"name": "--train", "type": bool, "default": False, "help": "Train the policy or evaluate the policy"},
        {"name": "--render", "type": bool, "default": True, "help": "Render with Unity"},
        {"name": "--trial", "type": int, "default": 1, "help": "PPO trial number"},
        {"name": "--iter", "type": int, "default": 100, "help": "PPO iter number"},
        {"name": "--retrain", "type": bool, "default": False, "help": "if retrain"},
        {"name": "--nocontrol", "type": bool, "default": False, "help": "if load action and value net parameters"},
        {"name": "--rollouts", "type": int, "default": 1000, "help": "Number of rollouts"},
        {"name": "--dir", "type": str, "default": "./datasets", "help": "Where to place rollouts"},
        {"name": "--logdir", "type": str, "default": "./exp_dir", "help": "Directory where results are logged"},
        {"name": "--ckpt", "type": str, "default": None, "help": "Path to the checkpoint"},
        {"name": "--recon", "type": bool, "default": True, "help": "Path to the checkpoint"},
        {"name": "--refined_encoder", "type": bool, "default": False, "help": "If use refined encoder"},
    ]
    args = get_args(add_args)
    set_seed(1)
    if args.task == "mavrl_zoo":
        env_cfg = ZooTaskCfg()
        env_cfg.camera_params.use_stereo_vision = False
        env_cfg.env.flight_upper_bound = [15.0, 7.5, 2.0]
        env_cfg.env.create_texture = True
        env_cfg.env.enable_pc_loader = True
        # env_cfg.env.poisson_radius_origin = 1.2
    elif args.task == "mavrl_task":
        env_cfg = MAVRLTaskCfg()
        env_cfg.camera_params.use_stereo_vision = False
        env_cfg.env.create_texture = True
        env_cfg.env.goal_arrive_threshold = 1.2
        env_cfg.env.enable_pc_loader = True
        env_cfg.env.max_episode_length = 500
        # env_cfg.tree_asset_params.num_assets = 10
        # env_cfg.forest_asset_params.num_assets = 35
        # env_cfg.env.poisson_radius_origin = 3.2
        # env_cfg.asset_config.free_space = 6
        # env_cfg.tree_asset_params.num_assets = 45
        # env_cfg.forest_asset_params.num_assets = 5
        # env_cfg.env.poisson_radius_origin = 2.6
    elif args.task == "mavrl_task_agile":
        env_cfg = MAVRLTaskAgileCfg()
        env_cfg.camera_params.use_stereo_vision = False
        env_cfg.env.create_texture = True
        env_cfg.env.goal_arrive_threshold = 1.2
        env_cfg.env.enable_pc_loader = True
        env_cfg.env.max_episode_length = 500

    env, env_cfg = task_registry.make_env(name=args.task, args=args, env_cfg=env_cfg)
    print("Number of environments", env_cfg.env.num_envs)
    test_env = mavrl_vec_rnn_env.MavrlEnvVecRNN(env, reconstruction_members=[True, True, False])
    test_env.set_seed(1)
    rsg_root = os.path.dirname(os.path.abspath(__file__))
    log_dir = rsg_root + "/../saved/"
    device = test_env.device
    ckpt = log_dir + args.ckpt
    lstm = ckpt + "/Encoder/lstm_iter_{0:05d}.pth".format(args.iter)
    mu_linear = ckpt + "/Encoder/mu_linear_iter_{0:05d}.pth".format(args.iter)
    lstm_weights = torch.load(lstm, map_location=device)
    mu_linear_weights = torch.load(mu_linear, map_location=device)
    if not args.refined_encoder:
        encoder = ckpt + "/Encoder/encoder_iter_{0:05d}.pth".format(args.iter)
        encoder_weights = torch.load(encoder, map_location=device)
        if args.recon:
            deconder = log_dir + "LSTM_Resnet_VAE_02/Encoder/iter_03950.pth"
            pre_weights = torch.load(deconder, map_location=device)
            decoder_weights = {}
            # if args.task == "mavrl_zoo":
            #     decoder = log_dir + "../exp_vae_320/vae_resnet_16ch_zoo/checkpoint.tar"
            # elif args.task == "mavrl_task":
            #     decoder = log_dir + "../exp_vae_320/vae_resnet_16ch_outdoor_stereo/checkpoint.tar"
            # decoder_weights_load = torch.load(decoder, map_location=device)
            # for key in decoder_weights_load["state_dict"].keys():
            #     if key.startswith('decoder.'):
            #         decoder_weights[key] = decoder_weights_load["state_dict"][key]
            for key in pre_weights.keys():
                if key.startswith('features_decoder.'):
                    decoder_key = key.replace('features_decoder.', '')
                    decoder_weights[decoder_key] = pre_weights[key]
        else:
            decoder_weights = None
    else:
        if args.recon:
            # decoder_weights = {}
            # if args.task == "mavrl_zoo":
            #     decoder = log_dir + "../exp_vae_320/vae_resnet_from_latent_zoo/vae_500.tar"
            # elif args.task == "mavrl_task":
            #     decoder = log_dir + "../exp_vae_320/vae_resnet_from_latent_seed_12/vae_500.tar"
            # # decoder = log_dir + "../exp_vae_320/vae_resnet_16ch_outdoor_stereo_refined/checkpoint.tar"
            # decoder_weights_load = torch.load(decoder, map_location=device)
            # for key in decoder_weights_load["state_dict"].keys():
            #     if key.startswith('decoder.'):
            #         decoder_weights[key] = decoder_weights_load["state_dict"][key]
            deconder = log_dir + "LSTM_Resnet_VAE_02/Encoder/iter_03950.pth"
            pre_weights = torch.load(deconder, map_location=device)
            decoder_weights = {}
            for key in pre_weights.keys():
                if key.startswith('features_decoder.'):
                    decoder_key = key.replace('features_decoder.', '')
                    decoder_weights[decoder_key] = pre_weights[key]
        else:
            decoder_weights = None

        if args.task == "mavrl_zoo":
            encoder = log_dir + "../exp_vae_320/vae_resnet_from_latent_zoo/vae_500.tar"
        elif args.task == "mavrl_task" or args.task == "mavrl_task_agile":
            encoder = log_dir + "../exp_vae_320/vae_resnet_from_latent_seed_12/vae_500.tar"
        encoder_weights_load = torch.load(encoder, map_location=device)
        encoder_weights = {}
        # decoder_weights = {}
        for key in encoder_weights_load["state_dict"].keys():
            if key.startswith('encoder.'):
                encoder_weights[key] = encoder_weights_load["state_dict"][key]
            # elif key.startswith('decoder.'):
            #     decoder_weights[key] = encoder_weights_load["state_dict"][key]
        # if not args.recon:
        #     decoder_weights = None

    policy = ckpt + "/Policy/iter_{0:05d}.pth".format(args.iter)
    rms = ckpt + "/RMS/iter_{0:05d}.pth".format(args.iter)
    policy_weights = torch.load(policy, map_location=device)

    test_env.load_features_extractor(encoder_weights, lstm_weights, mu_linear_weights, decoder_weights)
    test_env.load_rms(rms, env_cfg.env.num_envs)

    policy = MlPolicy(**policy_weights["data"])
    policy.action_net = torch.nn.Sequential(policy.action_net, torch.nn.Tanh())
    policy.load_state_dict(policy_weights["state_dict"], strict=True)
    policy.to(device)

    if not args.train:
        model = PPO(
            tensorboard_log=log_dir,
            policy=policy,
            env=test_env,
            use_tanh_act=True,
            gae_lambda=0.95,
            gamma=0.99,
            n_steps=int(env_cfg.env.max_episode_length * 2),
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
            test_mode=True,
        )
        # model.eval(total_timesteps=1e6)
        model.learn(total_timesteps=int(6.0e7), log_interval=(10, 20), normailize_state = env_cfg.LatentSpaceCfg.normalize_obs, if_easy_start=False, curriculum_steps=8e6)
if __name__ == '__main__':
    main()
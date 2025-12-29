import numpy as np
import os
import argparse
from datetime import datetime
import time
from gymnasium import spaces
from isaacgym import gymutil
from aerial_gym.envs.mavrl import mavrl_vec_env
from aerial_gym.utils import get_args, task_registry
from aerial_gym.mav_baselines.torch.recurrent_ppo.policies import MlpLstmPolicy
from aerial_gym.mav_baselines.torch.recurrent_ppo.ppo_recurrent import RecurrentPPO
import torch
from omegaconf import OmegaConf

def learning_rate_schedule(progress_remaining):
    """
    Custom learning rate schedule.
    :param progress_remaining: A float, the proportion of training remaining (1 at the beginning, 0 at the end)
    :return: The learning rate as a float.
    """
    # Example: Linearly decreasing learning rate
    return 5e-6 * progress_remaining

def main():
    add_args = [{"name": "--train", "type": bool, "default": True, "help": "Train the policy or evaluate the policy"},
        {"name": "--render", "type": bool, "default": True, "help": "Render with Unity"},
        {"name": "--trial", "type": int, "default": 1, "help": "PPO trial number"},
        {"name": "--iter", "type": int, "default": 100, "help": "PPO iter number"},
        {"name": "--retrain", "type": bool, "default": True, "help": "if retrain"},
        {"name": "--nocontrol", "type": bool, "default": False, "help": "if load action and value net parameters"},
        {"name": "--rollouts", "type": int, "default": 1000, "help": "Number of rollouts"},
        {"name": "--dir", "type": str, "default": "./datasets", "help": "Where to place rollouts"},
        {"name": "--logdir", "type": str, "default": "./exp_dir", "help": "Directory where results are logged"},
    ]
    args = get_args(add_args)
    env, env_cfg = task_registry.make_env(name=args.task, args=args)
    print("Number of environments", env_cfg.env.num_envs)
    train_env = mavrl_vec_env.MavrlEnvVec(env)

    rsg_root = os.path.dirname(os.path.abspath(__file__))
    log_dir = rsg_root + "/../saved"
    device = train_env.device

    if (args.retrain or not args.train):
        policy_weight = "./../saved/RecurrentPPO_{0}/Policy/iter_{1:05d}.pth".format(args.trial, args.iter)
        encoder_weight = "./../saved/RecurrentPPO_{0}/Encoder/iter_{1:05d}.pth".format(args.trial, args.iter)
        env_rms = "./../saved/RecurrentPPO_{0}/RMS/iter_{1:05d}.pth".format(args.trial, args.iter)
        train_env.load_rms(env_rms, env_cfg.env.num_envs)

        saved_variables = torch.load(policy_weight, map_location=device)
        encoder_variables = torch.load(encoder_weight, map_location=device)

        # print saved_variables["state_dict"].keys()
        # for key in saved_variables["state_dict"].keys():
        #     print(key)

        # print("encoder_variables: ", encoder_variables.keys())
        saved_variables["data"].pop('features_extractor_kwargs')
        # saved_variables["data"].pop('features_extractor_class')
        # saved_variables["data"]['observation_space'] = env.observation_space
        # saved_variables["data"]['action_space'] = env.action_space
        # if env_cfg.control.controller == "lee_velocity_control":
        # saved_variables["state_dict"]['log_std'] = torch.tensor([-0.5, -0.5, -0.5, -0.5], device=device)
        features_extractor_params = None
        if env_cfg.LatentSpaceCfg.use_resnet_vae:
            vae_config_path = '../mav_baselines/torch/controlNet/models/encoder.yaml'
            features_extractor_params = {"ddconfig": OmegaConf.load(vae_config_path)['ddconfig']}
        policy = MlpLstmPolicy(features_extractor_kwargs = features_extractor_params,
                               features_dim=env_cfg.LatentSpaceCfg.vae_dims, 
                               enable_critic_lstm = False,
                               shared_lstm = True,
                               states_dim=14,
                               use_kl_loss=env_cfg.LatentSpaceCfg.use_kl_latent_loss,
                                **saved_variables["data"])
        policy.action_net = torch.nn.Sequential(policy.action_net, torch.nn.Tanh())
        policy.reconstruction_members = [1, 1, 0]
        # print("saved_variables: ", saved_variables["state_dict"].keys())
        if args.nocontrol:
            saved_variables["state_dict"].pop('action_net.0.weight')
            saved_variables["state_dict"].pop('action_net.0.bias')
            saved_variables["state_dict"].pop('value_net.weight')
            saved_variables["state_dict"].pop('value_net.bias')
            saved_variables["state_dict"].pop('mlp_extractor.value_net.0.weight')
            saved_variables["state_dict"].pop('mlp_extractor.value_net.0.bias')
            saved_variables["state_dict"].pop('mlp_extractor.policy_net.0.weight')
            saved_variables["state_dict"].pop('mlp_extractor.policy_net.0.bias')
            saved_variables["state_dict"].pop('mlp_extractor.policy_net.2.weight')
            saved_variables["state_dict"].pop('mlp_extractor.policy_net.2.bias')
            saved_variables["state_dict"].pop('mlp_extractor.value_net.2.weight')
            saved_variables["state_dict"].pop('mlp_extractor.value_net.2.bias')

        # encoer = {
        #     'conv1.weight': saved_variables["state_dict"]['features_extractor.conv1.weight'],
        #     'conv1.bias': saved_variables["state_dict"]['features_extractor.conv1.bias'],
        #     'conv2.weight': saved_variables["state_dict"]['features_extractor.conv2.weight'],
        #     'conv2.bias': saved_variables["state_dict"]['features_extractor.conv2.bias'],
        #     'conv3.weight': saved_variables["state_dict"]['features_extractor.conv3.weight'],
        #     'conv3.bias': saved_variables["state_dict"]['features_extractor.conv3.bias'],
        #     'conv4.weight': saved_variables["state_dict"]['features_extractor.conv4.weight'],
        #     'conv4.bias': saved_variables["state_dict"]['features_extractor.conv4.bias'],
        #     'conv5.weight': saved_variables["state_dict"]['features_extractor.conv5.weight'],
        #     'conv5.bias': saved_variables["state_dict"]['features_extractor.conv5.bias'],
        #     'conv6.weight': saved_variables["state_dict"]['features_extractor.conv6.weight'],
        #     'conv6.bias': saved_variables["state_dict"]['features_extractor.conv6.bias'],
        #     'linear.weight': saved_variables["state_dict"]['features_extractor.linear.weight'],
        #     'linear.bias': saved_variables["state_dict"]['features_extractor.linear.bias'],
        #     'fc_logsigma.weight': saved_variables["state_dict"]['features_extractor.fc_logsigma.weight'],
        #     'fc_logsigma.bias': saved_variables["state_dict"]['features_extractor.fc_logsigma.bias'],
        # }
        train_env.load_features_extractor(encoder_variables)

        policy.load_state_dict(saved_variables["state_dict"], strict=False)
        # policy.log_std_init = -0.5
        policy.to(device)
    else:
        policy = "MlpLstmPolicy"

    if args.train:
        model = RecurrentPPO(
            tensorboard_log=log_dir,
            policy=policy,
            policy_kwargs=dict(
                activation_fn=torch.nn.ReLU,
                # net_arch=[dict(pi=[256, 256], vf=[512, 512])],
                net_arch=dict(pi=[256, 256], vf=[512, 512]),
                log_std_init=-0.0,
                use_beta = False,
            ),
            env=train_env,
            learning_rate=learning_rate_schedule,
            use_tanh_act=True,
            gae_lambda=0.95,
            gamma=0.99,
            n_steps=env_cfg.env.max_episode_length,
            n_seq=1,
            ent_coef=0.0,
            vf_coef=0.2,
            max_grad_norm=0.5,
            lstm_layer=1,
            batch_size=16000,
            clip_range=0.2,
            use_sde=False,  # don't use (gSDE), doesn't work
            retrain=args.retrain,
            device=device,
            verbose=1,
            states_dim=14,
            features_dim=env_cfg.LatentSpaceCfg.vae_dims,
            if_change_maps=True,
            save_encoder=True,
            use_kl_latent_loss=env_cfg.LatentSpaceCfg.use_kl_latent_loss,
        )
        model.learn(total_timesteps=int(6.4e7), log_interval=(10, 20), 
                    normailize_state = env_cfg.LatentSpaceCfg.normalize_obs, 
                    if_easy_start=False, deterministic=False)

if __name__ == '__main__':
    main()

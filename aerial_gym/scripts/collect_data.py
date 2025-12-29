import os
from isaacgym import gymutil
from aerial_gym.envs.mavrl import mavrl_vec_env_v2
from aerial_gym.utils import get_args, task_registry
from aerial_gym.mav_baselines.torch.recurrent_ppo.policies import MultiInputLstmPolicy
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
    return 5e-4 * progress_remaining

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
    train_env = mavrl_vec_env_v2.MavrlEnvVecV2(env)

    rsg_root = os.path.dirname(os.path.abspath(__file__))
    log_dir = rsg_root + "/../saved"
    device = train_env.device

    policy = "MultiInputLstmPolicy"

    if env_cfg.LatentSpaceCfg.use_resnet_vae:
        vae_config_path = '../mav_baselines/torch/controlNet/models/encoder.yaml'
        features_extractor_params = {"ddconfig": OmegaConf.load(vae_config_path)['ddconfig']}

    pre_control_policy = None
    if not args.nocontrol:
        policy_weight = "./../saved/RecurrentPPO_{0}/Policy/iter_{1:05d}.pth".format(args.trial, args.iter)
        saved_variables = torch.load(policy_weight, map_location=device)
        env_rms = "./../saved/RecurrentPPO_{0}/RMS/iter_{1:05d}.pth".format(args.trial, args.iter)
        train_env.load_rms(env_rms, env_cfg.env.num_envs)

        encoder_weight = "./../saved/RecurrentPPO_{0}/Encoder/iter_{1:05d}.pth".format(args.trial, args.iter)
        encoder_variables = torch.load(encoder_weight, map_location=device)
        # print(saved_variables["state_dict"].keys())
        pre_control_policy = {
            'action_net.0.weight': saved_variables["state_dict"]['action_net.0.weight'],
            'action_net.0.bias': saved_variables["state_dict"]['action_net.0.bias'],
            'value_net.weight': saved_variables["state_dict"]['value_net.weight'],
            'value_net.bias': saved_variables["state_dict"]['value_net.bias'],
            'mlp_extractor.value_net.0.weight': saved_variables["state_dict"]['mlp_extractor.value_net.0.weight'],
            'mlp_extractor.value_net.0.bias': saved_variables["state_dict"]['mlp_extractor.value_net.0.bias'],
            'mlp_extractor.policy_net.0.weight': saved_variables["state_dict"]['mlp_extractor.policy_net.0.weight'],
            'mlp_extractor.policy_net.0.bias': saved_variables["state_dict"]['mlp_extractor.policy_net.0.bias'],
            'mlp_extractor.policy_net.2.weight': saved_variables["state_dict"]['mlp_extractor.policy_net.2.weight'],
            'mlp_extractor.policy_net.2.bias': saved_variables["state_dict"]['mlp_extractor.policy_net.2.bias'],
            'mlp_extractor.value_net.2.weight': saved_variables["state_dict"]['mlp_extractor.value_net.2.weight'],
            'mlp_extractor.value_net.2.bias': saved_variables["state_dict"]['mlp_extractor.value_net.2.bias'],
            'log_std': saved_variables["state_dict"]['log_std'],
            'lstm_actor.weight_ih_l0': saved_variables["state_dict"]['lstm_actor.weight_ih_l0'],
            'lstm_actor.weight_hh_l0': saved_variables["state_dict"]['lstm_actor.weight_hh_l0'],
            'lstm_actor.bias_ih_l0': saved_variables["state_dict"]['lstm_actor.bias_ih_l0'],
            'lstm_actor.bias_hh_l0': saved_variables["state_dict"]['lstm_actor.bias_hh_l0'],
        }
        for key in encoder_variables.keys():
            pre_control_policy['features_extractor.' + key] = encoder_variables[key]

    if args.train:
        model = RecurrentPPO(
            tensorboard_log=log_dir,
            policy=policy,
            policy_kwargs=dict(
                features_extractor_kwargs = features_extractor_params,
                activation_fn=torch.nn.ReLU,
                net_arch=dict(pi=[256, 256], vf=[512, 512]),
                log_std_init=-0.0,
                use_beta = False,
            ),
            env=train_env,
            learning_rate=learning_rate_schedule,
            use_tanh_act=True,
            gae_lambda=0.95,
            gamma=0.99,
            n_steps=int(env_cfg.env.max_episode_length / 2),
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
            only_lstm_training=True,
            features_dim=env_cfg.LatentSpaceCfg.vae_dims,
            states_dim=14,
            if_change_maps=True,
            reconstruction_members=[False, False, True],
            save_lstm_dataset=True,
            control_policy=pre_control_policy,
            use_kl_latent_loss=env_cfg.LatentSpaceCfg.use_kl_latent_loss,
        )
        model.learn_lstm(total_timesteps=int(4e7), log_interval=(10, 10))

if __name__ == '__main__':
    main()

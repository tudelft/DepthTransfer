import os
from os.path import join, exists
from aerial_gym.envs.mavrl import mavrl_vec_env
from aerial_gym.envs.base.zoo_task_config import ZooTaskCfg
from aerial_gym.envs.base.mavrl_task_config import MAVRLTaskCfg
from aerial_gym.utils import get_args, task_registry
from aerial_gym.mav_baselines.torch.ppo.normal_ppo import PPO
import torch

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
        {"name": "--trial", "type": int, "default": 1, "help": "PPO trial number"},
        {"name": "--iter", "type": int, "default": 100, "help": "PPO iter number"},
        {"name": "--load_vae", "type": bool, "default": False, "help": "if load vae"},
        {"name": "--rollouts", "type": int, "default": 1000, "help": "Number of rollouts"},
        {"name": "--logdir", "type": str, "default": "./exp_dir", "help": "Directory where results are logged"},
    ]
    args = get_args(add_args)
    if args.task == "mavrl_zoo":
        env_cfg = ZooTaskCfg()
    elif args.task == "mavrl_task":
        env_cfg = MAVRLTaskCfg()
    if args.load_vae:
        curriculum_steps = 6e6
    else:
        env_cfg.robot_asset.collision_mask = 1
        curriculum_steps = 0
    env, env_cfg = task_registry.make_env(name=args.task, args=args, env_cfg=env_cfg)
    train_env = mavrl_vec_env.MavrlEnvVec(env)

    rsg_root = os.path.dirname(os.path.abspath(__file__))
    log_dir = rsg_root + "/../saved"
    device = train_env.device

    if args.load_vae:
        vae_logdir = "../exp_vae_320/"
        if env_cfg.LatentSpaceCfg.use_resnet_vae:
            if args.task == "mavrl_zoo":
                vae_file = join(vae_logdir, 'vae_resnet_16ch_zoo', 'checkpoint.tar')
            elif args.task == "mavrl_task":
                vae_file = join(vae_logdir, 'vae_resnet_16ch_outdoor', 'checkpoint.tar')
                # vae_file = join(vae_logdir, 'vae_resnet_16ch_outdoor_stereo', 'checkpoint.tar')
        else:
            vae_file = join(vae_logdir, 'vae', 'best.tar')
        assert exists(vae_file), "No trained VAE in the logdir..."
        state_vae = torch.load(vae_file, map_location=device)

        encoder = {}
        for key in state_vae["state_dict"].keys():
            if key.startswith('encoder'):
                encoder[key] = state_vae["state_dict"][key]
            if key.startswith('mu_linear') or key.startswith('logvar_linear'):
                encoder[key] = state_vae["state_dict"][key]
        train_env.load_features_extractor(encoder)

    if args.task == "mavrl_zoo":
        n_steps = env_cfg.env.max_episode_length
    elif args.task == "mavrl_task":
        n_steps = int(env_cfg.env.max_episode_length / 2)

    policy = "MlpPolicy"
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
        model.learn(total_timesteps=int(6.0e7), log_interval=(10, 20), 
                    normailize_state = env_cfg.LatentSpaceCfg.normalize_obs, 
                    if_easy_start=True, curriculum_steps=curriculum_steps, reset_goal_threshold=True)

if __name__ == '__main__':
    main()
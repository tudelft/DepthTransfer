import os
from os.path import join, exists
import numpy as np
from gymnasium import spaces
from isaacgym import gymutil
from aerial_gym.envs.base.mavrl_task_config import MAVRLTaskCfg
from aerial_gym.utils import get_args, task_registry
from aerial_gym.mav_baselines.torch.recurrent_ppo.policies import MultiInputLstmPolicy
from aerial_gym.mav_baselines.torch.recurrent_ppo.ppo_recurrent import RecurrentPPO
import torch
from stable_baselines3.common.utils import get_device
from omegaconf import OmegaConf

def learning_rate_schedule(progress_remaining):
    """
    Custom learning rate schedule.
    :param progress_remaining: A float, the proportion of training remaining (1 at the beginning, 0 at the end)
    :return: The learning rate as a float.
    """
    # Example: Linearly decreasing learning rate
    return 5e-4 * progress_remaining

def configure_random_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)

def main():
    add_args = [{"name": "--train", "type": bool, "default": True, "help": "Train the policy or evaluate the policy"},
        {"name": "--render", "type": bool, "default": True, "help": "Render with Unity"},
        {"name": "--trial", "type": int, "default": 1, "help": "PPO trial number"},
        {"name": "--iter", "type": int, "default": 100, "help": "PPO iter number"},
        {"name": "--retrain", "type": bool, "default": True, "help": "if retrain"},
        {"name": "--nocontrol", "type": bool, "default": False, "help": "if load action and value net parameters"},
        {"name": "--rollouts", "type": int, "default": 1000, "help": "Number of rollouts"},
        {"name": "--dir", "type": str, "default": "../saved/dataset_outdoor_env", "help": "Where to place rollouts"},
        {"name": "--lstm_exp", "type": str, "default": "LSTM", "help": "Directory where results are logged"},
        # {"name": "--recon", "nargs": '+', "type": int, "default": "[0, 0, 1]", "help": "Recurrent model"},
    ]
    recon = [1, 1, 0]
    args = get_args(add_args)
    config = MAVRLTaskCfg()
    # env = mavrl_vec_env_v2.MavrlEnvVecV2(env)

    rsg_root = os.path.dirname(os.path.abspath(__file__))
    log_dir = rsg_root + "/../saved"
    print("log_dir: ", log_dir)
    device = get_device("auto")

    vae_logdir = "../exp_vae_320/"
    if config.LatentSpaceCfg.use_resnet_vae:
        vae_file = join(vae_logdir, 'vae_resnet_16ch', 'checkpoint.tar')
    else:
        vae_file = join(vae_logdir, 'vae', 'best.tar')
    assert exists(vae_file), "No trained VAE in the logdir..."
    state_vae = torch.load(vae_file, map_location=device)
    print("Loading VAE at epoch {} "
        "with test error {}".format(state_vae['epoch'], state_vae['precision']))
    
    # policy = "MultiInputLstmPolicy"
    policy_weight = "./../saved/RecurrentPPO_{0}/Policy/iter_{1:05d}.pth".format(args.trial, args.iter)
    saved_variables = torch.load(policy_weight, map_location=device)
    # print(saved_variables["data"])
    saved_variables["data"].pop('features_extractor_class')
    saved_variables["data"].pop('features_extractor_kwargs')
    saved_variables["data"]['observation_space'] = spaces.Dict({
            'image': spaces.Box(
                low=0,
                high=255,
                shape=(1, 224, 320),
                dtype='uint8'
            ),
            'state': spaces.Box(
                np.ones([1, config.LatentSpaceCfg.state_dims]) * -np.inf,
                np.ones([1, config.LatentSpaceCfg.state_dims]) * np.inf,
                dtype=np.float64,
            ),  
        })
    features_extractor_params = None
    if config.LatentSpaceCfg.use_resnet_vae:
        vae_config_path = '../mav_baselines/torch/controlNet/models/encoder.yaml'
        features_extractor_params = {"ddconfig": OmegaConf.load(vae_config_path)['ddconfig']}
    policy = MultiInputLstmPolicy(features_extractor_kwargs = features_extractor_params,
                                features_dim=config.LatentSpaceCfg.vae_dims,
                                reconstruction_members=recon,
                                reconstruction_steps=10,
                                states_dim=14,
                                **saved_variables["data"])
    policy.action_net = torch.nn.Sequential(policy.action_net, torch.nn.Tanh())
    policy.load_state_dict(saved_variables["state_dict"], strict=True)
    policy.to(device)
    # for name, parameters in policy.named_parameters():
    #     print(name,':',parameters.size())
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

        # 'features_extractor.encoder.0.weight': state_vae['state_dict']['encoder.0.weight'],
        # 'features_extractor.encoder.0.bias': state_vae['state_dict']['encoder.0.bias'],
        # 'features_extractor.encoder.2.weight': state_vae['state_dict']['encoder.2.weight'],
        # 'features_extractor.encoder.2.bias': state_vae['state_dict']['encoder.2.bias'],
        # 'features_extractor.encoder.4.weight': state_vae['state_dict']['encoder.4.weight'],
        # 'features_extractor.encoder.4.bias': state_vae['state_dict']['encoder.4.bias'],
        # 'features_extractor.encoder.6.weight': state_vae['state_dict']['encoder.6.weight'],
        # 'features_extractor.encoder.6.bias': state_vae['state_dict']['encoder.6.bias'],
        # 'features_extractor.encoder.8.weight': state_vae['state_dict']['encoder.8.weight'],
        # 'features_extractor.encoder.8.bias': state_vae['state_dict']['encoder.8.bias'],
        # 'features_extractor.encoder.10.weight': state_vae['state_dict']['encoder.10.weight'],
        # 'features_extractor.encoder.10.bias': state_vae['state_dict']['encoder.10.bias'],
        # 'features_extractor.encoder.12.weight': state_vae['state_dict']['encoder.12.weight'],
        # 'features_extractor.encoder.12.bias': state_vae['state_dict']['encoder.12.bias'],
        # 'features_extractor.mu_linear.weight': state_vae['state_dict']['mu_linear.weight'],
        # 'features_extractor.mu_linear.bias': state_vae['state_dict']['mu_linear.bias'],
        # 'features_extractor.logvar_linear.weight': state_vae['state_dict']['logvar_linear.weight'],
        # 'features_extractor.logvar_linear.bias': state_vae['state_dict']['logvar_linear.bias'],
    }
    for key in policy.state_dict().keys():
        # if the key is start with 'features_extractor', then replace it with 'state_vae'
        if key.startswith('features_extractor'):
            old_key = key.replace('features_extractor.', '')
            pre_control_policy[key] = state_vae["state_dict"][old_key]
        
    model = RecurrentPPO(
        tensorboard_log=log_dir,
        policy=policy,
        policy_kwargs=dict(
        activation_fn=torch.nn.ReLU,
        net_arch=[dict(pi=[256, 256], vf=[512, 512])],
        ),
        use_tanh_act=True,
        gae_lambda=0.95,
        gamma=0.99,
        n_steps=1000,
        n_seq=1,
        ent_coef=0.0,
        vf_coef=0.5,
        max_grad_norm=0.5,
        lstm_layer=1,
        batch_size=500,
        n_epochs=2000,
        clip_range=0.2,
        use_sde=False,  # don't use (gSDE), doesn't work
        retrain=args.retrain,
        verbose=1,
        only_lstm_training=True,
        features_dim=config.LatentSpaceCfg.vae_dims,
        states_dim=14,
        reconstruction_members=recon,
        reconstruction_steps=10,
        train_lstm_without_env=True,
        lstm_dataset_path=args.dir,
        lstm_weight_saved_path=args.lstm_exp,
        control_policy=pre_control_policy,
    )
    if args.train:
        model.train_lstm_from_dataset(use_log_depth=False)
    else:
        model.test_lstm_seperate()

if __name__ == "__main__":
    main()
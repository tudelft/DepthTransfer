import os
from os.path import join, exists
import numpy as np
from gymnasium import spaces
from isaacgym import gymutil
from aerial_gym.envs.base.zoo_task_config import ZooTaskCfg
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
        {"name": "--trial", "type": int, "default": 1, "help": "PPO trial number"},
        {"name": "--iter", "type": int, "default": 100, "help": "PPO iter number"},
        {"name": "--retrain", "type": bool, "default": False, "help": "if retrain"},
        {"name": "--dataset", "type": str, "default": "../saved/dataset_outdoor_gt", "help": "Where to place rollouts"},
        {"name": "--lstm_exp", "type": str, "default": "LSTM", "help": "Directory where results are logged"},
    ]
    recon = [1, 1, 0] # past, present, future
    args = get_args(add_args)
    if args.task == "mavrl_zoo":
        env_cfg = ZooTaskCfg()
    elif args.task == "mavrl_task":
        env_cfg = MAVRLTaskCfg()

    rsg_root = os.path.dirname(os.path.abspath(__file__))
    log_dir = rsg_root + "/../saved"
    device = get_device("auto")

    vae_logdir = "../exp_vae_320/"
    vae_file = join(vae_logdir, 'vae_resnet_outdoor', 'checkpoint.tar')
    assert exists(vae_file), "No trained VAE in the logdir..."
    state_vae = torch.load(vae_file, map_location=device)
    print("Loading VAE at epoch {} "
        "with test error {}".format(state_vae['epoch'], state_vae['precision']))
    
    policy = "MultiInputLstmPolicy"

    if env_cfg.LatentSpaceCfg.use_resnet_vae:
        vae_config_path = '../mav_baselines/torch/controlNet/models/encoder.yaml'
        features_extractor_params = {"ddconfig": OmegaConf.load(vae_config_path)['ddconfig']}

    pre_control_policy = {}
    for key in state_vae["state_dict"].keys():
        policy_key = 'features_extractor.' + key
        pre_control_policy[policy_key] = state_vae["state_dict"][key]
        
    observation_space = spaces.Dict({
            'image': spaces.Box(
                low=0,
                high=255,
                shape=(1, 224, 320),
                dtype='uint8'
            ),
            'state': spaces.Box(
                np.ones([1, env_cfg.LatentSpaceCfg.state_dims]) * -np.inf,
                np.ones([1, env_cfg.LatentSpaceCfg.state_dims]) * np.inf,
                dtype=np.float64,
            ),  
        })
    action_space = spaces.Box(
            low = np.ones(4) * -1.,
            high = np.ones(4) * 1.,
            dtype=np.float64,
        )
    
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
        features_dim=env_cfg.LatentSpaceCfg.vae_dims,
        states_dim=14,
        reconstruction_members=recon,
        reconstruction_steps=10,
        train_lstm_without_env=True,
        lstm_dataset_path=args.dataset,
        lstm_weight_saved_path=args.lstm_exp,
        control_policy=pre_control_policy,
        observation_space=observation_space,
        action_space=action_space,
    )
    if args.train:
        model.train_lstm_from_dataset(use_log_depth=False)
    else:
        model.test_lstm_seperate()

if __name__ == "__main__":
    main()
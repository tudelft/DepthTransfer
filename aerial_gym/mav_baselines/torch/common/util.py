import numpy as np
import pandas as pd
from torchvision.utils import save_image
import torch as th
from aerial_gym.mav_baselines.torch.recurrent_ppo.recurrent.type_aliases import RNNStates
from stable_baselines3.common.utils import obs_as_tensor

vision_columns = [
    "episode_id",
    "done",
    "reward",
    "t",
    "px",
    "py",
    "pz",
    "qw",
    "qx",
    "qy",
    "qz",
    "vx",
    "vy",
    "vz",
    "goal_x",
    "goal_y",
    "goal_z",
    "act11",
    "act12",
    "act13",
    "act14", 
]
columns = [
    "episode_id",
    "done",
    "reward",
    "t",
    "px",
    "py",
    "pz",
    "qw",
    "qx",
    "qy",
    "qz",
    "vx",
    "vy",
    "vz",
    "goal_x",
    "goal_y",
    "goal_z",
    "wx",
    "wy",
    "wz",
    "ax",
    "ay",
    "az",
    "mot1",
    "mot2",
    "mot3",
    "mot4",
    "thrust1",
    "thrust2",
    "thrust3",
    "thrust4",
    "targetx",
    "targety",
    "targetz",
    "targetr",
    "act1",
    "act2",
    "act3",
    "act4",
]

LSIZE, RED_SIZE = 32, 256

def traj_rollout(env, policy, max_ep_length=1000):
    traj_df = pd.DataFrame(columns=vision_columns)
    max_ep_length = max_ep_length
    # features = np.zeros([max_ep_length, LSIZE], dtype=np.float64)
    # labels = np.zeros([max_ep_length, 1, 28, 28], dtype=np.float64)
    obs = env.reset(random=False)
    episode_id = np.zeros(shape=(env.num_envs, 1))
    ave_reward = 0
    success_trial = 0
    trial = 0
    lstm_states = None
    while trial < 25:
        act, lstm_states = policy.predict(obs, state=lstm_states, deterministic=True)
        act = np.array(act, dtype=np.float64)
        #
        obs, rew, done, info = env.step(act)

        if done[0]==True:
            trial += 1
            lstm_states = None
            if rew[0]>=0:
                success_trial += 1
        ave_reward += rew
        # labels[i, :] = np.expand_dims(env.getLabelImage(), 0)
        episode_id[done] += 1

        state = env.getQuadState()
        action = env.getQuadAct()
        # reshape vector
        done = done[:, np.newaxis]
        rew = rew[:, np.newaxis]

        # stack all the data
        data = np.hstack((episode_id, done, rew, state, action))
        data_frame = pd.DataFrame(data=data, columns=vision_columns)

        # append trajectory
        traj_df = pd.concat([traj_df, data_frame], axis=0, ignore_index=True)
    return traj_df, ave_reward / max_ep_length, success_trial / trial, trial

def lstm_rollout(env, policy, device, logdir, iteration):
    max_ep_length = 200
    obs = env.reset(random=False)
    labels = np.zeros([max_ep_length, 1, 28, 28], dtype=np.float64)
    episode_id = np.zeros(shape=(env.num_envs, 1))
    single_hidden_state_shape = policy.lstm_hidden_state_shape
    _last_episode_starts = np.ones((1,), dtype=bool)
    _last_lstm_states = (
                th.zeros(single_hidden_state_shape,  device=device),
                th.zeros(single_hidden_state_shape,  device=device),
                )
    time_stamp = 0
    saved_images = []

    recon_next_plot = None
    recon_previous_plot = None
    recon_current_plot = None

    for i in range(max_ep_length):
        act, _ = policy.predict(obs, deterministic=True)
        act = np.array(act, dtype=np.float64)
        obs, rew, done, info = env.step(act)
        obs_torch = obs_as_tensor(obs,  device=device)
        latent_obs = policy.to_latent(obs_torch)
        episode_starts = th.tensor(_last_episode_starts, dtype=th.float32, device=device)
        recon, n_seq, _last_lstm_states= policy.predict_lstm(latent_obs, _last_lstm_states, episode_starts, is_eva=True)
        _last_episode_starts = done
        time_stamp += 1
        plot = []
        # state = env.getQuadState()
        if done[0]:
            time_stamp = 0
        if time_stamp % (20-policy.reconstruction_steps) == 0:
            if recon[0] is not None:
                obs_previous = obs_torch['image'][0].clone().detach().float() / 255.0
            
        if time_stamp % 20 == 0:
            if recon[0] is not None:
                recon_previous_plot = recon[0][0]
            if recon[1] is not None:
                recon_current_plot = recon[1][0]
                obs_current = obs_torch['image'][0].clone().detach().float() / 255.0
            if recon[2] is not None:
                recon_next_plot = recon[2][0]

        if time_stamp % (20+policy.reconstruction_steps) == 0:
            obs_next = obs_torch['image'][0].clone().detach().float() / 255.0
            time_stamp = 0
            if recon[0] is not None:
                plot.append(obs_previous)
                plot.append(recon_previous_plot) 
            if recon[1] is not None:
                plot.append(obs_current)
                plot.append(recon_current_plot)
            if recon[2] is not None:
                plot.append(obs_next)
                plot.append(recon_next_plot)

            saved_images.append(th.stack(plot, dim=0))
            # print("timestamp20: ", state)
    save_image(th.cat(saved_images), logdir + "/iter_{0:05d}.png".format(iteration))

def plot3d_traj(ax3d, pos, vel):
    sc = ax3d.scatter(
        pos[:, 0],
        pos[:, 1],
        pos[:, 2],
        c=np.linalg.norm(vel, axis=1),
        cmap="jet",
        s=1,
        alpha=0.5,
    )
    ax3d.view_init(elev=40, azim=50)
    #
    # ax3d.set_xticks([])
    # ax3d.set_yticks([])
    # ax3d.set_zticks([])

    #
    # ax3d.get_proj = lambda: np.dot(
    # Axes3D.get_proj(ax3d), np.diag([1.0, 1.0, 1.0, 1.0]))
    # zmin, zmax = ax3d.get_zlim()
    # xmin, xmax = ax3d.get_xlim()
    # ymin, ymax = ax3d.get_ylim()
    # x_f = 1
    # y_f = (ymax - ymin) / (xmax - xmin)
    # z_f = (zmax - zmin) / (xmax - xmin)
    # ax3d.set_box_aspect((x_f, y_f * 2, z_f * 2))

def test_vision_policy(env, model):
    max_ep_length = env.max_episode_steps
    num_rollouts = 20
    for n_roll in range(num_rollouts):
        obs, done, ep_len = env.reset(), False, 0
        while not (done or (ep_len >= max_ep_length)):
            act, _ = model.predict(obs, deterministic=True)
            obs, rew, done, info = env.step(act)
            ep_len += 1

def test_policy(env, model, render=False):
    max_ep_length = env.max_episode_steps
    num_rollouts = 1
    frame_id = 0
    if render:
        env.connectUnity()
    for n_roll in range(num_rollouts):
        obs, done, ep_len = env.reset(), False, 0
        while not ((ep_len >= max_ep_length)):
            act, _ = model.predict(obs, deterministic=True)
            obs, rew, done, info = env.step(act)
            #
            # print(obs)
            env.render(ep_len)

            # ======Gray Image=========
            # gray_img = np.reshape(
            #     env.getImage()[0], (env.img_height, env.img_width))
            # cv2.imshow("gray_img", gray_img)
            # cv2.waitKey(100)

            # ======RGB Image=========
            # img =env.getImage(rgb=True) 
            # rgb_img = np.reshape(
            #    img[0], (env.img_height, env.img_width, 3))
            # cv2.imshow("rgb_img", rgb_img)
            # os.makedirs("./images", exist_ok=True)
            # cv2.imwrite("./images/img_{0:05d}.png".format(frame_id), rgb_img)
            # cv2.waitKey(100)

            # # # ======Depth Image=========
            # depth_img = np.reshape(env.getDepthImage()[
            #                        0], (env.img_height, env.img_width))
            # os.makedirs("./depth", exist_ok=True)
            # cv2.imwrite("./depth/img_{0:05d}.png".format(frame_id), depth_img.astype(np.uint16))
            # cv2.imshow("depth", depth_img)
            # cv2.waitKey(100)

            #
            ep_len += 1
            frame_id += 1

    #
    if render:
        env.disconnectUnity()
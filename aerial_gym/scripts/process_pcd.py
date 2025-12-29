import argparse
import gc
import multiprocessing as mp
import os
import time
import warnings
from typing import Dict, Any, List, Tuple
import numpy as np
import open3d as o3d
import torch

def save_data(
    global_env_id: int,
    exp_dir: str,
    # ep_dict: Dict,
    pcd_points: np.ndarray,
):
    t_start = time.time()

    # torch.save(ep_dict, os.path.join(exp_dir, f"log_{global_env_id}.pt"))
    o3d.io.write_point_cloud(
        os.path.join(exp_dir, f"pcd_{global_env_id}.ply"), pcd_from_np_array(pcd_points)
    )

    t_end = time.time()
    print(f"[save data] env {global_env_id}, process time {t_end - t_start} s")

def empty_log_items_dict() -> Dict[str, List[Any]]:
    return {
        "t": [],
        "main_depth": [],
        "action": [],
        "pos": [],
        "rot": [],
        "lin_vel": [],
        "ang_vel": [],
        "is_finished": [],
        "is_crashed": [],
        "is_out_of_bounds": [],
    }

def pcd_from_np_array(points: np.ndarray) -> o3d.geometry.PointCloud:
    return o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points))


def convert_o3d(mat: np.ndarray) -> np.ndarray:
    ret = mat.copy()
    ret[:3, 0] = -mat[:3, 1]
    ret[:3, 1] = -mat[:3, 2]
    ret[:3, 2] = mat[:3, 0]
    return ret


def pinhole_depth_to_pcd(
    depth: np.ndarray, intrinsic: o3d.camera.PinholeCameraIntrinsic, tf: np.ndarray
) -> o3d.geometry.PointCloud:
    pcd: o3d.geometry.PointCloud = o3d.geometry.PointCloud.create_from_depth_image(
        depth=o3d.geometry.Image(depth),
        intrinsic=intrinsic,
    )  # noqa
    return pcd.transform(tf)

def filter_points_within_bounds(
    pcd: o3d.geometry.PointCloud, bounds: dict
) -> o3d.geometry.PointCloud:
    """
    Filters points in a point cloud that are within specified bounds.

    Args:
        pcd: The input point cloud.
        bounds: A dictionary specifying the min and max bounds for x, y, z. 
                Example: {'x': (-10, 10), 'y': (-5, 5), 'z': (0, 20)}

    Returns:
        o3d.geometry.PointCloud: The filtered point cloud.
    """
    points = np.asarray(pcd.points)
    
    # Extract bounds
    x_min, x_max = bounds['x']
    y_min, y_max = bounds['y']
    z_min, z_max = bounds['z']
    
    # Apply filtering condition
    mask = (
        (points[:, 0] >= x_min) & (points[:, 0] <= x_max) &
        (points[:, 1] >= y_min) & (points[:, 1] <= y_max) &
        (points[:, 2] >= z_min) & (points[:, 2] <= z_max)
    )
    
    # Filter points
    filtered_points = points[mask]
    
    # Create a new point cloud with the filtered points
    filtered_pcd = o3d.geometry.PointCloud()
    filtered_pcd.points = o3d.utility.Vector3dVector(filtered_points)
    
    return filtered_pcd

def test_pcd_log(
    env_id: int,
    extra_depth: torch.Tensor,  # (num_steps, 6, cam_h, cam_w)
    pcd_proc_params: Dict[str, Any],
    pcd_points: np.ndarray,
    position_w: torch.Tensor,  # (num_steps, 3)
    quaternion_w: torch.Tensor,  # (num_steps, 4)
    lower_bound: List[float],
    upper_bound: List[float],
    exp_dir: str,
):
    # prepare for pcd integration
    pcd = pcd_from_np_array(pcd_points)
    cam_intrinsic = o3d.camera.PinholeCameraIntrinsic(
        width=pcd_proc_params["w"],
        height=pcd_proc_params["h"],
        fx=pcd_proc_params["fx"],
        fy=pcd_proc_params["fy"],
        cx=pcd_proc_params["cx"],
        cy=pcd_proc_params["cy"],
    )

    for i in range(extra_depth.shape[0]):
        t_start = time.time()

        q_w_first = quaternion_w[i, 0, :].roll(1, 0)

        # extra depth images front the batch
        depth_front = extra_depth[i, 0].numpy()
        depth_back = extra_depth[i, 1].numpy()
        depth_left = extra_depth[i, 2].numpy()
        depth_right = extra_depth[i, 3].numpy()
        # depth_up = extra_depth[i][0, 4].numpy()
        # depth_down = extra_depth[i][0, 5].numpy()
        # transform matrices of cameras
        q_front = q_w_first.numpy()

        mat_front: np.ndarray = np.eye(4)
        mat_front[:3, :3] = o3d.geometry.get_rotation_matrix_from_quaternion(
            q_front
        )
        mat_front[:3, 3] = position_w[i, 0, :].numpy() + np.array([0.15, 0, 0])
        # print(position_w[i][0, 0, :])
        mat_back = mat_front.copy()
        mat_back[:3, :2] *= -1
        mat_back[:3, 3] = position_w[i, 0, :].numpy() + np.array([-0.15, 0, 0])

        mat_left = mat_front.copy()
        mat_left[:3, 0] = mat_front[:3, 1]
        mat_left[:3, 1] = -mat_front[:3, 0]
        mat_left[:3, 3] = position_w[i, 0, :].numpy() + np.array([0, 0.15, 0])

        mat_right = mat_left.copy()
        mat_right[:3, :2] *= -1
        mat_right[:3, 3] = position_w[i, 0, :].numpy() + np.array([0, -0.15, 0])
        # create pcds
        pcd_front = pinhole_depth_to_pcd(
            depth_front, cam_intrinsic, convert_o3d(mat_front)
        )
        pcd_back = pinhole_depth_to_pcd(
            depth_back, cam_intrinsic, convert_o3d(mat_back)
        )
        pcd_left = pinhole_depth_to_pcd(
            depth_left, cam_intrinsic, convert_o3d(mat_left)
        )
        pcd_right = pinhole_depth_to_pcd(
            depth_right, cam_intrinsic, convert_o3d(mat_right)
        )
        pcd +=  pcd_front + pcd_back + pcd_left + pcd_right #+ pcd_up + pcd_down

    bounds = {
        'x': (lower_bound[0], upper_bound[0]),
        'y': (lower_bound[1], upper_bound[1]),
        'z': (lower_bound[2], upper_bound[2]),
    }
    pcd = filter_points_within_bounds(pcd, bounds)
    # down-sample the pcd before returning
    pcd = pcd.voxel_down_sample(pcd_proc_params["voxel_size"])
    o3d.io.write_point_cloud(
        os.path.join(exp_dir, f"pcd_{env_id}.ply"), pcd_from_np_array(np.asarray(pcd.points))
    )
    return 

def test_ep_log(
    env_id: int,
    sim_dt: float,
    ctrl_freq: float,
    ep_dict: Dict[str, Any],  # dict containing keys "ep_0", "ep_1", ...
    env_step: List[torch.Tensor],
    episode_id: List[torch.Tensor],
    main_depth: List[torch.Tensor],
    action: List[torch.Tensor],
    pos: List[torch.Tensor],
    rot: List[torch.Tensor],
    lin_vel: List[torch.Tensor],
    ang_vel: List[torch.Tensor],
    is_finished: List[torch.Tensor],
    is_crashed: List[torch.Tensor],
    is_out_of_bounds: List[torch.Tensor],
    env_trial_limit: int = 10,
    exp_dir: str = "exp_dir",
):
    n_steps = len(env_step)
    for i in range(1, n_steps):
        step_ep_id = int(episode_id[i])
        if step_ep_id >= env_trial_limit:
            break
        ep_name = f"ep_{step_ep_id}"
        # if env_id == 1 and (step_ep_id == 4 or step_ep_id == 3 or step_ep_id == 5 or step_ep_id == 2 or step_ep_id == 1):
        #     print(ep_name)
        #     print(is_finished[i])
        #     print(is_crashed[i])
        #     print(is_out_of_bounds[i])
        #     print(pos[i, 0])
        if not ep_name in ep_dict:
            ep_dict[ep_name] = empty_log_items_dict()
        step_ep_prog = float(env_step[i])
        
        ep_dict[ep_name]["main_depth"].append(main_depth[i])
        ep_dict[ep_name]["is_finished"].append(is_finished[i])
        ep_dict[ep_name]["is_crashed"].append(is_crashed[i])
        ep_dict[ep_name]["is_out_of_bounds"].append(is_out_of_bounds[i])
        for j in range(pos.shape[1]):
            step_t = step_ep_prog * sim_dt * ctrl_freq + j * sim_dt
            # if step_ep_id == 0:
            #     print(step_t)
            ep_dict[ep_name]["t"].append(step_t)
            ep_dict[ep_name]["action"].append(action[i, j, 0])
            ep_dict[ep_name]["pos"].append(pos[i, j, 0])
            ep_dict[ep_name]["rot"].append(rot[i, j, 0])
            ep_dict[ep_name]["lin_vel"].append(lin_vel[i, j, 0])
            ep_dict[ep_name]["ang_vel"].append(ang_vel[i, j, 0])
        # print(ep_name, len(ep_dict[ep_name]["t"]))
        # print(ep_name, len(ep_dict[ep_name]["main_depth"]))
    torch.save(ep_dict, os.path.join(exp_dir, f"log_{env_id}.pt"))
    return

def main():
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("--exp_dir", type=str, required=True)
    arg_parser.add_argument("--num_processes", type=int, default=1)
    arg_parser.add_argument("--voxel_size", type=float, default=0.05)
    arg_parser.add_argument("--pcd_update_itv", type=int, default=25)

    args = arg_parser.parse_args()
    exp_dir: str = args.exp_dir
    num_processes: int = args.num_processes
    voxel_size: float = args.voxel_size
    pcd_update_itv: int = args.pcd_update_itv

    cfg: Dict[str, Any] = torch.load(os.path.join(exp_dir, "config.pt"))
    num_envs_log: int = cfg["env"]["num_envs"]
    ctrl_freq: int = cfg["env"]["num_control_steps_per_env_step"]
    sim_dt: float = cfg["sim"]["dt"]
    cam_w: int = cfg["pcd_camera_params"]["width"]
    cam_h: int = cfg["pcd_camera_params"]["height"]
    cam_hfov: float = cfg["pcd_camera_params"]["horizontal_fov_deg"]
    lower_bound: List[float] = cfg["logging"]["lower_bound"]
    upper_bound: List[float] = cfg["logging"]["upper_bound"]
    trial_limit: int = cfg["logging"]["trial_nums"]


    env_episode_data: Dict[str, Dict[str, Dict[str, List[Any]]]] = {
        f"env_{i}": {} for i in range(num_envs_log)
    }
    env_pcd_points: Dict[str, np.ndarray] = {
        f"env_{i}": np.empty((0, 3)) for i in range(num_envs_log)
    }
    log_date = torch.load(os.path.join(exp_dir, "log_data.pth"))
    pcd_depth = torch.stack(log_date['pcd']['depth']).permute(1, 0, 2, 3, 4)
    pcd_pos = torch.stack(log_date['pcd']['pos']).permute(1, 0, 2, 3)
    pcd_quat = torch.stack(log_date['pcd']['quat']).permute(1, 0, 2, 3)

    # for key in log_date.keys():
    #     print(key, len(log_date[key]))
    # print(log_date['episode_id'])
    env_step = torch.stack(log_date['env_step']).permute(1, 0)
    episode_id = torch.stack(log_date['episode_id']).permute(1, 0)
    main_depth = torch.stack(log_date['main_depth']).permute(1, 0, 2, 3)
    action = torch.stack(log_date['action']).permute(2, 0, 1, 3, 4)
    pos = torch.stack(log_date['pos']).permute(2, 0, 1, 3, 4)
    rot = torch.stack(log_date['rot']).permute(2, 0, 1, 3, 4)
    lin_vel = torch.stack(log_date['linvel']).permute(2, 0, 1, 3, 4)
    ang_vel = torch.stack(log_date['angvel']).permute(2, 0, 1, 3, 4)
    is_finished = torch.stack(log_date['is_finished']).permute(1, 0)
    is_crashed = torch.stack(log_date['is_crashed']).permute(1, 0)
    is_out_of_bounds = torch.stack(log_date['is_out_of_bounds']).permute(1, 0)

    # clear original log data
    del log_date

    with mp.Pool(min(num_processes, num_envs_log)) as pool:
        pool.starmap(
            test_ep_log,
            [ 
                (
                    env_id,
                    sim_dt,
                    ctrl_freq,
                    env_episode_data[f"env_{env_id}"], 
                    env_step[env_id],
                    episode_id[env_id],
                    main_depth[env_id],
                    action[env_id],
                    pos[env_id],
                    rot[env_id],
                    lin_vel[env_id],
                    ang_vel[env_id],
                    is_finished[env_id],
                    is_crashed[env_id],
                    is_out_of_bounds[env_id],
                    trial_limit,
                    exp_dir,
                )
                for env_id in range(num_envs_log)
            ],
        )
    print("done processing ep logs")
    # clear memory of main depth
    del main_depth
    
    # pcd process params
    cam_fx = cam_w / (2 * np.tan(np.deg2rad(cam_hfov) / 2))
    cam_fy = cam_fx * cam_h / cam_w
    cam_cx = cam_w / 2
    cam_cy = cam_h / 2
    pcd_proc_params: Dict[str, Any] = {
        "w": cam_w,
        "h": cam_h,
        "fx": cam_fx,
        "fy": cam_fy,
        "cx": cam_cx,
        "cy": cam_cy,
        "voxel_size": voxel_size,
        "pcd_update_itv": pcd_update_itv,
    }
    with mp.Pool(min(num_processes, num_envs_log)) as pool:
        pool.starmap(
            test_pcd_log,
            [
                (
                    env_id, 
                    pcd_depth[env_id],
                    pcd_proc_params,
                    env_pcd_points[f"env_{env_id}"],
                    pcd_pos[env_id],
                    pcd_quat[env_id],
                    lower_bound,
                    upper_bound,
                    exp_dir,
                )
                for env_id in range(num_envs_log)
            ],
        )

if __name__ == "__main__":
    main()

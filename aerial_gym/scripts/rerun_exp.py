import argparse
import multiprocessing as mp
import os
import time
import warnings
from datetime import datetime
from typing import Dict, Any, List

import matplotlib
import numpy as np
import open3d as o3d
import rerun as rr
import rerun.blueprint as rrb
import torch
from rerun_loader_urdf import URDFLogger
from scipy.spatial.transform import Rotation as R

def rrb_single_env(num_episodes: int) -> rrb.Blueprint:
    # blueprint = rrb.Blueprint(rrb.Spatial3DView())
    blueprint = rrb.Horizontal(
        rrb.Spatial3DView(name="3D"),
        rrb.Vertical(
            rrb.Horizontal(
                rrb.Vertical(
                    # Note that we re-project the annotations into the 2D views:
                    # For this to work, the origin of the 2D views has to be a pinhole camera,
                    # this way the viewer knows how to project the 3D annotations into the 2D views.
                    rrb.Tabs(
                        # add a spatial 2D view for each episode
                        contents=
                        [
                            rrb.Spatial2DView(
                                name=f"episode_{i}_camera",
                                origin=f"/world/episode_{i}/camera/depth/",
                                contents=[
                                    "$origin/image",
                                    f"/world/episode_{i}/camera/**"
                                ],
                            )
                            for i in range(num_episodes)
                        ],
                    ),
                    rrb.Tabs(
                        # add a spatial 2D view for each episode
                        contents=
                        [
                            rrb.TimeSeriesView(
                                origin=f"/world/episode_{i}/camera/recon/",
                                contents=[
                                    "$origin/image",
                                    f"/world/episode_{i}/camera/**"
                                ],
                            )
                            for i in range(num_episodes)
                        ],
                    ),
                    row_shares=[1, 1],
                ),
                rrb.Tabs(
                # add a spatial 2D view for each episode
                    contents=
                    [
                        rrb.TimeSeriesView(
                            origin=f"/world/episode_{i}/scalar/lin_vel/",
                            # Set a custom Y axis.
                            axis_y=rrb.ScalarAxis(range=(-5.0, 5.0), zoom_lock=True),
                            # Configure the legend.
                            plot_legend=rrb.PlotLegend(visible=False),
                            # Set time different time ranges for different timelines.
                            time_ranges=[
                                # Sliding window depending on the time cursor for the first timeline.
                                rrb.VisibleTimeRange(
                                    "timeline0",
                                    start=rrb.TimeRangeBoundary.cursor_relative(seq=-1),
                                    end=rrb.TimeRangeBoundary.cursor_relative(),
                                ),
                                # Time range from some point to the end of the timeline for the second timeline.
                                rrb.VisibleTimeRange(
                                    "timeline1",
                                    start=rrb.TimeRangeBoundary.absolute(seconds=2.0),
                                    end=rrb.TimeRangeBoundary.infinite(),
                                ),
                            ],
                        )
                        for i in range(num_episodes)
                    ],
                ),
                column_shares=[1, 2],
            ),
            rrb.Vertical(
                rrb.Tabs(
                    # add a spatial 2D view for each episode
                    contents=
                    [
                        rrb.TimeSeriesView(
                            origin=f"/world/episode_{i}/scalar/action/",
                            # Set a custom Y axis.
                            axis_y=rrb.ScalarAxis(range=(-1.0, 1.0), zoom_lock=True),
                            # Configure the legend.
                            plot_legend=rrb.PlotLegend(visible=False),
                            # Set time different time ranges for different timelines.
                            time_ranges=[
                                # Sliding window depending on the time cursor for the first timeline.
                                rrb.VisibleTimeRange(
                                    "timeline0",
                                    start=rrb.TimeRangeBoundary.cursor_relative(seq=-1),
                                    end=rrb.TimeRangeBoundary.cursor_relative(),
                                ),
                                # Time range from some point to the end of the timeline for the second timeline.
                                rrb.VisibleTimeRange(
                                    "timeline1",
                                    start=rrb.TimeRangeBoundary.absolute(seconds=2.0),
                                    end=rrb.TimeRangeBoundary.infinite(),
                                ),
                            ],
                        )
                        for i in range(num_episodes)
                    ],
                ),

            ),
        ),
        column_shares=[4, 3],
    )
    return blueprint


def log_world_frame_and_pcd(pcd: o3d.geometry.PointCloud, pcd_colormap: str):
    # log world frame
    rr.log(
        f"world_frame",
        rr.Transform3D(axis_length=1.0),
        static=True,
    )

    # load and log pcd
    pcd_points = np.asarray(pcd.points)
    pcd_points_z = pcd_points[:, -1]
    pcd_points_min_z = pcd_points_z.min()
    pcd_points_max_z = pcd_points_z.max()
    pcd_z_norm = matplotlib.colors.Normalize(
        vmin=pcd_points_min_z, vmax=pcd_points_max_z
    )

    pcd_cmap = matplotlib.colormaps[pcd_colormap]
    rr.log(
        f"pcd",
        rr.Points3D(pcd_points, colors=pcd_cmap(pcd_z_norm(pcd_points_z))),
        static=True,
    )

def log_episode_data(
    ctrl_freq: int,
    ep_dict: Dict[str, Any],
    cam_params: Dict[str, Any],
    vel_colormap: str,
    vel_max_cmap: float,
    traj_line_weight: float,
    drone_urdf: str,
    num_episodes: int,
    log_cam: bool,
    ep_prefix: str,
    only_traj: bool = False,
):
    num_substeps = ctrl_freq
    sim_dt = ep_dict["ep_1"]["t"][1] - ep_dict["ep_1"]["t"][0]
    for key in ep_dict.keys():
        print(f"key: {key}")
    step_dt = sim_dt * num_substeps
    vel_cmap = matplotlib.colormaps[vel_colormap]
    # load urdf
    urdf_logger = None
    if drone_urdf is not None:
        urdf_logger = URDFLogger(drone_urdf, None)

    for i in range(num_episodes):
        num_steps = len(ep_dict[f"ep_{i}"]["t"])
        # log trajectory
        pos_tensor = torch.stack(ep_dict[f"ep_{i}"]["pos"])
        line_start = pos_tensor[10:-1]  # (N-1, 3)
        line_end = pos_tensor[11:]  # (N-1, 3)
        line_data = torch.stack((line_start, line_end), dim=1).numpy()  # (N-1, 2, 3)
        ep_vel_norm = (
            torch.stack(ep_dict[f"ep_{i}"]["lin_vel"]).norm(dim=1)
        )
        vel_line_avg = ((ep_vel_norm[10:-1] + ep_vel_norm[11:]) / 2).numpy()
        rr.log(
            f"/world/episode_{ep_prefix}{i}/trajectory",
            rr.LineStrips3D(
                line_data,
                colors=vel_cmap(vel_line_avg / vel_max_cmap),
                radii=rr.Radius.ui_points(traj_line_weight),
            ),
            static=True,
        )
        for j in range(num_steps):
            # log time
            substep_t = ep_dict[f"ep_{i}"]["t"][j]
            substep_pos = ep_dict[f"ep_{i}"]["pos"][j]
            substep_rot = ep_dict[f"ep_{i}"]["rot"][j]
            substep_lin_vel = ep_dict[f"ep_{i}"]["lin_vel"][j]
            substep_action = ep_dict[f"ep_{i}"]["action"][j]
            substep_lin_vel_norm = substep_lin_vel.norm()
            vel_mapped_color = vel_cmap(substep_lin_vel_norm / vel_max_cmap)
            rr.set_time_seconds("sim_time", substep_t)
            # log position
            rr.log(
                f"/world/episode_{ep_prefix}{i}/position",
                rr.Points3D(
                    substep_pos,
                    colors=vel_mapped_color,
                ),
            )

            rr.log(
                f"/world/episode_{ep_prefix}{i}/scalar/action/x",
                rr.Scalar(substep_action[0]),
                rr.SeriesLine(),
            )
            rr.log(
                f"/world/episode_{ep_prefix}{i}/scalar/action/y",
                rr.Scalar(substep_action[1]),
                rr.SeriesLine(),
            )
            rr.log(
                f"/world/episode_{ep_prefix}{i}/scalar/action/z",
                rr.Scalar(substep_action[2]),
                rr.SeriesLine(),
            )

            rr.log(
                f"/world/episode_{ep_prefix}{i}/scalar/lin_vel/x",
                rr.Scalar(substep_lin_vel[0]),
                rr.SeriesLine(),
            )
            rr.log(
                f"/world/episode_{ep_prefix}{i}/scalar/lin_vel/y",
                rr.Scalar(substep_lin_vel[1]),
                rr.SeriesLine(),
            )
            rr.log(
                f"/world/episode_{ep_prefix}{i}/scalar/lin_vel/z",
                rr.Scalar(substep_lin_vel[2]),
                rr.SeriesLine(),
            )


            # log camera
            if log_cam and j % num_substeps == 0:
                step_depth = ep_dict[f"ep_{i}"]["main_depth"][j//num_substeps]
                # convert substep_rot from euler to rotation matrix
                substep_rot_matrix = torch.tensor(R.from_euler('xyz', substep_rot.numpy()).as_matrix())
                # get camera pose
                cam_pos = substep_pos
                tf_world_to_cam = torch.eye(4)
                tf_world_to_cam[:3, :3] = substep_rot_matrix
                tf_world_to_cam[:3, 3] = cam_pos + torch.tensor([0.15, 0.0, 0.0])
                rr.log(
                    f"/world/episode_{ep_prefix}{i}/camera",
                    rr.Pinhole(
                        focal_length=float(cam_params["f"]),
                        width=int(cam_params["w"]),
                        height=int(cam_params["h"]),
                        camera_xyz=rr.ViewCoordinates.FLU,
                        image_plane_distance=1.0,
                    ),
                )
                rr.log(
                    f"/world/episode_{ep_prefix}{i}/camera/depth",
                    rr.DepthImage(
                        step_depth,
                        meter=1,
                        colormap=rr.components.Colormap(1),  # gray scale
                    ),
                )
                rr.log(
                    f"/world/episode_{ep_prefix}{i}/camera",
                    rr.Transform3D(
                        translation=tf_world_to_cam[:3, 3],
                        mat3x3=tf_world_to_cam[:3, :3],
                        axis_length=0.0,
                    ),
                )


@rr.shutdown_at_exit
def proc_env(
    ctrl_freq: int,
    cam_params: Dict[str, Any],
    pcd_colormap: str,
    vel_colormap: str,
    vel_max_cmap: float,
    traj_line_weight: float,
    drone_urdf: str,
    exp_dir: str,
    env_id: int,
    num_episodes: int,
):
    t_start = time.time()
    # load episode data from file
    ep_dict: Dict = torch.load(os.path.join(exp_dir, f"log_{env_id}.pt"))
    exp_name = os.path.basename(os.path.normpath(exp_dir))
    rrd_file = os.path.join(exp_dir, f"env_{env_id}.rrd")
    rr.init(application_id=exp_name, recording_id=f"env_{env_id}")
    rr.save(path=rrd_file, default_blueprint=rrb_single_env(num_episodes))
    # rr.connect()
    # load pcd
    pcd = o3d.io.read_point_cloud(os.path.join(exp_dir, f"pcd_{env_id}.ply"))
    # log world frame and pcd
    log_world_frame_and_pcd(pcd, pcd_colormap)
    # log episode data
    log_episode_data(
        ctrl_freq,
        ep_dict,
        cam_params,
        vel_colormap,
        vel_max_cmap,
        traj_line_weight,
        drone_urdf,
        num_episodes,
        True,
        "",
    )
    # end stopwatch
    t_end = time.time()
    print(
        f"[process env] env {env_id}, process time {t_end - t_start}, created {rrd_file}"
    )
    return

def main():
    # info
    print("+++ Rerunning experiment")

    # args
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("--exp_dir", type=str, required=True)
    arg_parser.add_argument("--num_processes", type=int, default=4)
    arg_parser.add_argument("--pcd_colormap", type=str, default="turbo")
    arg_parser.add_argument("--vel_colormap", type=str, default="plasma")
    arg_parser.add_argument("--vel_max_cmap", type=float, default=6.0)
    arg_parser.add_argument("--traj_line_weight", type=float, default=1.5)
    arg_parser.add_argument("--drone_urdf", type=str)
    args = arg_parser.parse_args()
    exp_dir: str = args.exp_dir
    num_processes: int = args.num_processes
    pcd_colormap: str = args.pcd_colormap
    vel_colormap: str = args.vel_colormap
    vel_max_cmap: float = args.vel_max_cmap
    traj_line_weight: float = args.traj_line_weight
    drone_urdf: str = args.drone_urdf
    # env_id = 2
    # rr.init(application_id='test_pcd', recording_id=f"env_{env_id}")
    # rr.connect()
    # pcd = o3d.io.read_point_cloud(os.path.join(exp_dir, f"pcd_{env_id}.ply"))
    # log_world_frame_and_pcd(pcd, pcd_colormap)
    # get info from cfg
    cam_params: Dict[str, Any] = {}
    cfg: Dict[str, Any] = torch.load(os.path.join(exp_dir, "config.pt"))
    num_envs_log: int = cfg["env"]["num_envs"]
    ctrl_freq: int = cfg["env"]["num_control_steps_per_env_step"]
    num_episodes: int = cfg["logging"]["trial_nums"]
    if len(cam_params) == 0:
        w = cfg["camera_params"]["width"]
        h = cfg["camera_params"]["height"]
        hfov = cfg["camera_params"]["horizontal_fov_deg"]
        max_depth = cfg["camera_params"]["max_range"]
        f = w / (2 * np.tan(np.deg2rad(hfov) / 2))
        cam_params["w"] = w
        cam_params["h"] = h
        cam_params["f"] = f
        cam_params["depth_scale"] = max_depth

    with mp.Pool(min(num_processes, num_envs_log)) as pool:
        pool.starmap(
            proc_env,
            [
                (
                    ctrl_freq,
                    cam_params,
                    pcd_colormap,
                    vel_colormap,
                    vel_max_cmap,
                    traj_line_weight,
                    drone_urdf,
                    exp_dir,
                    env_id,
                    num_episodes,
                )
                for env_id in range(num_envs_log)
            ],
        )

if __name__ == "__main__":
    main()


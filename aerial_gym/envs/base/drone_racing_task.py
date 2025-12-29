# general python imports
import time
import logging
import trimesh as tm
from urdfpy import URDF
import warp as wp
import math
import numpy as np
import random
import os
from typing import Optional
import torch
import sys
import gym
from gymnasium import spaces

# isaacgym imports
from isaacgym import gymutil, gymtorch, gymapi
from isaacgym.torch_utils import *
from aerial_gym.utils.torch_utils_contrib import *

from aerial_gym import AERIAL_GYM_ROOT_DIR
from aerial_gym.envs.base.base_task import BaseTask
from .mavrl_task_config import MAVRLTaskCfg
# from .zoo_task_config import ZooTaskCfg
from aerial_gym.envs.base.drone_racing_task_config import DroneRacingTaskCfg
from aerial_gym.envs.controllers.controller import Controller
from aerial_gym.envs.reference.reference_base import ReferenceBase

from aerial_gym.utils.asset_manager import AssetManager
# from aerial_gym.utils.mavrl_asset_manager import MAVRLAssetManager
from aerial_gym.utils.sampler import Sampler
from aerial_gym.utils.helpers import asset_class_to_AssetOptions, class_to_dict
from aerial_gym.utils.episode_logger import EpisodeLogger
from aerial_gym.utils.warp_sensor import WarpSensor
from aerial_gym.utils.sgm_depth import SGM

from aerial_gym.dr.waypoint import (
    WaypointTrackerParams,
    WaypointTracker,
    WaypointData,
    WaypointGeneratorParams,
    WaypointGenerator,
    RandWaypointOptions,
)

from aerial_gym.dr.managers.obstacle_manager import (
    ObstacleManager,
    RandObstacleOptions,
    )

from aerial_gym.dr.managers.drone_manager import (
    DroneManager,
    DroneManagerParams,
    RandDroneOptions,
)

from aerial_gym.dr.env import (
    EnvCreatorParams,
    EnvCreator,
)

logger = logging.getLogger(__name__)

class DroneRacingTask(BaseTask):

    def __init__(self, cfg: DroneRacingTaskCfg, sim_params, physics_engine, sim_device, headless):
        self.cfg = cfg
        self.sim_params = sim_params
        self.physics_engine = physics_engine
        self.sim_device_id = sim_device
        self.headless = headless
        self.enable_debug_viz = self.cfg.env.enableDebugVis
        self.disable_obstacle_man = self.cfg.env.disableObstacleManager

        if self.cfg.env.debug:
            logger.setLevel(logging.DEBUG)
        
        self.enable_isaacgym_cameras = self.cfg.env.enable_isaacgym_cameras
        self.use_stereo_vision = self.cfg.camera_params.use_stereo_vision

        if self.enable_isaacgym_cameras:
            self.cfg.env.num_observations += self.cfg.LatentSpaceCfg.state_dims

        self.sensor_config = self.cfg.sensor_config

        if self.sensor_config.sensor_type == "lidar":
            self.sensor_params = self.cfg.lidar_params
        elif self.sensor_config.sensor_type == "camera":
            self.sensor_params = self.cfg.camera_params
        else:
            # Raise an error with message
            raise ValueError("Sensor type not supported. Use either lidar or camera")
        
        # save control data for debugging
        self.save_control_data = self.cfg.control.save_control_data
        self.control_data_path = self.cfg.control.control_data_path
        if self.save_control_data:
            if not os.path.exists(self.control_data_path):
                os.makedirs(self.control_data_path)
            if not os.path.exists(self.control_data_path + "/states_data"):
                os.makedirs(self.control_data_path + "/states_data")
            if not os.path.exists(self.control_data_path + "/actions_data"):
                os.makedirs(self.control_data_path + "/actions_data")
            if not os.path.exists(self.control_data_path + "/control_data"):
                os.makedirs(self.control_data_path + "/control_data")
            if not os.path.exists(self.control_data_path + "/reference_data"):
                os.makedirs(self.control_data_path + "/reference_data")

        self.gravity = torch.tensor(self.cfg.sim.gravity, device=self.sim_device_id, requires_grad=False)
        super().__init__(self.cfg, sim_params, physics_engine, sim_device, headless)

        # initialize observation and action spaces
        self.obs_space = spaces.Dict({
            'image': spaces.Box(
                low=0,
                high=255,
                shape=(1, 256, 256),
                dtype='uint8'
            ),
            'state': spaces.Box(
                np.ones([1, self.cfg.LatentSpaceCfg.state_dims]) * -np.inf,
                np.ones([1, self.cfg.LatentSpaceCfg.state_dims]) * np.inf,
                dtype=np.float64,
            ),  
        })
        self.act_space = spaces.Box(
            low = np.ones(self.num_actions) * -1.,
            high = np.ones(self.num_actions) * 1.,
            dtype=np.float64,
        )
        self.obs_dict = {
            'image': torch.zeros((self.num_envs, 1, self.cfg.LatentSpaceCfg.imput_image_size, self.cfg.LatentSpaceCfg.imput_image_size), dtype=torch.uint8, device=self.device),
            'state': torch.zeros((self.num_envs, 1, self.num_obs), dtype=torch.float32, device=self.device)
        }
        self.state_observations = torch.zeros((self.num_envs, 1, self.num_obs), dtype=torch.float32, device=self.device)

        self.root_tensor = self.gym.acquire_actor_root_state_tensor(self.sim)
        self.contact_force_tensor = self.gym.acquire_net_contact_force_tensor(self.sim)
        self.force_sensor_tensor = self.gym.acquire_force_sensor_tensor(self.sim)

        if self.enable_isaacgym_cameras:
            self.full_camera_array = torch.zeros((self.num_envs, self.sensor_params.height, self.sensor_params.width), device=self.device, requires_grad=False)
    
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_net_contact_force_tensor(self.sim)
        self.gym.refresh_force_sensor_tensor(self.sim)

        #TODO drones and waypoints generation and bounaries
        # self.num_waypoints_to_track = self.cfg.way_point_generator.num_waypoints - 1
        # create waypoint tracker
        self.waypoint_tracker = WaypointTracker(
            WaypointTrackerParams(
                num_envs=self.num_envs,
                device=self.device,
                num_waypoints=self.num_waypoints_to_track,
            )
        )
        # generate random multiple tracks with or without obstacles
        self.waypoint_generator = WaypointGenerator(
            self._param_from_cfg(WaypointGeneratorParams, class_to_dict(self.cfg.way_point_generator))
        )
        self.rand_waypoint_opts = self._param_from_cfg(
            RandWaypointOptions, class_to_dict(self.cfg.initRandOpt.randWaypointOptions)
        )
        self.next_waypoint_id: torch.Tensor = torch.ones(
            self.num_envs, dtype=torch.long, device=self.device
        )  # also need it for the first reset_idx

        self.obstacle_manager = ObstacleManager(self.env_creator)
        self.rand_obstacle_opts = self._param_from_cfg(
            RandObstacleOptions, class_to_dict(self.cfg.initRandOpt.randObstacleOptions)
        )
        self.drone_manager = DroneManager(
            DroneManagerParams(num_envs=self.num_envs, device=self.device)
        )
        self.rand_drone_opts = self._param_from_cfg(
            RandDroneOptions, class_to_dict(self.cfg.initRandOpt.randDroneOptions)
        )

        self.num_actors_per_env = self.env_creator.num_actors_per_env
        self.bodies_per_map = self.env_creator.num_actors_per_env + self.num_robots_per_map * (self.robot_num_bodies-1) # Number of links in the environment + robot

        # print(f"Number of actors per environment: {self.num_actors_per_env}")
        # print(f"Number of bodies per environment: {self.bodies_per_map}")
        # print(f"Number of robots per environment: {self.num_robots_per_map}")
        # num_actors = self.env_asset_manager.get_env_actor_count() + self.num_robots_per_map # Number of obstacles in the environment + one robot
        # bodies_per_map = self.env_asset_manager.get_env_link_count() + self.num_robots_per_map * self.robot_num_bodies # Number of links in the environment + robot
        self.unfolded_vec_root_tensor = gymtorch.wrap_tensor(self.root_tensor)
        self.vec_root_tensor = self.unfolded_vec_root_tensor.view(self.num_maps, self.num_actors_per_env, 13)
        self.root_states = self.vec_root_tensor[:, :self.num_robots_per_map, :]
        self.root_positions = self.root_states[..., 0:3]
        self.root_quats = self.root_states[..., 3:7]
        self.root_linvels = self.root_states[..., 7:10]
        self.root_angvels = self.root_states[..., 10:13]
        self.env_asset_root_states = self.vec_root_tensor[:, self.num_robots_per_map:, :]

        # action ranges
        max_acc = torch.FloatTensor(self.cfg.env.act_max).to(self.device).requires_grad_(False)
        min_acc = torch.FloatTensor(self.cfg.env.act_min).to(self.device).requires_grad_(False)
        self.mean_acc = (max_acc + min_acc) / 2.0
        self.std_acc = (max_acc - min_acc) / 2.0

        # if any of the shape is 0 set opbject to none
        if any([self.env_asset_root_states.shape[0] == 0, self.env_asset_root_states.shape[1] == 0, self.env_asset_root_states.shape[2] == 0]):
            self.env_asset_root_states = None
        

        #TODO initialize the state of obstacles and gates
        # self.init_obstacles()

        # kinds of noise
        force_noise = self.cfg.env_force_perturbations_noise
        self.env_force_sampler = Sampler(force_noise.enable, force_noise.distribution, force_noise.dist_params, transform_after_sampling=force_noise.transform_after_sampling, size=torch.Size([self.num_maps, self.num_robots_per_map, 3]), device=self.device)
        
        torque_noise = self.cfg.env_torque_perturbations_noise
        self.env_torque_sampler = Sampler(torque_noise.enable, torque_noise.distribution, torque_noise.dist_params, transform_after_sampling=torque_noise.transform_after_sampling, size=torch.Size([self.num_maps, self.num_robots_per_map, 3]), device=self.device)

        disturbance_application_prob = self.cfg.env_disturbance_application_probability
        self.env_disturbance_application_sampler = Sampler(disturbance_application_prob.enable, disturbance_application_prob.distribution, disturbance_application_prob.dist_params, transform_after_sampling=disturbance_application_prob.transform_after_sampling, size=torch.Size([self.num_maps, self.num_robots_per_map]), device=self.device)

        velocity_noise = self.cfg.velocity_measurement_noise
        self.velocity_noise_sampler = Sampler(velocity_noise.enable, velocity_noise.distribution, velocity_noise.dist_params, transform_after_sampling=velocity_noise.transform_after_sampling, size=torch.Size([self.num_maps, self.num_robots_per_map, 3]), device=self.device)

        euler_angle_noise = self.cfg.angle_measurement_noise
        self.euler_angle_noise_sampler = Sampler(euler_angle_noise.enable, euler_angle_noise.distribution, euler_angle_noise.dist_params, transform_after_sampling=euler_angle_noise.transform_after_sampling, size=torch.Size([self.num_maps, self.num_robots_per_map, 3]), device=self.device)

        angular_velocity_noise = self.cfg.angular_velocity_measurement_noise
        self.angular_velocity_noise_sampler = Sampler(angular_velocity_noise.enable, angular_velocity_noise.distribution, angular_velocity_noise.dist_params, transform_after_sampling=angular_velocity_noise.transform_after_sampling, size=torch.Size([self.num_maps, self.num_robots_per_map, 3]), device=self.device)


        self.privileged_obs_buf = None
        if self.vec_root_tensor.shape[1] > 1:
            if self.get_privileged_obs:
                self.privileged_obs_buf = torch.zeros((self.num_envs, 4), device=self.device, requires_grad=False)


        self.contact_forces = gymtorch.wrap_tensor(self.contact_force_tensor).view(self.num_maps, self.bodies_per_map, 3)[:, :self.robot_num_bodies*self.num_robots_per_map]

        self.collisions = torch.zeros(self.num_maps, self.num_robots_per_map * self.robot_num_bodies, device=self.device)
        self.ones = torch.ones(self.num_maps, self.num_robots_per_map, device=self.device, requires_grad=False)
        self.zeros = torch.zeros(self.num_maps, self.num_robots_per_map, device=self.device, requires_grad=False)
        self.zeros_3d = torch.zeros((self.num_maps, self.num_robots_per_map, 3), device=self.device, requires_grad=False)
        self.ones_3d = torch.ones((self.num_maps, self.num_robots_per_map, 3), device=self.device, requires_grad=False)
        self.zeros_int = torch.zeros((self.num_maps, self.num_robots_per_map), dtype=torch.int64, device=self.device, requires_grad=False)
        self.ones_int = torch.ones((self.num_maps, self.num_robots_per_map), dtype=torch.int64, device=self.device, requires_grad=False)
        self.ones_bodied = torch.ones(self.num_maps, self.num_robots_per_map * self.robot_num_bodies, device=self.device, requires_grad=False)
        self.rew_buf = torch.zeros((self.num_envs, len(self.cfg.RLParamsCfg.names)), device=self.device, requires_grad=False)
        self.terminal_rewards = torch.zeros(self.num_maps, self.num_robots_per_map, device=self.device)

        self.vehicle_frame_euler_angles = torch.zeros((self.num_maps, self.num_robots_per_map, 3), device=self.device, requires_grad=False)
        self.root_euler_angles = torch.zeros((self.num_maps, self.num_robots_per_map, 3), device=self.device, requires_grad=False)
        self.vehicle_frame_quats = torch.zeros((self.num_maps, self.num_robots_per_map, 4), device=self.device, requires_grad=False)
        self.angvels_body_frame = torch.zeros((self.num_maps, self.num_robots_per_map, 3), device=self.device, requires_grad=False)
        self.root_linvels_obs = torch.zeros((self.num_maps, self.num_robots_per_map, 3), device=self.device, requires_grad=False)
        self.linvels_body_frame = torch.zeros((self.num_maps, self.num_robots_per_map, 3), device=self.device, requires_grad=False)
        self.linvels_vehicle_frame = torch.zeros((self.num_maps, self.num_robots_per_map, 3), device=self.device, requires_grad=False)

        self.counter = 0

        self.action_input = torch.zeros(
            (self.num_envs, self.cfg.env.num_actions), dtype=torch.float32, device=self.device, requires_grad=False)
        self.prev_action_input = torch.zeros(
            (self.num_maps, self.num_robots_per_map, self.cfg.env.num_actions), dtype=torch.float32, device=self.device, requires_grad=False)
        self.forces = torch.zeros((self.num_maps, self.bodies_per_map, 3),
                                  dtype=torch.float32, device=self.device, requires_grad=False)
        self.torques = torch.zeros((self.num_maps, self.bodies_per_map, 3),
                                   dtype=torch.float32, device=self.device, requires_grad=False)

        self.successes = torch.zeros_like(self.ones)
        self.out_of_bounds = torch.zeros_like(self.ones)

        self.total_trials = torch.zeros(self.num_envs, device=self.device, requires_grad=False)
        self.success_trials = torch.zeros(self.num_envs, device=self.device, requires_grad=False)

        # use acceleration control and reference
        self.controller = Controller(self.robot_inertia, self.cfg.control, self.num_envs, self.device)
        self.reference = ReferenceBase(self.cfg.FeasibilityCfg, self.num_envs, self.cfg.env.num_control_steps_per_env_step, self.device)

        # Getting environment bounds
        self.env_lower_bound = torch.zeros(
            (self.num_envs, 3), dtype=torch.float32, device=self.device, requires_grad=False)
        self.env_upper_bound = torch.zeros(
            (self.num_envs, 3), dtype=torch.float32, device=self.device, requires_grad=False)
        
        # self.goal_positions = torch.zeros((self.num_maps, self.num_robots_per_map, self.cfg.env.goal_num_per_episode + 1, 3), dtype=torch.float32, device=self.device, requires_grad=False)
        self.curr_goal_positions = torch.zeros((self.num_maps, self.num_robots_per_map, 3), dtype=torch.float32, device=self.device, requires_grad=False)
        # self.curr_goal_index = torch.zeros(self.num_maps, self.num_robots_per_map, dtype=torch.int64, device=self.device, requires_grad=False)
        self.flight_lower_bound = torch.FloatTensor([-self.cfg.envCreator.env_size/2, -self.cfg.envCreator.env_size/2, 0.0]
                                                    ).to(self.device).expand(self.num_maps, self.num_robots_per_map, -1)
        self.flight_upper_bound = torch.FloatTensor([self.cfg.envCreator.env_size/2, self.cfg.envCreator.env_size/2, self.cfg.envCreator.env_size]
                                                    ).to(self.device).expand(self.num_maps, self.num_robots_per_map, -1)
        
        if self.viewer:
            cam_pos_x, cam_pos_y, cam_pos_z = self.cfg.viewer.pos[0], self.cfg.viewer.pos[1], self.cfg.viewer.pos[2]
            cam_target_x, cam_target_y, cam_target_z = self.cfg.viewer.lookat[0], self.cfg.viewer.lookat[1], self.cfg.viewer.lookat[2]
            cam_pos = gymapi.Vec3(cam_pos_x, cam_pos_y, cam_pos_z)
            cam_target = gymapi.Vec3(cam_target_x, cam_target_y, cam_target_z)
            cam_ref_env = self.cfg.viewer.ref_env
            
            self.gym.viewer_camera_look_at(self.viewer, None, cam_pos, cam_target)
        
        self.start_time = time.time()

    def create_sim(self):
        self.sim = self.gym.create_sim(self.sim_device_id, self.graphics_device_id, self.physics_engine, self.sim_params)
        if self.cfg.env.create_ground_plane:
            self._create_ground_plane()
        self._create_envs()
        self.progress_buf = torch.zeros(
            self.num_envs, device=self.sim_device, dtype=torch.int64)
        
    def _create_ground_plane(self):
        plane_params = gymapi.PlaneParams()
        plane_params.normal = gymapi.Vec3(0.0, 0.0, 1.0)
        self.gym.add_ground(self.sim, plane_params)
        return
    
    def _create_envs(self):
        print("\n\n\n\n\n CREATING ENVIRONMENT \n\n\n\n\n\n")
        asset_path = self.cfg.robot_asset.file.format(
            AERIAL_GYM_ROOT_DIR=AERIAL_GYM_ROOT_DIR)
        asset_root = os.path.dirname(asset_path)
        asset_file = os.path.basename(asset_path)
        asset_options = asset_class_to_AssetOptions(self.cfg.robot_asset)

        robot_asset = self.gym.load_asset(self.sim, asset_root, asset_file, asset_options)
        #TODO create the environment
        self.env_creator = EnvCreator(self.gym, self.sim, self._param_from_cfg(EnvCreatorParams, class_to_dict(self.cfg.envCreator)))
        self.env_creator.create(robot_asset, [0.0, 0.0, self.env_creator.params.env_size / 2])
        self.robot_num_bodies = self.gym.get_asset_rigid_body_count(robot_asset)

        self.drone_actors = self.env_creator.quad_actors
        self.envs = self.env_creator.envs
        self.num_waypoints_to_track = self.cfg.way_point_generator.num_waypoints - 1

        self.camera_handles = []
        self.camera_tensors = []
        if self.use_stereo_vision:
            self.camera_tensors_right = []
            self.sgm_handles = []

        # Set Camera Properties
        camera_props = gymapi.CameraProperties()
        camera_props.enable_tensors = True
        camera_props.width = self.sensor_params.width
        camera_props.height = self.sensor_params.height
        camera_props.far_plane = self.sensor_params.max_range
        camera_props.horizontal_fov = self.sensor_params.horizontal_fov_deg
        # local camera transform
        local_transform = gymapi.Transform()
        # position of the camera relative to the body
        local_transform.p = gymapi.Vec3(0.15, 0.00, 0.10)
        # orientation of the camera relative to the body
        local_transform.r = gymapi.Quat(0.0, 0.0, 0.0, 1.0)

        for i in range(self.num_maps):
            for j in range(self.num_robots_per_map):
                env_handle = self.envs[self.num_robots_per_map*i+j]
                actor_handle = self.drone_actors[self.num_robots_per_map*i+j]
                camera_handle = self.gym.create_camera_sensor(env_handle, camera_props)
                self.gym.attach_camera_to_body(camera_handle, env_handle, self.gym.get_actor_rigid_body_handle(env_handle, actor_handle, 0),
                                               local_transform, gymapi.FOLLOW_TRANSFORM)
                camera_tensor = self.gym.get_camera_image_gpu_tensor(self.sim, env_handle, camera_handle, gymapi.IMAGE_DEPTH)
                torch_cam_tensor = gymtorch.wrap_tensor(camera_tensor)
                self.camera_tensors.append(torch_cam_tensor)
        
        self.robot_body_props = self.gym.get_actor_rigid_body_properties(self.envs[0],self.drone_actors[0])
        self.robot_mass = 0
        self.robot_inertia = torch.zeros((self.num_envs,3,3), device=self.device, requires_grad=False)
        for prop in self.robot_body_props:
            self.robot_mass += prop.mass
            self.robot_inertia[:, 0,0] += prop.inertia.x.x
            self.robot_inertia[:, 1,1] += prop.inertia.y.y
            self.robot_inertia[:, 2,2] += prop.inertia.z.z
        print("Total robot mass: ", self.robot_mass)
        print("Robot Inertia:", self.robot_inertia[0])
        print("\n\n\n\n\n ENVIRONMENT CREATED \n\n\n\n\n\n")

    def reset(self, if_easy_start=False):
        self._randomize_racing_tracks()
        assert self.waypoint_data is not None
        self.waypoint_tracker.set_waypoint_data(self.waypoint_data)
        self.drone_manager.set_waypoint(self.waypoint_data)

        self.reset_idx(torch.arange(self.num_envs, device=self.device))
        # self.gym.step_graphics(self.sim)
        obs, privileged_obs, *_ = self.step(torch.zeros(self.num_envs, self.num_actions, device=self.device, requires_grad=False))
        return obs, privileged_obs

    def reset_idx(self, env_idx, if_reset_obstacles=True, if_easy_start=False):
        if len(env_idx) == 0:
            return
        if if_reset_obstacles:
            self._randomize_racing_tracks()
            assert self.waypoint_data is not None
            self.waypoint_tracker.set_waypoint_data(self.waypoint_data)
            self.drone_manager.set_waypoint(self.waypoint_data)

        drone_state, action, next_wp_id = self.drone_manager.compute(
            self.rand_drone_opts, False, env_idx)
        drone_state_tmp = drone_state[env_idx]
        # update actor root state and submit teleportation
        # if other actors are changed elsewhere, this line will submit those changes too
        # self.actor_root_state[self.drone_actor_id_flat[env_idx]] = drone_state[env_idx]
        maps_ids = env_idx // self.num_robots_per_map
        robot_ids = env_idx % self.num_robots_per_map
        pos_setting_start = 0
        pos_setting_end = 0
        for map_id in maps_ids.unique():
            map_env_ids = env_idx[maps_ids == map_id]
            pos_setting_end += len(map_env_ids)
            map_robot_ids = robot_ids[maps_ids == map_id]
            self.root_states[map_id, map_robot_ids, :] = drone_state_tmp[pos_setting_start:pos_setting_end]
            pos_setting_start = pos_setting_end
            self.terminal_rewards[map_id, map_robot_ids] = 0.0
            self.collisions[map_id, map_robot_ids * self.robot_num_bodies] = 0
        self.gym.set_actor_root_state_tensor(self.sim, self.root_tensor)
        self.progress_buf[env_idx] = 0

        # update next waypoint id
        self.next_waypoint_id[env_idx] = next_wp_id[env_idx]
        self.waypoint_tracker.set_init_drone_state_next_wp(
            drone_state, next_wp_id, env_idx
        )
        if self.cfg.control.use_reference_model:
            r, p, y = get_euler_xyz(drone_state_tmp[:, 3:7])
            drone_eulers = torch.zeros((len(env_idx), 3), device=self.device)
            drone_eulers[:, 0] = ssa(r)
            drone_eulers[:, 1] = ssa(p)
            drone_eulers[:, 2] = ssa(y) 

            drone_positions = drone_state_tmp[:, :3]
            drone_velocities = drone_state_tmp[:, 7:10]
            self.reference.reset_reference(env_idx, drone_eulers, torch.zeros_like(drone_eulers), drone_positions, 
                                           drone_velocities, torch.zeros_like(drone_positions))
            self.controller.reset(drone_eulers, env_idx)

        if self.cfg.control.randomize_params:
            self.reset_controllers(env_idx)
    
    def reset_controllers(self, env_ids):
        self.controller.randomize_params(env_ids)

    def _randomize_racing_tracks(self):
        # generate random waypoints for multiple tracks
        self.waypoint_data = self.waypoint_generator.compute(self.rand_waypoint_opts)
        if self.viewer and self.enable_debug_viz:
            self.gym.clear_lines(self.viewer)
            self.waypoint_data.visualize(self.gym, self.envs, self.viewer, 1)
        # place random obstacles around the waypoints if enabled
        # sometimes we do not want to compute gate and obstacles at all
        # e.g. for large amount of envs, state-only drone racing
        if not self.disable_obstacle_man:
            obs_actor_pose, obs_actor_id = self.obstacle_manager.compute(
                waypoint_data=self.waypoint_data, rand_obs_opts=self.rand_obstacle_opts
            )
            self.vec_root_tensor = self.vec_root_tensor.view(self.num_maps*self.num_actors_per_env, 13)
            self.vec_root_tensor[obs_actor_id, :7] = obs_actor_pose[obs_actor_id].to(self.device)
            self.vec_root_tensor = self.vec_root_tensor.view(self.num_maps, self.num_actors_per_env, 13)

    def _param_from_cfg(self, param_class, cfg_dict: dict):
        p = param_class()
        for key in cfg_dict.keys():
            assert hasattr(p, key), f"{p}, {key}"
            # if key starts with __, it is a private attribute, skip
            if key.startswith("__"):
                continue
            setattr(p, key, cfg_dict[key])
        if hasattr(p, "device"):
            p.device = self.device
        if hasattr(p, "num_envs"):
            p.num_envs = self.num_envs
        return p
    
    def compute_vehicle_frame_states(self):
        r, p, y = get_euler_xyz_3d(self.root_quats)
        r = ssa(r)
        p = ssa(p)
        y = ssa(y)
        euler_noise = self.euler_angle_noise_sampler.sample(self.zeros_3d, torch.ones_like(self.zeros_3d))
        self.root_euler_angles[:, :, 0] = r + 0.01 * euler_noise[:, :, 0]
        self.root_euler_angles[:, :, 1] = p + 0.01 * euler_noise[:, :, 1]
        self.root_euler_angles[:, :, 2] = y + 0.02 * euler_noise[:, :, 2]

        # vehicle frame is the same but with 0 roll and pitch 
        self.vehicle_frame_euler_angles[:] = self.zeros_3d
        self.vehicle_frame_euler_angles[:, :, 2] = self.root_euler_angles[:, :, 2]

        # vehicle frame quats
        self.vehicle_frame_quats[:] = quat_from_euler_xyz(self.vehicle_frame_euler_angles[:, :, 0], self.vehicle_frame_euler_angles[:, :, 1], self.vehicle_frame_euler_angles[:, :, 2])

        self.angvels_body_frame[:] = quat_rotate_inverse_3d(self.root_quats, self.root_angvels)
        angulvel_noise = self.angular_velocity_noise_sampler.sample(self.zeros_3d, torch.ones_like(self.zeros_3d))
        self.angvels_body_frame += 0.017 * angulvel_noise
        vel_noise = self.velocity_noise_sampler.sample(self.zeros_3d, 0.05 * self.root_linvels)
        self.root_linvels_obs = self.root_linvels + vel_noise
        self.linvels_body_frame[:] = quat_rotate_inverse_3d(self.root_quats, self.root_linvels_obs)
        self.linvels_vehicle_frame[:] = quat_rotate_inverse_3d(self.vehicle_frame_quats, self.root_linvels_obs)


    def render_cameras(self):
        if self.enable_isaacgym_cameras:
            self.render_isaacgym_cameras()
        else:
            return
    
    def render_isaacgym_cameras(self):
        self.gym.render_all_camera_sensors(self.sim)
        self.gym.start_access_image_tensors(self.sim)
        self.dump_images()
        self.gym.end_access_image_tensors(self.sim)
        return
    
    def dump_images(self):
        # only if cameras are enabled
        if self.enable_isaacgym_cameras and not self.use_stereo_vision:
            for env_id in range(self.num_envs):
                # the depth values are in -ve z axis, so we need to flip it to positive
                self.full_camera_array[env_id] = -self.camera_tensors[env_id]
        elif self.enable_isaacgym_cameras and self.use_stereo_vision:
            for env_id in range(self.num_envs):
                self.full_camera_array[env_id] = self.sgm_handles[env_id].compute_depth(self.camera_tensors[env_id], self.camera_tensors_right[env_id])


    def compute_resets(self):
        self.reset_buf[:] = 0
        # terminate for timeout
        self.reset_buf[self.progress_buf >= self.cfg.env.max_episode_length] = 1
        self.terminal_rewards[self.progress_buf.view(self.num_maps, self.num_robots_per_map,) >= self.cfg.env.max_episode_length] = self.cfg.RLParamsCfg.r_timeout
        self.total_trials[self.progress_buf >= self.cfg.env.max_episode_length] += 1
        # terminate for collision
        robot_body_ids = torch.arange(self.num_robots_per_map, device=self.device)*self.robot_num_bodies
        collision_robots = self.collisions[:, robot_body_ids].view(self.num_envs,)
        self.reset_buf[collision_robots > 0] = 1
        self.terminal_rewards[collision_robots.view(self.num_maps, self.num_robots_per_map,) > 0] = self.cfg.RLParamsCfg.r_collision
        self.total_trials[collision_robots > 0] += 1
        # terminate for reaching goal
        self.reset_buf[self.successes.view(self.num_envs,) > 0] = 1
        self.terminal_rewards[self.successes > 0] = self.cfg.RLParamsCfg.r_arrive
        self.total_trials[self.successes.view(self.num_envs,) > 0] += 1
        self.success_trials[self.successes.view(self.num_envs,) > 0] += 1
        # terminate for out of bounds
        self.reset_buf[self.out_of_bounds.view(self.num_envs,) > 0] = 1
        self.terminal_rewards[self.out_of_bounds > 0] = self.cfg.RLParamsCfg.r_exceed
        self.total_trials[self.out_of_bounds.view(self.num_envs,) > 0] += 1
        return
    
    def get_obs(self):
        self.state_observations = self.compute_state_observations()
        obs_image = torch.clamp(self.full_camera_array, self.sensor_params.min_range, self.sensor_params.max_range)
        obs_image = (obs_image - self.sensor_params.min_range) / (self.sensor_params.max_range - self.sensor_params.min_range) * 255.0

        self.obs_dict['image'] = obs_image.unsqueeze(1).to(torch.uint8)
        self.obs_dict['state'] = self.state_observations
        # print("Observation: ", self.obs_dict['state'][0, :])


    def compute_state_observations(self):
        # print("self.curr_goal_positions: ", self.curr_goal_positions[0, 0, :])
        goal_dir = self.curr_goal_positions - self.root_positions
        log_distance = torch.log(torch.norm(goal_dir[:, :, :2], dim=2) + 1.0)
        horizon_vel_ = torch.norm(self.linvels_body_frame[:, :, :2], dim=2)
        chi = torch.atan2(self.linvels_body_frame[:, :, 1], self.linvels_body_frame[:, :, 0])
        # beta = torch.atan2(goal_dir[:, :, 1], goal_dir[:, :, 0])
        normalized_goal_dir = goal_dir / (torch.norm(goal_dir, dim=2).unsqueeze(-1) + 1e-6)
        normalized_vel_dir = self.root_linvels / (torch.norm(self.root_linvels, dim=2).unsqueeze(-1) + 1e-6)
        # print("Normalized Goal Dir: ", normalized_goal_dir[0, 0, :])
        
        goal_penalty = self.cfg.RLParamsCfg.distance_coeff * log_distance
        speed_penalty = torch.where(horizon_vel_ > self.cfg.RLParamsCfg.start_penalty_vel, self.cfg.RLParamsCfg.vel_coeff * horizon_vel_, self.zeros)
        # vertical_penalty = self.cfg.RLParamsCfg.vert_coeff * (torch.abs(goal_dir[:, :, 2]) + 0.6*torch.abs(self.linvels_vehicle_frame[:, :, 2]))
        vertical_penalty = self.cfg.RLParamsCfg.vert_coeff * torch.abs(goal_dir[:, :, 2])

        # angular_penalty = self.cfg.RLParamsCfg.angular_vel_coeff * torch.abs(chi + self.vehicle_frame_euler_angles[:, :, 2] - beta)
        angular_penalty = self.cfg.RLParamsCfg.angular_vel_coeff * (torch.abs(normalized_goal_dir[:, :, 0] - normalized_vel_dir[:, :, 0]) + 
                                                                    torch.abs(normalized_goal_dir[:, :, 1] - normalized_vel_dir[:, :, 1]) +
                                                                    torch.abs(normalized_goal_dir[:, :, 2] - normalized_vel_dir[:, :, 2]))
        input_penalty = self.cfg.RLParamsCfg.input_coeff * torch.norm(self.angvels_body_frame, dim=2)
        yaw_penalty = self.cfg.RLParamsCfg.yaw_coeff * torch.abs(chi)
        total_penalty = goal_penalty + speed_penalty + vertical_penalty + angular_penalty + input_penalty + yaw_penalty
        self.rew_buf = torch.stack([goal_penalty, speed_penalty, vertical_penalty, angular_penalty, input_penalty, yaw_penalty, total_penalty], dim=2).view(self.num_envs, -1)
        return torch.stack([log_distance, self.root_positions[:, :, 2], normalized_goal_dir[:, :, 0], normalized_goal_dir[:, :, 1], normalized_goal_dir[:, :, 2], 
                            self.linvels_body_frame[:, :, 0], self.linvels_body_frame[:, :, 1], self.linvels_body_frame[:, :, 2],
                            self.angvels_body_frame[:, :, 0], self.angvels_body_frame[:, :, 1], self.angvels_body_frame[:, :, 2],
                            self.root_euler_angles[:, :, 0], self.root_euler_angles[:, :, 1], self.root_euler_angles[:, :, 2]], dim=2).view(self.num_envs, 1, -1)


    def step(self, actions):
        self.counter += 1
        timesteps = self.cfg.env.num_control_steps_per_env_step
        if timesteps < 1:
            print("[WARNING] Physics simulation is not running. timestep is less than 1.")
        time_durations = self.randomize_time_durations(timesteps)
        if self.save_control_data:
            self.states_file = open(self.control_data_path + "/states_data/states_data_" + str(self.counter) + ".txt", "ab")
            self.actions_file = open(self.control_data_path + "/actions_data/actions_data_" + str(self.counter) + ".txt", "ab")
            self.control_file = open(self.control_data_path + "/control_data/control_data_" + str(self.counter) + ".txt", "ab")
            self.reference_file = open(self.control_data_path + "/reference_data/reference_data_" + str(self.counter) + ".txt", "ab")
        # print("actions: ", actions[0, :])
        for dt in time_durations:
            self.pre_physics_step(actions, dt)
            self.physics_renders()
            if self.save_control_data:
                self.compute_vehicle_frame_states()

        if self.save_control_data:
            self.states_file.close()
            self.actions_file.close()
            self.control_file.close()
            self.reference_file.close()

        self.render(sync_frame_time=True)
        if self.enable_isaacgym_cameras:
            if self.cfg.env.manual_camera_trigger == False:
                self.render_cameras()
                
        self.progress_buf += 1

        # track waypoint
        track_state = self.root_states.clone().detach().reshape(self.num_envs, 13)
        self.waypoint_passing, self.next_waypoint_id = self.waypoint_tracker.compute(track_state)
        all_env_ids = torch.arange(self.num_envs, device=self.device)
        # print("self.waypoint_data.position: ", self.waypoint_data.position[0, :, :])
        self.curr_goal_positions = self.waypoint_data.position.to(self.device)[all_env_ids, self.next_waypoint_id, :].unsqueeze(1)
        # print("self.curr_goal_positions: ", self.curr_goal_positions[0, 0, :])
        # print("self.root_positions: ", self.root_positions[0, 0, :])
        # finish_curr_goal = torch.where(torch.norm(self.curr_goal_positions - self.root_positions, dim=2) < 0.2, self.ones_int, self.zeros_int)
        # self.curr_goal_index += finish_curr_goal
        # self.curr_goal_positions = torch.gather(self.goal_positions, 2, self.curr_goal_index.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, 1, 3)).squeeze(2)
        self.successes[:, 0] = torch.eq(self.next_waypoint_id, 0)
        _out_of_bounds = torch.where((self.root_positions < self.flight_lower_bound) | (self.root_positions > self.flight_upper_bound), self.ones_3d, self.zeros_3d)
        self.out_of_bounds = torch.where(torch.sum(_out_of_bounds, dim=2) > 0, self.ones, self.zeros)

        self.compute_resets()
        # print("Reset Buffer: ", self.reset_buf[0])
        reset_env_ids = self.reset_buf.nonzero(as_tuple=False).squeeze(-1)
        _terminal_rewards = self.terminal_rewards.view(-1).clone().detach()
        # if not self.save_control_data:
        self.reset_idx(reset_env_ids, if_reset_obstacles=False)
        self.compute_vehicle_frame_states()
        self.get_obs()
        self.rew_buf[:, -1] = torch.where(self.reset_buf>0, _terminal_rewards, self.rew_buf[:, -1])
        # self.rew_buf[:, -1] = torch.where(finish_curr_goal.view(-1) > 0, self.cfg.RLParamsCfg.r_arrive, self.rew_buf[:, -1])

        self.prev_action_input[:] = self.action_input.view(self.num_maps, self.num_robots_per_map, -1)
        return self.obs_dict, self.privileged_obs_buf, self.rew_buf, self.reset_buf, self.extras
    
    def randomize_time_durations(self, timesteps):
        target_sum = 0.1
        fixed_value = 0.01
        num_large_values = random.randint(2, 5)
        num_small_values = timesteps - num_large_values
        large_values_range = (0.005, 0.04)
        large_values = torch.FloatTensor(num_large_values).uniform_(*large_values_range).to(self.device)
        large_sum = large_values.sum()
        small_values_sum = fixed_value * num_small_values
        scaling_factor = (target_sum - small_values_sum) / large_sum
        scaled_large_values = large_values * scaling_factor
        small_values = torch.full((num_small_values,), fixed_value).to(self.device)
        final_distribution = torch.cat([scaled_large_values, small_values])
        final_distribution = final_distribution[torch.randperm(10)].to(self.device)
        return final_distribution
        
    def action_acc_bounding(self, actions):
        self.action_input[:] = actions * self.std_acc + self.mean_acc

    def pre_physics_step(self, _actions, duration):
        actions = _actions.to(self.device)
        actions = torch.clamp(actions, -1.0, 1.0)
        # actions[:] = self.apply_action_noise(actions)
        if self.cfg.control.controller == 'lee_acceleration_control':
            self.action_acc_bounding(actions)
        # print("Actions: ", self.action_input)
        # clear actions for reset envs
        if self.save_control_data:
            # state_np = self.root_states[0, :, :].cpu().numpy()
            pos_np = self.root_positions[0, :, :3].cpu().numpy()
            euler_np = self.root_euler_angles[0, :, :].cpu().numpy()
            vel_np = self.root_linvels_obs[0, :, :].cpu().numpy()
            angvel_np = self.angvels_body_frame[0, :, :].cpu().numpy()
            state_np = np.concatenate((pos_np, euler_np, vel_np, angvel_np), axis=1)
            actions_np = self.action_input[:self.num_robots_per_map].cpu().numpy()
            # save states
            np.savetxt(self.states_file, state_np, delimiter=' ' , fmt='%1.4f')
            # save actions
            np.savetxt(self.actions_file, actions_np, delimiter=' ' , fmt='%2.6f')

        self.forces[:] = 0.0
        self.torques[:, :] = 0.0
        control_state = self.root_states.clone().detach().reshape(self.num_envs, 13)
        if self.cfg.control.use_reference_model:
            euler_setpoints, output_thrusts_mass_normalized = self.controller.get_euler_setpoints_from_acc(control_state, self.action_input, self.reference, dt=duration)
            self.reference.attitude_ref_euler_float_update(euler_setpoints, duration, output_thrusts_mass_normalized)
            if self.save_control_data:
                euler_angles_np = self.reference.euler_angles[:self.num_robots_per_map].cpu().numpy()
                rates_np = self.reference.rates[:self.num_robots_per_map].cpu().numpy()
                pos_setpoints_np = self.reference.linear_pos[:self.num_robots_per_map].cpu().numpy()
                vel_setpoints_np = self.reference.linear_vel[:self.num_robots_per_map].cpu().numpy()
                acc_setpoints_np = self.reference.linear_accel[:self.num_robots_per_map].cpu().numpy()
                setpoints_np = np.concatenate((euler_angles_np, rates_np, pos_setpoints_np, vel_setpoints_np, acc_setpoints_np), axis=1)
                np.savetxt(self.reference_file, setpoints_np, delimiter=' ' , fmt='%2.6f')
            thrust_command, output_torques_inertia_normalized, angvel_err = self.controller.update_command(control_state, self.reference)
        else:
            output_thrusts_mass_normalized, output_torques_inertia_normalized = self.controller(control_state, self.action_input)
        robot_body_ids = torch.arange(self.num_robots_per_map, device=self.device)*self.robot_num_bodies

        if self.cfg.control.use_reference_model:
            self.forces[:, robot_body_ids, 2] = self.robot_mass * thrust_command.reshape(self.num_maps, self.num_robots_per_map)
            self.torques[:, robot_body_ids] = output_torques_inertia_normalized.reshape(self.num_maps, self.num_robots_per_map, 3)
        else:
            self.forces[:, robot_body_ids, 2] = self.robot_mass * (-self.sim_params.gravity.z) * output_thrusts_mass_normalized.reshape(self.num_maps, self.num_robots_per_map)
            self.torques[:, robot_body_ids] = torch.bmm(self.robot_inertia, output_torques_inertia_normalized.unsqueeze(2)).squeeze(2).reshape(self.num_maps, self.num_robots_per_map, 3)
        # negative thrusts should be clipped to 0
        self.forces = torch.where(self.forces < 0, torch.zeros_like(self.forces), self.forces)

        self.sample_noise_per_physics_step(self.forces[:, robot_body_ids], self.torques[:, robot_body_ids])
        apply_disturbance_per_env = self.env_disturbance_application_sampler.sample().unsqueeze(2)
        self.forces[:, robot_body_ids] += (apply_disturbance_per_env * (self.force_noise + self.extra_force_disturbance))
        self.torques[:, robot_body_ids] += (apply_disturbance_per_env * (self.torque_noise + self.extra_torque_disturbance))

        if self.save_control_data:
            force_np = self.forces[0, robot_body_ids, 2].cpu().numpy()
            force_np = force_np.reshape(-1, 1)
            torque_np = self.torques[0, robot_body_ids].cpu().numpy()
            angvel_err_np = angvel_err[:self.num_robots_per_map].cpu().numpy()
            # concatenate forces and torques
            control_np = np.concatenate((force_np, torque_np, angvel_err_np), axis=1)
            # print("Control: ", control_np)
            np.savetxt(self.control_file, control_np, delimiter=' ' , fmt='%2.8f')
        # apply actions
        self.gym.apply_rigid_body_force_tensors(self.sim, gymtorch.unwrap_tensor(self.forces), gymtorch.unwrap_tensor(self.torques), gymapi.LOCAL_SPACE)

    def physics_renders(self):
        self.gym.simulate(self.sim)
        self.gym.fetch_results(self.sim, True) # use only when device is not "cpu"
        self.post_physics_step()
        self.check_collisions()

    def post_physics_step(self):
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_net_contact_force_tensor(self.sim)

    def check_collisions(self):
        self.collisions[:] = torch.where(torch.norm(self.contact_forces, dim=2) > 0.1, self.ones_bodied, self.collisions)

    def sample_noise_per_physics_step(self, forces, torques):
        # additive noise for force and torque inputs with std devs scaled by magnitudes
        self.force_noise = self.env_force_sampler.sample(self.zeros_3d, 0.01*forces)
        self.torque_noise = self.env_torque_sampler.sample(self.zeros_3d, 0.01*torques)

        # additive noise for extra lateral disturbances
        self.extra_force_disturbance = self.env_force_sampler.sample(self.zeros_3d, 0.01*torch.ones_like(forces))
        self.extra_torque_disturbance = self.env_torque_sampler.sample(self.zeros_3d, 0.004*torch.ones_like(torques))

    def cal_success_rate(self):
        return torch.sum(self.success_trials)/torch.sum(self.total_trials)
    
    def getRewardNames(self):
        return self.cfg.RLParamsCfg.names
    
@torch.jit.script
def ssa(a: torch.Tensor) -> torch.Tensor:
    '''Smallest signed angle'''
    return torch.remainder(a+np.pi,2*np.pi) - np.pi

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
import torch
import sys
import cv2
from gymnasium import spaces
from typing import Dict, Any, Optional, List, Tuple

# isaacgym imports
from isaacgym import gymutil, gymtorch, gymapi
from isaacgym.torch_utils import *
from aerial_gym.utils.torch_utils_contrib import *
from aerial_gym import AERIAL_GYM_ROOT_DIR, AERIAL_GYM_ROOT_DIR
from aerial_gym.envs.base.base_task import BaseTask
from .mavrl_task_config import MAVRLTaskCfg
from .zoo_task_config import ZooTaskCfg
from aerial_gym.envs.controllers.controller import Controller
from aerial_gym.envs.reference.reference_base import ReferenceBase

from aerial_gym.utils.asset_manager import AssetManager
from aerial_gym.utils.mavrl_asset_manager import MAVRLAssetManager
from aerial_gym.utils.sampler import Sampler
from aerial_gym.utils.helpers import asset_class_to_AssetOptions, class_to_dict
from aerial_gym.utils.warp_sensor import WarpSensor
from aerial_gym.utils.imu_simulator_v2 import TensorIMUV2
from aerial_gym.utils.sgm_depth import SGM
from aerial_gym.utils.pcd_cameras import PCDCameraManager

class MAVRLTask(BaseTask):

    def __init__(self, cfg : ZooTaskCfg, sim_params, physics_engine, sim_device, headless):
        self.cfg = cfg
        self.sim_params = sim_params
        self.physics_engine = physics_engine
        self.sim_device_id = sim_device
        self.headless = headless

        self.enable_isaacgym_cameras = self.cfg.env.enable_isaacgym_cameras
        self.enable_pc_loader = self.cfg.env.enable_pc_loader
        self.use_stereo_vision = self.cfg.camera_params.use_stereo_vision
        self.use_warp = self.cfg.env.use_warp_rendering
        self.seed = self.cfg.seed
        self.env_asset_manager = MAVRLAssetManager(self.cfg, sim_device)
        # self.sample_timestep_for_latency = self.cfg.env.sample_timestep_for_latency
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
        print("Control data path: ", self.control_data_path)
        # Print near and far out of range values
        print("Near out of range value: ", self.sensor_params.near_out_of_range_value)
        print("Far out of range value: ", self.sensor_params.far_out_of_range_value)
        # sensor fovs
        print("Horizontal FOV: ", self.sensor_params.horizontal_fov_deg)
        print("Vertical FOV: ", self.sensor_params.vertical_fov_deg)

        self.imu_params = self.cfg.imu_config
        self.gravity = torch.tensor(self.cfg.sim.gravity, device=self.sim_device_id, requires_grad=False)
        
        # update number of observations before calling super class that initializes Box action spaces
        super().__init__(self.cfg, sim_params, physics_engine, sim_device, headless)
        # initialize observation and action spaces
        self.obs_space = spaces.Dict({
            'image': spaces.Box(
                low=0,
                high=255,
                shape=(1, self.sensor_params.height, self.sensor_params.width),
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
            'image': torch.zeros((self.num_envs, 1, self.cfg.LatentSpaceCfg.imput_image_size[0], self.cfg.LatentSpaceCfg.imput_image_size[1]), dtype=torch.uint8, device=self.device),
            'state': torch.zeros((self.num_envs, 1, self.num_obs), dtype=torch.float32, device=self.device)
        }
        self.state_observations = torch.zeros((self.num_envs, 1, self.num_obs), dtype=torch.float32, device=self.device)

        self.root_tensor = self.gym.acquire_actor_root_state_tensor(self.sim)
        self.contact_force_tensor = self.gym.acquire_net_contact_force_tensor(self.sim)
        self.force_sensor_tensor = self.gym.acquire_force_sensor_tensor(self.sim)

        if self.enable_isaacgym_cameras:
            self.full_camera_array = torch.zeros((self.num_envs, self.sensor_params.height, self.sensor_params.width), dtype=torch.float32, device=self.device, requires_grad=False)
            if self.use_stereo_vision and self.cfg.camera_params.stereo_ground_truth:
                self.stereo_ground_camera_array = torch.zeros((self.num_envs, self.sensor_params.height, self.sensor_params.width), dtype=torch.float32, device=self.device, requires_grad=False)
        self.goal_threshold = self.cfg.env.goal_arrive_threshold

        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_net_contact_force_tensor(self.sim)
        self.gym.refresh_force_sensor_tensor(self.sim)

        self.robot_spawning_pos_min = torch.tensor(self.cfg.robot_spawning_config.min_position_ratio, device=self.device).expand(self.num_envs, -1)
        self.robot_spawning_pos_max = torch.tensor(self.cfg.robot_spawning_config.max_position_ratio, device=self.device).expand(self.num_envs, -1)
        self.robot_spawning_euler_min = torch.tensor(self.cfg.robot_spawning_config.min_euler_angles_absolute, device=self.device).expand(self.num_envs, -1)
        self.robot_spawning_euler_max = torch.tensor(self.cfg.robot_spawning_config.max_euler_angles_absolute, device=self.device).expand(self.num_envs, -1)
        self.goal_spawning_pos_min = torch.tensor(self.cfg.goal_spawning_config.min_position_ratio, device=self.device).expand(self.num_envs, -1)
        self.goal_spawning_pos_max = torch.tensor(self.cfg.goal_spawning_config.max_position_ratio, device=self.device).expand(self.num_envs, -1)

        self.robot_spawning_offset = torch.tensor(self.cfg.robot_spawning_config.offset, device=self.device).expand(self.num_envs, -1)
        self.goal_spawning_offset = torch.tensor(self.cfg.goal_spawning_config.offset, device=self.device).expand(self.num_envs, -1)
        self.random_start_yaw = self.cfg.robot_spawning_config.random_start_yaw

        num_actors = self.env_asset_manager.get_env_actor_count() + self.num_robots_per_map # Number of obstacles in the environment + one robot
        bodies_per_map = self.env_asset_manager.get_env_link_count() + self.num_robots_per_map * self.robot_num_bodies # Number of links in the environment + robot
        self.unfolded_vec_root_tensor = gymtorch.wrap_tensor(self.root_tensor)
        self.vec_root_tensor = self.unfolded_vec_root_tensor.view(self.num_maps, num_actors, 13)
        self.root_states = self.vec_root_tensor[:, :self.num_robots_per_map, :]
        self.root_positions = self.root_states[..., 0:3]
        self.root_quats = self.root_states[..., 3:7]
        self.root_linvels = self.root_states[..., 7:10]
        self.root_angvels = self.root_states[..., 10:13]
        self.env_asset_root_states = self.vec_root_tensor[:, self.num_robots_per_map:, :]

        self.max_inclination_angle = self.cfg.env.max_inclination_angle
        self.max_yawrate = self.cfg.env.max_yawrate
        self.max_speed = self.cfg.env.max_speed
        max_acc = torch.FloatTensor(self.cfg.env.act_max).to(self.device).requires_grad_(False)
        min_acc = torch.FloatTensor(self.cfg.env.act_min).to(self.device).requires_grad_(False)

        self.mean_acc = (max_acc + min_acc) / 2.0
        self.std_acc = (max_acc - min_acc) / 2.0

        # cannot use both IsaacGym cameras and warp
        if not self.enable_isaacgym_cameras:
            print("[ERROR] Do not use both IsaacGym cameras and warp")
            sys.exit(0)

        # if any of the shape is 0 set opbject to none
        if any([self.env_asset_root_states.shape[0] == 0, self.env_asset_root_states.shape[1] == 0, self.env_asset_root_states.shape[2] == 0]):
            self.env_asset_root_states = None
        self.init_obstacles()

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

        self.contact_forces = gymtorch.wrap_tensor(self.contact_force_tensor).view(self.num_maps, bodies_per_map, 3)[:, :self.robot_num_bodies*self.num_robots_per_map]
        if self.imu_params.use_imu:
            self.force_sensor = gymtorch.wrap_tensor(self.force_sensor_tensor).view(self.num_envs, -1)
            self.accel_meas = torch.zeros((self.num_envs, 3), device=self.device, requires_grad=False)
            self.angvel_meas = torch.zeros((self.num_envs, 3), device=self.device, requires_grad=False)

        self.collisions = torch.zeros(self.num_maps, self.num_robots_per_map * self.robot_num_bodies, device=self.device)
        self.ones = torch.ones(self.num_maps, self.num_robots_per_map, device=self.device, requires_grad=False)
        self.zeros = torch.zeros(self.num_maps, self.num_robots_per_map, device=self.device, requires_grad=False)
        self.zeros_3d = torch.zeros((self.num_maps, self.num_robots_per_map, 3), device=self.device, requires_grad=False)
        self.ones_3d = torch.ones((self.num_maps, self.num_robots_per_map, 3), device=self.device, requires_grad=False)
        self.zeros_int = torch.zeros((self.num_maps, self.num_robots_per_map), dtype=torch.int64, device=self.device, requires_grad=False)
        self.ones_int = torch.ones((self.num_maps, self.num_robots_per_map), dtype=torch.int64, device=self.device, requires_grad=False)
        self.rew_buf = torch.zeros((self.num_envs, len(self.cfg.RLParamsCfg.names)), device=self.device, requires_grad=False)
        self.terminal_rewards = torch.zeros(self.num_maps, self.num_robots_per_map, device=self.device)

        self.ones_bodied = torch.ones(self.num_maps, self.num_robots_per_map * self.robot_num_bodies, device=self.device, requires_grad=False)

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
        self.forces = torch.zeros((self.num_maps, bodies_per_map, 3),
                                  dtype=torch.float32, device=self.device, requires_grad=False)
        self.torques = torch.zeros((self.num_maps, bodies_per_map, 3),
                                   dtype=torch.float32, device=self.device, requires_grad=False)

        self.successes = torch.zeros_like(self.ones)
        self.out_of_bounds = torch.zeros_like(self.ones)

        self.total_trials = torch.zeros(self.num_envs, device=self.device, dtype=torch.int16, requires_grad=False)
        self.available_total_trials = torch.zeros(self.num_envs, device=self.device, dtype=torch.int16, requires_grad=False)
        self.success_trials = torch.zeros(self.num_envs, device=self.device, dtype=torch.int16, requires_grad=False)

        self.controller = Controller(self.robot_inertia, self.cfg.control, self.num_envs, self.device)
        self.reference = ReferenceBase(self.cfg.FeasibilityCfg, self.num_envs, self.cfg.env.num_control_steps_per_env_step, self.device)

        # Getting environment bounds
        self.env_lower_bound = torch.zeros(
            (self.num_envs, 3), dtype=torch.float32, device=self.device, requires_grad=False)
        self.env_upper_bound = torch.zeros(
            (self.num_envs, 3), dtype=torch.float32, device=self.device, requires_grad=False)
        
        self.goal_positions = torch.zeros((self.num_maps, self.num_robots_per_map, self.cfg.env.goal_num_per_episode + 1, 3), dtype=torch.float32, device=self.device, requires_grad=False)
        self.curr_goal_positions = torch.zeros((self.num_maps, self.num_robots_per_map, 3), dtype=torch.float32, device=self.device, requires_grad=False)
        self.curr_goal_index = torch.zeros(self.num_maps, self.num_robots_per_map, dtype=torch.int64, device=self.device, requires_grad=False)
        self.flight_lower_bound = torch.FloatTensor(self.cfg.env.flight_lower_bound).to(self.device).expand(self.num_maps, self.num_robots_per_map, -1)
        self.flight_upper_bound = torch.FloatTensor(self.cfg.env.flight_upper_bound).to(self.device).expand(self.num_maps, self.num_robots_per_map, -1)

        if self.viewer:
            cam_pos_x, cam_pos_y, cam_pos_z = self.cfg.viewer.pos[0], self.cfg.viewer.pos[1], self.cfg.viewer.pos[2]
            cam_target_x, cam_target_y, cam_target_z = self.cfg.viewer.lookat[0], self.cfg.viewer.lookat[1], self.cfg.viewer.lookat[2]
            cam_pos = gymapi.Vec3(cam_pos_x, cam_pos_y, cam_pos_z)
            cam_target = gymapi.Vec3(cam_target_x, cam_target_y, cam_target_z)
            cam_ref_env = self.cfg.viewer.ref_env
            
            self.gym.viewer_camera_look_at(self.viewer, None, cam_pos, cam_target)
        
        # logging
        self.log_data_dict: Optional[Dict[str, Any]] = None
        if self.enable_pc_loader:
            self._init_log_data_dict()
            # initialize start and end points for evaluation trials
            self.total_pos_rand_samples = torch.rand((self.cfg.logging.trial_nums*5, self.num_envs, 3), device=self.device)
            self.total_goal_rand_samples = torch.rand((self.cfg.logging.trial_nums*5, self.num_envs, 3), device=self.device)

    def create_sim(self):
        self.sim = self.gym.create_sim(
            self.sim_device_id, self.graphics_device_id, self.physics_engine, self.sim_params)
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

        if self.imu_params.use_imu:
            sensor_props = gymapi.ForceSensorProperties()
            sensor_props.enable_forward_dynamics_forces = True
            sensor_props.enable_constraint_solver_forces = True
            sensor_props.use_world_frame = self.imu_params.world_frame
            position = gymapi.Vec3(self.imu_params.pos[0], self.imu_params.pos[1], self.imu_params.pos[2])

            # NOTE orientation of the sensor relative to the body is set to zero here,
            # and the measured value is transformed to the desired orientationin the imu simulator code.
            # This is done to let the physics engine handle the effects of rotating
            # frames such as coriolis and centrifugal forces on the force sensor.
            quat_imu = gymapi.Quat(0.0, 0.0, 0.0, 1.0)
            force_sensor_pose = gymapi.Transform(p=position, r=quat_imu)
            force_sensor_idx = self.gym.create_asset_force_sensor(robot_asset, 0, force_sensor_pose, sensor_props)

            self.imu_sensor = TensorIMUV2(num_envs=self.num_envs, 
                                        dt=self.dt,
                                        config=self.imu_params,
                                        gravity=self.gravity,
                                        device=self.device)
        self.robot_num_bodies = self.gym.get_asset_rigid_body_count(robot_asset)

        start_pose = gymapi.Transform()
        # create env instance
        pos = torch.tensor([0, 0, 0], device=self.device)
        start_pose.p = gymapi.Vec3(*pos)
        self.env_spacing = self.cfg.env.env_spacing
        env_lower = gymapi.Vec3(-self.env_spacing, -
                                self.env_spacing, -self.env_spacing)
        env_upper = gymapi.Vec3(
            self.env_spacing, self.env_spacing, self.env_spacing)
        self.actor_handles = []
        self.env_asset_handles = []
        self.envs = []
        self.camera_handles = []
        self.camera_tensors = []
        if self.use_stereo_vision:
            self.camera_tensors_right = []
            self.sgm_handles = []
            if self.cfg.camera_params.stereo_ground_truth:
                self.stereo_ground_truth_tensors = []

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

        self.segmentation_counter = 0

        self.global_vertex_counter = 0
        self.global_asset_counter = 0
        self.global_vertex_to_asset_index_map = []
        self.global_vertex_semantics = []
        item_lookup_dict = {}

        # configure point cloud loader
        if self.enable_pc_loader:
            self.pcd_cameras = PCDCameraManager(self.gym, self.sim, self.cfg.pcd_camera_params)

        for i in range(self.num_maps):
            self.env_meshes = []
            env_handle = self.gym.create_env(self.sim, env_lower, env_upper, int(np.sqrt(self.num_maps)))

            for j in range(self.num_robots_per_map):
                # if i % 10 == 0:
                #     print("Env :", i, " created.")
                self.global_asset_counter += 1
                actor_handle = self.gym.create_actor(env_handle, robot_asset, start_pose, "robot", i, self.cfg.robot_asset.collision_mask, 0)

                if self.enable_isaacgym_cameras and not self.use_stereo_vision:
                    cam_handle = self.gym.create_camera_sensor(env_handle, camera_props)
                    self.gym.attach_camera_to_body(cam_handle, env_handle, self.gym.get_actor_rigid_body_handle(env_handle, actor_handle, 0), local_transform, gymapi.FOLLOW_TRANSFORM)
                    self.camera_handles.append(cam_handle)
                    camera_tensor = self.gym.get_camera_image_gpu_tensor(self.sim, env_handle, cam_handle, gymapi.IMAGE_DEPTH)
                    torch_cam_tensor = gymtorch.wrap_tensor(camera_tensor)
                    self.camera_tensors.append(torch_cam_tensor)
                elif self.enable_isaacgym_cameras and self.use_stereo_vision:
                    sgm_handle = SGM(self.sensor_params.width, self.sensor_params.height, self.sensor_params.horizontal_fov_deg, self.cfg.camera_params.baseline, self.device)
                    self.sgm_handles.append(sgm_handle)
                    
                    left_local_transform = gymapi.Transform()
                    right_local_transform = gymapi.Transform()
                    left_local_transform.p = gymapi.Vec3(0.15, self.cfg.camera_params.baseline / 2.0, 0.10)
                    right_local_transform.p = gymapi.Vec3(0.15, -self.cfg.camera_params.baseline / 2.0, 0.10)
                    left_local_transform.r = gymapi.Quat(0.0, 0.0, 0.0, 1.0)
                    right_local_transform.r = gymapi.Quat(0.0, 0.0, 0.0, 1.0)
                    left_cam_handle = self.gym.create_camera_sensor(env_handle, camera_props)
                    right_cam_handle = self.gym.create_camera_sensor(env_handle, camera_props)
                    self.gym.attach_camera_to_body(left_cam_handle, env_handle, self.gym.get_actor_rigid_body_handle(env_handle, actor_handle, 0), left_local_transform, gymapi.FOLLOW_TRANSFORM)
                    self.gym.attach_camera_to_body(right_cam_handle, env_handle, self.gym.get_actor_rigid_body_handle(env_handle, actor_handle, 0), right_local_transform, gymapi.FOLLOW_TRANSFORM)
                    left_camera_tensor = self.gym.get_camera_image_gpu_tensor(self.sim, env_handle, left_cam_handle, gymapi.IMAGE_COLOR)
                    right_camera_tensor = self.gym.get_camera_image_gpu_tensor(self.sim, env_handle, right_cam_handle, gymapi.IMAGE_COLOR)
                    torch_left_cam_tensor = gymtorch.wrap_tensor(left_camera_tensor)
                    torch_right_cam_tensor = gymtorch.wrap_tensor(right_camera_tensor)
                    self.camera_tensors.append(torch_left_cam_tensor)
                    self.camera_tensors_right.append(torch_right_cam_tensor)

                    if self.cfg.camera_params.stereo_ground_truth:
                        stereo_ground_truth_handle = self.gym.create_camera_sensor(env_handle, camera_props)
                        self.gym.attach_camera_to_body(stereo_ground_truth_handle, env_handle, 
                                                       self.gym.get_actor_rigid_body_handle(env_handle, actor_handle, 0), 
                                                       left_local_transform, gymapi.FOLLOW_TRANSFORM)
                        stereo_ground_truth_tensor = self.gym.get_camera_image_gpu_tensor(self.sim, env_handle, stereo_ground_truth_handle, gymapi.IMAGE_DEPTH)
                        torch_stereo_ground_truth_tensor = gymtorch.wrap_tensor(stereo_ground_truth_tensor)
                        self.stereo_ground_truth_tensors.append(torch_stereo_ground_truth_tensor)

                if self.enable_pc_loader and j == 0:
                    self.pcd_cameras.create_pcd_cameras_single_drone(env_handle, self.gym.get_actor_rigid_body_handle(env_handle, actor_handle, 0))

            env_asset_list = self.env_asset_manager.prepare_assets_for_simulation()
            asset_counter = 0

            # have the segmentation counter be the max defined semantic id + 1. Use this to set the semantic mask of objects that are
            # do not have a defined semantic id in the config file, but still requre one. Increment for every instance in the next snippet
            for dict_item in env_asset_list:
                self.segmentation_counter = max(self.segmentation_counter, int(dict_item["semantic_id"])+1)

            for dict_item in env_asset_list:
                folder_path = dict_item["asset_folder_path"]
                filename = dict_item["asset_file_name"]
                folder_path_plus_filename = os.path.join(folder_path, filename)
                asset_options = dict_item["asset_options"]
                whole_body_semantic = dict_item["body_semantic_label"]
                per_link_semantic = dict_item["link_semantic_label"]
                semantic_masked_links = dict_item["semantic_masked_links"]
                semantic_id = dict_item["semantic_id"]
                color = dict_item["color"]
                collision_mask = dict_item["collision_mask"]
                create_texture = dict_item["create_texture"]

                assert not (whole_body_semantic and per_link_semantic)
                if semantic_id < 0:
                    object_segmentation_id = self.segmentation_counter
                    self.segmentation_counter += 1
                else:
                    object_segmentation_id = semantic_id

                # lookup asset before loading
                if folder_path_plus_filename in item_lookup_dict:
                    loaded_asset = item_lookup_dict[folder_path_plus_filename]
                else:
                    loaded_asset = self.gym.load_asset(self.sim, folder_path, filename, asset_options)
                    item_lookup_dict[folder_path_plus_filename] = loaded_asset

                segmentation_counter_before_warp = self.segmentation_counter - 1
                segmentation_id_before_warp = object_segmentation_id

                asset_counter += 1

                self.segmentation_counter = segmentation_counter_before_warp
                object_segmentation_id = segmentation_id_before_warp
                assert not (whole_body_semantic and per_link_semantic)
                if semantic_id < 0:
                    object_segmentation_id = self.segmentation_counter
                    self.segmentation_counter += 1
                else:
                    object_segmentation_id = semantic_id

                env_asset_handle = self.gym.create_actor(env_handle, loaded_asset, start_pose, "env_asset_"+str(asset_counter), i, collision_mask, object_segmentation_id)
                self.env_asset_handles.append(env_asset_handle)
                if len(self.gym.get_actor_rigid_body_names(env_handle, env_asset_handle)) > 1:
                    print("Env asset has rigid body with more than 1 link: ", len(self.gym.get_actor_rigid_body_names(env_handle, env_asset_handle)))
                    sys.exit(0)

                if per_link_semantic:
                    rigid_body_names = None
                    if len(semantic_masked_links) == 0:
                        rigid_body_names = self.gym.get_actor_rigid_body_names(env_handle, env_asset_handle)
                    else:
                        rigid_body_names = semantic_masked_links
                    for rb_index in range(len(rigid_body_names)):
                        self.segmentation_counter += 1
                        self.gym.set_rigid_body_segmentation_id(env_handle, env_asset_handle, rb_index, self.segmentation_counter)
            
                if color is None:
                    color = np.random.randint(low=50,high=200,size=3)

                if create_texture:
                    # create a random pixelArray as a numpy array of type uint8_t with size [height, width*4] of packed RGBA values. Alpha values should always be 255 on input.
                    pixelArray = np.random.randint(low=50, high=255, size=(256, 256, 4), dtype=np.uint8)
                    # set Alpha values to 255
                    # pixelArray[:, :, 3] = 255
                    pixelArray.reshape(256, 256*4)
                    texture_handle = self.gym.create_texture_from_buffer(self.sim, 256, 256, pixelArray)
                    self.gym.set_rigid_body_texture(env_handle, env_asset_handle, 0, gymapi.MESH_VISUAL, texture_handle)

                else:
                    self.gym.set_rigid_body_color(env_handle, env_asset_handle, 0, gymapi.MESH_VISUAL,
                        gymapi.Vec3(color[0]/255,color[1]/255,color[2]/255))

            self.envs.append(env_handle)
            self.actor_handles.append(actor_handle)
        
        self.robot_body_props = self.gym.get_actor_rigid_body_properties(self.envs[0],self.actor_handles[0])
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
        # Total number of vertices in all meshes:
        print("\n\n\n\n\n\n", "Total number of vertices in all meshes: ", self.global_vertex_counter, "\n\n\n\n\n")


    def reset_idx(self, env_ids, if_reset_obstacles=True, if_easy_start=False, if_set_seed=False):
        if if_reset_obstacles and if_set_seed:
            self.set_seed(self.seed)

        if len(env_ids) == 0:
            return
        self.reset_robots(env_ids)

        if if_reset_obstacles:
            self.reset_obstacles(if_easy_start=if_easy_start)
            # self.extras['success rate'] = self.cal_success_rate()
            # print("Success rate: ", self.cal_success_rate().cpu().numpy())
            self.total_trials[env_ids] = 0
            self.available_total_trials[env_ids] = 0
            self.success_trials[env_ids] = 0

        self.gym.set_actor_root_state_tensor(self.sim, self.root_tensor)
        self.progress_buf[env_ids] = 0

    def update_warp_meshes(self, env_ids):
        self.vertex_maps_per_env_updated[:] = tf_apply(self.unfolded_vec_root_tensor[self.global_vertex_to_asset_index_map, 3:7], 
                                                    self.unfolded_vec_root_tensor[self.global_vertex_to_asset_index_map, 0:3],
                                                    self.vertex_maps_per_env_original[:])
        
        self.warp_sensor.refit_meshes(self.warp_mesh_per_env, [0])
        # for e_i in env_ids:
        #     self.warp_mesh_per_env[e_i].refit()
        return
    
    def update_warp_cam_tfs(self, env_ids):
        warp_cam_sampled_pos = torch_rand_float_tensor(self.warp_sensor_pos_min, self.warp_sensor_pos_max)
        warp_cam_sampled_rot = torch_rand_float_tensor(self.warp_sensor_rot_min, self.warp_sensor_rot_max)
        sampled_cam_quat_from_euler = quat_from_euler_xyz(warp_cam_sampled_rot[:, 0], warp_cam_sampled_rot[:, 1], warp_cam_sampled_rot[:, 2])
        self.sensor_local_pos[env_ids] = warp_cam_sampled_pos[env_ids]
        self.sensor_local_quat[env_ids] = sampled_cam_quat_from_euler[env_ids]
        return
    
    def get_boundaries(self, arrays):
        # point_pairs = [
        #     torch.tensor([[0.0, 0.05, 0.3], [0.05, 0.95, 0.6]], device=self.device),     # [0, 0]
        #     torch.tensor([[0.05, 0.95, 0.3], [0.95, 1.0, 0.6]], device=self.device),   # [0, 1]
        #     torch.tensor([[0.05, 0.0, 0.3], [0.95, 0.05, 0.6]], device=self.device),     # [1, 0]
        #     torch.tensor([[0.95, 0.05, 0.3], [1.0, 0.95, 0.6]], device=self.device)   # [1, 1]
        # ]
        # yaw = [
        #     0, -np.pi/2, np.pi/2, np.pi
        # ]
        point_pairs = [
            torch.tensor([[0.0, 0.05, 0.3], [0.05, 0.95, 0.6]], device=self.device),     # [0, 0]
            torch.tensor([[0.0, 0.05, 0.3], [0.05, 0.95, 0.6]], device=self.device),   # [0, 1]
            torch.tensor([[0.0, 0.05, 0.3], [0.05, 0.95, 0.6]], device=self.device),     # [1, 0]
            torch.tensor([[0.0, 0.05, 0.3], [0.05, 0.95, 0.6]], device=self.device)   # [1, 1]
        ]
        yaw = [
            0, 0, 0, 0
        ]
        results = torch.stack([point_pairs[(array[0] * 2 + array[1]).item()] for array in arrays])
        yaw_np = np.stack([yaw[(array[0] * 2 + array[1]).item()] for array in arrays])
        yaw_tensor = torch.tensor(yaw_np, device=self.device, dtype=torch.float32)
        return results, yaw_tensor
    
    def get_goal_boundaries(self, arrays):
        point_pairs = [
            torch.tensor([[0.95, 0.05, 0.3], [1.0, 0.95, 0.6]], device=self.device),     # [0, 0]
            torch.tensor([[0.95, 0.05, 0.3], [1.0, 0.95, 0.6]], device=self.device),   # [0, 1]
            torch.tensor([[0.95, 0.05, 0.3], [1.0, 0.95, 0.6]], device=self.device),     # [1, 0]
            torch.tensor([[0.95, 0.05, 0.3], [1.0, 0.95, 0.6]], device=self.device)   # [1, 1]
        ]
        yaw = [
            np.pi, np.pi, np.pi, np.pi
        ]
        results = torch.stack([point_pairs[(array[0] * 2 + array[1]).item()] for array in arrays])
        yaw_np = np.stack([yaw[(array[0] * 2 + array[1]).item()] for array in arrays])
        yaw_tensor = torch.tensor(yaw_np, device=self.device, dtype=torch.float32)
        return results, yaw_tensor
    
    # def get_boundaries(self, arrays):
    #     point_pairs = [
    #         torch.tensor([[0.0, 0.05, 0.3], [0.05, 0.95, 0.6]], device=self.device),     # [0, 0]
    #         torch.tensor([[0.0, 0.05, 0.3], [0.05, 0.95, 0.6]], device=self.device),   # [0, 1]
    #         torch.tensor([[0.0, 0.05, 0.3], [0.05, 0.95, 0.6]], device=self.device),     # [1, 0]
    #         torch.tensor([[0.95, 0.05, 0.3], [1.0, 0.95, 0.6]], device=self.device)   # [1, 1]
    #     ]
    #     yaw = [
    #         0, 0, np.pi, np.pi
    #     ]
    #     results = torch.stack([point_pairs[(array[0] * 2 + array[1]).item()] for array in arrays])
    #     yaw_np = np.stack([yaw[(array[0] * 2 + array[1]).item()] for array in arrays])
    #     yaw_tensor = torch.tensor(yaw_np, device=self.device, dtype=torch.float32)
    #     return results, yaw_tensor
    
    # def get_goal_boundaries(self, arrays):
    #     point_pairs = [
    #         torch.tensor([[0.0, 0.05, 0.3], [0.05, 0.95, 0.6]], device=self.device),     # [0, 0]
    #         torch.tensor([[0.95, 0.05, 0.3], [1.0, 0.95, 0.6]], device=self.device),   # [0, 1]
    #         torch.tensor([[0.95, 0.05, 0.3], [1.0, 0.95, 0.6]], device=self.device),     # [1, 0]
    #         torch.tensor([[0.95, 0.05, 0.3], [1.0, 0.95, 0.6]], device=self.device)   # [1, 1]
    #     ]
    #     yaw = [
    #         np.pi, np.pi, np.pi, np.pi
    #     ]
    #     results = torch.stack([point_pairs[(array[0] * 2 + array[1]).item()] for array in arrays])
    #     yaw_np = np.stack([yaw[(array[0] * 2 + array[1]).item()] for array in arrays])
    #     yaw_tensor = torch.tensor(yaw_np, device=self.device, dtype=torch.float32)
    #     return results, yaw_tensor

    def reset_robots(self, env_ids):
        num_resets = len(env_ids)

        # get environment lower and upper bounds
        self.env_lower_bound[env_ids] = self.env_asset_manager.env_lower_bound.tile(len(env_ids), 1)
        self.env_upper_bound[env_ids] = self.env_asset_manager.env_upper_bound.tile(len(env_ids), 1)

        # sampling area for drone and goal positions
        area = torch.randint(0, 2, (num_resets, 2), device=self.device)
        drone_boundaries, yaw_tensor = self.get_boundaries(area)
        drone_random_area_min = drone_boundaries[:, 0, :]
        drone_random_area_max = drone_boundaries[:, 1, :]
        # sample and set drone and goal positions
        drone_pos_rand_sample = torch.rand((num_resets, 3), device=self.device)
        if self.enable_pc_loader:
            for i in range(num_resets):
                if self.available_total_trials[env_ids[i]] < self.cfg.logging.trial_nums:
                    drone_pos_rand_sample[i] = self.total_pos_rand_samples[self.total_trials[env_ids[i]], env_ids[i]]

        drone_spawning_min_bounds = self.env_lower_bound[env_ids] + self.robot_spawning_offset[env_ids]
        drone_spawning_max_bounds = self.env_upper_bound[env_ids] - self.robot_spawning_offset[env_ids]
        drone_spawning_min_bounds[:, 2] = 0.8
        drone_spawning_max_bounds[:, 2] = 1.5
        drone_spawning_ratio_in_env_bound = drone_pos_rand_sample*(drone_random_area_max - drone_random_area_min) + drone_random_area_min
        drone_positions = drone_spawning_min_bounds + drone_spawning_ratio_in_env_bound*(drone_spawning_max_bounds - drone_spawning_min_bounds)
        drone_velocities = 0.0*torch_rand_float(-1.0, 1.0, (num_resets, 3), self.device)
        drone_eulers = torch_rand_float_tensor(self.robot_spawning_euler_min[env_ids], self.robot_spawning_euler_max[env_ids])
        drone_eulers[:, 2] += yaw_tensor
        # if reset the drone foward direction to be the same as the goal direction
        if not self.random_start_yaw:
            drone_eulers[:, 2] = yaw_tensor
        drone_quats = quat_from_euler_xyz(drone_eulers[:, 0], drone_eulers[:, 1], drone_eulers[:, 2])
        
        goal_positions_total = torch.zeros((num_resets, self.cfg.env.goal_num_per_episode + 1, 3), device=self.device)
        for i in range(self.cfg.env.goal_num_per_episode):
            if i % 2 == 0:
                goal_area = torch.neg(area)+1
            else:
                goal_area = area
            goal_boundaries, _ = self.get_goal_boundaries(goal_area)
            goal_random_area_min = goal_boundaries[:, 0, :]
            goal_random_area_max = goal_boundaries[:, 1, :]
            # drone_quats = quat_from_euler_xyz(torch.zeros_like(drone_eulers[:, 0]), torch.zeros_like(drone_eulers[:, 1]), yaw_tensor)

            # sample and set drone and goal positions
            goal_pos_rand_sample = torch.rand((num_resets, 3), device=self.device)

            if self.enable_pc_loader:
                for j in range(num_resets):
                    if self.available_total_trials[env_ids[j]] < self.cfg.logging.trial_nums:
                        goal_pos_rand_sample[j] = self.total_goal_rand_samples[self.total_trials[env_ids[j]], env_ids[j]]

            goal_spawning_min_bounds = self.env_lower_bound[env_ids] + self.goal_spawning_offset[env_ids]
            goal_spawning_max_bounds = self.env_upper_bound[env_ids] - self.goal_spawning_offset[env_ids]
            goal_spawning_min_bounds[:, 2] = 1.0
            goal_spawning_max_bounds[:, 2] = 1.5
            goal_spawning_ratio_in_env_bound = goal_pos_rand_sample*(goal_random_area_max - goal_random_area_min) + goal_random_area_min
            goal_positions = goal_spawning_min_bounds + goal_spawning_ratio_in_env_bound*(goal_spawning_max_bounds - goal_spawning_min_bounds)
            goal_positions_total[:, i, :] = goal_positions
        # set drone positions that are sampled within environment bounds
        # print("drone_positions: ", drone_positions)
        # print("goal_positions_total: ", goal_positions_total)
        maps_ids = env_ids // self.num_robots_per_map
        robot_ids = env_ids % self.num_robots_per_map
        pos_setting_start = 0
        pos_setting_end = 0
        for map_id in maps_ids.unique():
            map_env_ids = env_ids[maps_ids == map_id]
            pos_setting_end += len(map_env_ids)
            map_robot_ids = robot_ids[maps_ids == map_id]
            self.root_states[map_id, map_robot_ids, 0:3] = drone_positions[pos_setting_start:pos_setting_end]
            self.root_states[map_id, map_robot_ids, 3:7] = drone_quats[pos_setting_start:pos_setting_end]
            self.root_states[map_id, map_robot_ids, 7:10] = drone_velocities[pos_setting_start:pos_setting_end]
            self.root_states[map_id, map_robot_ids, 10:13] = torch.zeros((len(map_env_ids), 3), device=self.device)
            self.goal_positions[map_id, map_robot_ids] = goal_positions_total[pos_setting_start:pos_setting_end]
            self.curr_goal_index[map_id, map_robot_ids] = 0
            pos_setting_start = pos_setting_end
            self.terminal_rewards[map_id, map_robot_ids] = 0.0
            self.collisions[map_id, map_robot_ids * self.robot_num_bodies] = 0
            
        self.curr_goal_positions[maps_ids, robot_ids, :] = self.goal_positions[maps_ids, robot_ids, 0, :]

        if self.imu_params.use_imu:
            self.imu_sensor.reset_idx(env_ids)

        if self.cfg.control.use_reference_model:
            self.reference.reset_reference(env_ids, drone_eulers, torch.zeros_like(drone_eulers), drone_positions, 
                                           drone_velocities, torch.zeros_like(drone_positions))
            self.controller.reset(drone_eulers, env_ids)

        if self.cfg.control.randomize_params:
            self.reset_controllers(env_ids)
        return

    def reset_obstacles(self, if_easy_start=False):
        asset_pose_tensor = self.env_asset_manager.poission_sample_obstacles(if_easy_start=if_easy_start)
        # print("asset_pose_tensor: ", asset_pose_tensor)
        if self.env_asset_root_states is not None:
            self.env_asset_root_states[:, :, 0:3] = asset_pose_tensor[:, :, 0:3]
            euler_angles = asset_pose_tensor[:, :, 3:6]
            self.env_asset_root_states[:, :, 3:7] = quat_from_euler_xyz(euler_angles[..., 0], euler_angles[..., 1], euler_angles[..., 2])
            self.env_asset_root_states[:, :, 7:13] = 0.0

    def init_obstacles(self):
        self.env_asset_manager.randomize_pose()
        if self.env_asset_root_states is not None:
            self.env_asset_root_states[:, :, 0:3] = self.env_asset_manager.asset_pose_tensor[:, 0:3]
            euler_angles = self.env_asset_manager.asset_pose_tensor[:, 3:6]
            self.env_asset_root_states[:, :, 3:7] = quat_from_euler_xyz(euler_angles[..., 0], euler_angles[..., 1], euler_angles[..., 2])
            self.env_asset_root_states[:, :, 7:13] = 0.0
        return
    
    def reset_controllers(self, env_ids):
        self.controller.randomize_params(env_ids)

    def physics_renders(self):
        self.gym.simulate(self.sim)
        self.gym.fetch_results(self.sim, True) # use only when device is not "cpu"
        self.post_physics_step()
        self.check_collisions()

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
    
    def pcd_load_step(self):
        sampling_dis_per_step = self.cfg.logging.sampling_step
        lower_bound = self.cfg.logging.lower_bound
        upper_bound = self.cfg.logging.upper_bound
        # sampling_dis_per_step_th = sampling_dis_per_step * torch.ones(self.num_maps, self.num_robots_per_map, 3, device=self.device, requires_grad=False)
        # sampling_steps = torch.floor((self.env_upper_bound - self.env_lower_bound) / sampling_dis_per_step)
        # print("sampling_steps: ", sampling_steps.shape)
        # print("sampling_dis_per_step_th: ", sampling_dis_per_step_th.shape)
        x_range = torch.arange(
            lower_bound[0],
            upper_bound[0],
            sampling_dis_per_step[0],
            device=self.device
        )
        y_range = torch.arange(
            lower_bound[1],
            upper_bound[1],
            sampling_dis_per_step[1],
            device=self.device
        )
        z_range = torch.arange(
            lower_bound[2],
            upper_bound[2],
            sampling_dis_per_step[2],
            device=self.device
        )
        step_count = 0
        for x in x_range:
            for y in y_range:
                for z in z_range:
                    self.root_positions[:, 0, :] = torch.tensor([x, y, z], device=self.device)
                    self.root_quats[:, 0, :] = torch.tensor([0, 0, 0, 1], device=self.device)
                    self.root_linvels[:, 0, :] = torch.tensor([0, 0, 0], device=self.device)
                    self.root_angvels[:, 0, :] = torch.tensor([0, 0, 0], device=self.device)
                    self.gym.set_actor_root_state_tensor(self.sim, self.root_tensor)
                    self.physics_renders()
                    self.render(sync_frame_time=True)
                    self.render_pcd_cameras(step_count)
                    step_count += 1
        # torch.save(self.log_data_dict["pcd"], './log_data.pth')
        return True

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

        if self.enable_pc_loader:
            self.traj_pos = []
            self.traj_rot = []
            self.traj_vel = []
            self.traj_angvel = []
            self.traj_action = []

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
                
        finish_curr_goal = torch.where(torch.norm(self.curr_goal_positions - self.root_positions, dim=2) < self.goal_threshold, self.ones_int, self.zeros_int)
        self.curr_goal_index += finish_curr_goal
        self.curr_goal_positions = torch.gather(self.goal_positions, 2, self.curr_goal_index.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, 1, 3)).squeeze(2)
        self.successes[:] = torch.where(self.curr_goal_index >= self.cfg.env.goal_num_per_episode, self.ones, self.zeros)
        # print("self.curr_goal_index: ", self.curr_goal_index)
        # print("self.successes: ", self.successes)
        _out_of_bounds = torch.where((self.root_positions < self.flight_lower_bound) | (self.root_positions > self.flight_upper_bound), self.ones_3d, self.zeros_3d)
        self.out_of_bounds = torch.where(torch.sum(_out_of_bounds, dim=2) > 0, self.ones, self.zeros)
            
        self.compute_resets()
        reset_env_ids = self.reset_buf.nonzero(as_tuple=False).squeeze(-1)
        _terminal_rewards = self.terminal_rewards.view(-1).clone().detach()
        # if not self.save_control_data:
        self.reset_idx(reset_env_ids, if_reset_obstacles=False)
        self.compute_vehicle_frame_states()
        self.get_obs()
        self.rew_buf[:, -1] = torch.where(self.reset_buf>0, _terminal_rewards, self.rew_buf[:, -1])
        self.rew_buf[:, -1] = torch.where(finish_curr_goal.view(-1) > 0, self.cfg.RLParamsCfg.r_arrive, self.rew_buf[:, -1])

        if self.enable_pc_loader:
            self.log_data_dict['pos'].append(torch.stack(self.traj_pos))
            self.log_data_dict['rot'].append(torch.stack(self.traj_rot))
            self.log_data_dict['linvel'].append(torch.stack(self.traj_vel))
            self.log_data_dict['angvel'].append(torch.stack(self.traj_angvel))
            self.log_data_dict['action'].append(torch.stack(self.traj_action))
            self.log_data_dict['episode_id'].append(self.available_total_trials.cpu())
            self.log_data_dict['env_step'].append(self.progress_buf.cpu())
            obs_image = torch.clamp(self.full_camera_array, self.sensor_params.min_range, self.sensor_params.max_range)
            self.log_data_dict['main_depth'].append(obs_image.cpu())
            self.log_data_dict['is_finished'].append(self.successes.cpu().reshape(self.num_envs,))
            robot_body_ids = torch.arange(self.num_robots_per_map, device=self.device)*self.robot_num_bodies
            collision_robots = self.collisions[:, robot_body_ids].view(self.num_envs,)
            self.log_data_dict['is_crashed'].append(collision_robots.cpu())
            self.log_data_dict['is_out_of_bounds'].append(self.out_of_bounds.cpu().view(self.num_envs,))
            
        self.progress_buf += 1

        self.prev_action_input[:] = self.action_input.view(self.num_maps, self.num_robots_per_map, -1)
        return self.obs_dict, self.privileged_obs_buf, self.rew_buf, self.reset_buf, self.extras
    
    def cal_success_rate(self):
        return torch.sum(self.success_trials)/torch.sum(self.available_total_trials)
    
    def get_trial_num(self):
        return self.available_total_trials.cpu().numpy()
    
    def get_obs(self):
        self.state_observations = self.compute_state_observations()
        obs_image = torch.clamp(self.full_camera_array, self.sensor_params.min_range, self.sensor_params.max_range)
        obs_image = (obs_image - self.sensor_params.min_range) / (self.sensor_params.max_range - self.sensor_params.min_range) * 255.0

        self.obs_dict['image'] = obs_image.unsqueeze(1).to(torch.uint8)
        self.obs_dict['state'] = self.state_observations
        
        if self.cfg.camera_params.stereo_ground_truth:
            stereo_ground_truth_image = torch.clamp(self.stereo_ground_camera_array, self.sensor_params.min_range, self.sensor_params.max_range)
            stereo_ground_truth_image = (stereo_ground_truth_image - self.sensor_params.min_range) / (self.sensor_params.max_range - self.sensor_params.min_range) * 255.0
            self.obs_dict['stereo_ground_truth'] = stereo_ground_truth_image.unsqueeze(1).to(torch.uint8)

    def compute_state_observations(self):
        goal_dir = self.curr_goal_positions - self.root_positions
        log_distance = torch.log(torch.norm(goal_dir[:, :, :2], dim=2) + 1.0)
        horizon_vel_ = torch.norm(self.linvels_body_frame[:, :, :2], dim=2)
        # chi = torch.atan2(self.linvels_body_frame[:, :, 1], self.linvels_body_frame[:, :, 0])
        perception = torch.abs(self.linvels_body_frame[:, :, 1]) + torch.where(self.linvels_body_frame[:, :, 0] < 0, -self.linvels_body_frame[:, :, 0], self.zeros)
        # beta = torch.atan2(goal_dir[:, :, 1], goal_dir[:, :, 0])
        normalized_goal_dir = goal_dir / (torch.norm(goal_dir, dim=2).unsqueeze(-1) + 1e-6)
        normalized_vel_dir = self.root_linvels_obs / (torch.norm(self.root_linvels_obs, dim=2).unsqueeze(-1) + 1e-6)
        
        goal_penalty = self.cfg.RLParamsCfg.distance_coeff * log_distance
        speed_penalty = torch.where(horizon_vel_ > self.cfg.RLParamsCfg.start_penalty_vel, self.cfg.RLParamsCfg.vel_coeff * horizon_vel_, self.zeros)
        vertical_penalty = self.cfg.RLParamsCfg.vert_coeff * (torch.abs(goal_dir[:, :, 2]) + 0.8*torch.abs(self.linvels_vehicle_frame[:, :, 2]))
        # vertical_penalty = 0.5 * self.cfg.RLParamsCfg.vert_coeff * torch.abs(self.linvels_body_frame[:, :, 2])
        # angular_penalty = self.cfg.RLParamsCfg.angular_vel_coeff * torch.abs(chi + self.vehicle_frame_euler_angles[:, :, 2] - beta)
        angular_penalty = self.cfg.RLParamsCfg.angular_vel_coeff * (torch.abs(normalized_goal_dir[:, :, 0] - normalized_vel_dir[:, :, 0]) + 
                                                                    torch.abs(normalized_goal_dir[:, :, 1] - normalized_vel_dir[:, :, 1]) +
                                                                    torch.abs(normalized_goal_dir[:, :, 2] - normalized_vel_dir[:, :, 2]))
        input_penalty = self.cfg.RLParamsCfg.input_coeff * torch.norm(self.angvels_body_frame, dim=2)
        # yaw_penalty = self.cfg.RLParamsCfg.yaw_coeff * torch.abs(chi)
        yaw_penalty = self.cfg.RLParamsCfg.yaw_coeff * perception
        total_penalty = goal_penalty + speed_penalty + vertical_penalty + angular_penalty + input_penalty + yaw_penalty
        self.rew_buf = torch.stack([goal_penalty, speed_penalty, vertical_penalty, angular_penalty, input_penalty, yaw_penalty, total_penalty], dim=2).view(self.num_envs, -1)
        return torch.stack([log_distance, self.root_positions[:, :, 2], normalized_goal_dir[:, :, 0], normalized_goal_dir[:, :, 1], normalized_goal_dir[:, :, 2], 
                            self.root_linvels_obs[:, :, 0], self.root_linvels_obs[:, :, 1], self.root_linvels_obs[:, :, 2],
                            self.angvels_body_frame[:, :, 0], self.angvels_body_frame[:, :, 1], self.angvels_body_frame[:, :, 2],
                            self.root_euler_angles[:, :, 0], self.root_euler_angles[:, :, 1], self.root_euler_angles[:, :, 2]], dim=2).view(self.num_envs, 1, -1)

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

    def compute_resets(self):
        self.reset_buf[:] = 0
        # terminate for timeout
        self.reset_buf[self.progress_buf >= self.cfg.env.max_episode_length] = 1
        self.terminal_rewards[self.progress_buf.view(self.num_maps, self.num_robots_per_map,) >= self.cfg.env.max_episode_length] = self.cfg.RLParamsCfg.r_timeout
        # self.total_trials[self.progress_buf >= self.cfg.env.max_episode_length] += 1
        # terminate for collision
        robot_body_ids = torch.arange(self.num_robots_per_map, device=self.device)*self.robot_num_bodies
        collision_robots = self.collisions[:, robot_body_ids].view(self.num_envs,)
        self.reset_buf[collision_robots > 0] = 1
        self.terminal_rewards[collision_robots.view(self.num_maps, self.num_robots_per_map,) > 0] = self.cfg.RLParamsCfg.r_collision
        # self.total_trials[collision_robots > 0] += 1
        # terminate for reaching goal
        self.reset_buf[self.successes.view(self.num_envs,) > 0] = 1
        self.terminal_rewards[self.successes > 0] = self.cfg.RLParamsCfg.r_arrive
        # self.total_trials[self.successes.view(self.num_envs,) > 0] += 1
        self.success_trials[self.successes.view(self.num_envs,) > 0] += 1
        # terminate for out of bounds
        self.reset_buf[self.out_of_bounds.view(self.num_envs,) > 0] = 1
        self.terminal_rewards[self.out_of_bounds > 0] = self.cfg.RLParamsCfg.r_exceed
        # self.total_trials[self.out_of_bounds.view(self.num_envs,) > 0] += 1
        reset_ids = ((self.progress_buf >= self.cfg.env.max_episode_length) | 
                                                     (collision_robots > 0) | 
                                  (self.successes.view(self.num_envs,) > 0) | 
                                  (self.out_of_bounds.view(self.num_envs,) > 0))
        countable_reset_ids = reset_ids & (self.progress_buf > 1)
        self.total_trials[reset_ids] += 1
        self.available_total_trials[countable_reset_ids] += 1
        return

    def render_cameras(self):
        if self.enable_isaacgym_cameras:
            self.render_isaacgym_cameras()
        else:
            return
    
    def render_warp_cameras(self):
        self.warp_sensor.capture()
        return
    
    def render_isaacgym_cameras(self):
        self.gym.render_all_camera_sensors(self.sim)
        self.gym.start_access_image_tensors(self.sim)
        self.dump_images()
        self.gym.end_access_image_tensors(self.sim)
        return
    
    def render_pcd_cameras(self, step_count):
        self.gym.render_all_camera_sensors(self.sim)
        self.gym.start_access_image_tensors(self.sim)
        pcd_depth = self.pcd_cameras.get_pcd_camera_data()
        self.gym.end_access_image_tensors(self.sim)
        # save pcd depth to pth
        self.log_data_dict["pcd"]['pos'].append(self.root_positions.cpu())
        self.log_data_dict["pcd"]['quat'].append(self.root_quats.cpu())
        self.log_data_dict["pcd"]['depth'].append(pcd_depth.cpu())
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
                if self.cfg.camera_params.stereo_ground_truth:
                    self.stereo_ground_camera_array[env_id] = -self.stereo_ground_truth_tensors[env_id]

    def action_mixer(self, actions):
        if self.cfg.control.mix_actions:
            self.action_input[:, 0] = self.max_speed * (actions[:, 0]+1)*torch.cos(self.max_inclination_angle*actions[:, 1])/2.0
            self.action_input[:, 1] = 0
            self.action_input[:, 2] = self.max_speed * (actions[:, 0]+1)*torch.sin(self.max_inclination_angle*actions[:, 1])/2.0
            self.action_input[:, 3] = self.max_yawrate*actions[:, 2]
            if torch.any(self.action_input[:, 0] < 0):
                print("Negative vx: ", torch.count(self.action_input[:, 0] < 0).item())
        else:
            self.action_input[:] = self.max_speed*actions
    
    def action_acc_bounding(self, actions):
        self.action_input[:] = actions * self.std_acc + self.mean_acc

    def pre_physics_step(self, _actions, duration):
        actions = _actions.to(self.device)
        actions = torch.clamp(actions, -1.0, 1.0)
        # actions[:] = self.apply_action_noise(actions)
        if self.cfg.control.controller == 'lee_acceleration_control':
            self.action_acc_bounding(actions)
        elif self.cfg.control.controller == 'lee_velocity_control':
            self.action_mixer(actions)
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
        
        if self.enable_pc_loader:
            self.traj_pos.append(self.root_positions.cpu())
            self.traj_rot.append(self.root_euler_angles.cpu())
            self.traj_vel.append(self.root_linvels_obs.cpu())
            self.traj_angvel.append(self.angvels_body_frame.cpu())
            self.traj_action.append(self.action_input.cpu().reshape(self.num_maps, self.num_robots_per_map, -1))

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
        self.forces[:, robot_body_ids, 2] = self.robot_mass * thrust_command.reshape(self.num_maps, self.num_robots_per_map)

        # print("Thrusts: ", self.forces[0, robot_body_ids, 2])
        self.torques[:, robot_body_ids] = output_torques_inertia_normalized.reshape(self.num_maps, self.num_robots_per_map, 3)
        # negative thrusts should be clipped to 0
        self.forces = torch.where(self.forces < 0, torch.zeros_like(self.forces), self.forces)

        ## TODO: @mihirk284 perturbations are only being added to the thrust. There are no perturbations on the lateral forces. 
        # Need to add perturbations to the lateral forces as well.
        self.sample_noise_per_physics_step(self.forces[:, robot_body_ids], self.torques[:, robot_body_ids])
        # print("Extra force disturbance: ", self.extra_force_disturbance)
        # print("force_noise disturbance: ", self.force_noise)
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
        # forces = torch.cat([self.forces.view(self.num_envs*self.robot_num_bodies, 3), self.assets_forces], dim=0)
        # torques = torch.cat([self.torques.view(self.num_envs*self.robot_num_bodies, 3), self.assets_torques], dim=0)
        # apply actions
        self.gym.apply_rigid_body_force_tensors(self.sim, gymtorch.unwrap_tensor(self.forces), gymtorch.unwrap_tensor(self.torques), gymapi.LOCAL_SPACE)

    def sample_noise_per_physics_step(self, forces, torques):
        # additive noise for force and torque inputs with std devs scaled by magnitudes
        self.force_noise = self.env_force_sampler.sample(self.zeros_3d, 0.01*forces)
        self.torque_noise = self.env_torque_sampler.sample(self.zeros_3d, 0.01*torques)

        # additive noise for extra lateral disturbances
        self.extra_force_disturbance = self.env_force_sampler.sample(self.zeros_3d, 0.01*torch.ones_like(forces))
        self.extra_torque_disturbance = self.env_torque_sampler.sample(self.zeros_3d, 0.004*torch.ones_like(torques))

    def post_physics_step(self):
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_net_contact_force_tensor(self.sim)
        if self.imu_params.use_imu:
            self.gym.refresh_force_sensor_tensor(self.sim)

    def check_collisions(self):
        self.collisions[:] = torch.where(torch.norm(self.contact_forces, dim=2) > 0.1, self.ones_bodied, self.collisions)
        # f = self.contact_forces.nonzero(as_tuple=False)
        # if f.shape[0] > 0:
        #     print("Contact forces: ", f)

    def reset_reward_coeffs(self):
        # self.cfg.RLParamsCfg.distance_coeff = 1.0
        # self.cfg.RLParamsCfg.vel_coeff = 0.1
        # self.cfg.RLParamsCfg.vert_coeff = 0.1
        self.cfg.RLParamsCfg.angular_vel_coeff = 0.0
        # self.cfg.RLParamsCfg.input_coeff = 0.1
        self.cfg.RLParamsCfg.yaw_coeff = 0.0

    def getRewardNames(self):
        return self.cfg.RLParamsCfg.names
    
    def getStates(self):
        return self.root_positions.clone().detach().reshape(self.num_envs, 3), self.root_quats.clone().detach().reshape(self.num_envs, 4), self.root_linvels.clone().detach().reshape(self.num_envs, 3), self.root_angvels.clone().detach().reshape(self.num_envs, 3)

    def save_log_data(self, exp_dir):
        torch.save(self.log_data_dict, exp_dir + "/log_data.pth")
        #convert config to dict and save
        cfg_dict = class_to_dict(self.cfg)
        torch.save(cfg_dict, exp_dir + "/config.pt")
        return

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
    
    def _init_log_data_dict(self):
        data_keys = [
            # pre physics
            "env_step",
            "episode_id",
            "main_depth",
            "action",
            "pos",
            "rot",
            "linvel",
            "angvel",
            "is_finished",
            "is_crashed",
            "is_out_of_bounds",
        ]
        pcd_keys = [
            "depth",
            "pos",
            "quat",
        ]
        self.log_data_dict = {
            **{key: [] for key in data_keys},
        }
        self.log_data_dict["pcd"] = {key: [] for key in pcd_keys}

    def set_seed(self, seed):
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
        random.seed(seed)

    def reset_goal_threshold(self, threshold):
        self.goal_threshold = threshold
    
    @property
    def observation_space(self):
        return self.obs_space
    @property
    def action_space(self):
        return self.act_space

@torch.jit.script 
def compute_quadcopter_reward():
    reward = 0.0
    return reward

@torch.jit.script
def torch_rand_float_tensor(lower, upper):
    # type: (torch.Tensor, torch.Tensor) -> torch.Tensor
    return (upper - lower) * torch.rand_like(upper) + lower

@torch.jit.script
def ssa(a: torch.Tensor) -> torch.Tensor:
    '''Smallest signed angle'''
    return torch.remainder(a+np.pi,2*np.pi) - np.pi


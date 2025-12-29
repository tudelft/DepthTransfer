from .base_config import BaseConfig

import numpy as np
from aerial_gym import AERIAL_GYM_ROOT_DIR

from dataclasses import dataclass
import os 

THIN_SEMANTIC_ID = 1
TREE_SEMANTIC_ID = 2
OBJECT_SEMANTIC_ID = 3
CYBERZOO_SEMANTIC_ID = 4
FAKE_TREE_SEMANTIC_ID = 5
WALL_SEMANTIC_ID = 8

PI = np.pi

class ZooTaskCfg(BaseConfig):
    seed = 12
    class env:
        num_envs = 128
        num_maps = 1
        num_observations = 0  # 3 vehicle_frame_vel, 1 roll, 1 pitch, 3 angular_vels, 3 unit_direction_vec_to_goal, 1 distance_to_goal = 3 + 1 + 1 + 3 + 3 + 1 = 12
        get_privileged_obs = False # if True the states of all entitites in the environment will be returned as privileged observations, otherwise None will be returned
        num_actions = 4 # speed, inclination, yaw_rate
        env_spacing = 30.0  # not used with heightfields/trimeshes
        num_control_steps_per_env_step = 10 # number of control & physics steps between camera renders
        enable_isaacgym_cameras = True # enable onboard cameras
        enable_pc_loader = False # enable point cloud loader
        manual_camera_trigger = False # trigger camera captures manually
        reset_on_collision = True # reset environment when contact force on quadrotor is above a threshold
        create_ground_plane = False # create a ground plane
        use_warp_rendering = False # use warp rendering
        # sample_timestep_for_latency = True # sample the timestep for the latency noise
        debug = False # enable debug mode
        logging_sanity_check = False # enable logging sanity check
        log_file_name = "old_setup_but_with_body_frames_new_sf_config_gamma_0985_try2_rollout32_action_penalties_sampled_latent.txt" # log file name
        log_runs = False # log runs
        max_inclination_angle = PI / 3.0 # max inclination angle
        max_yawrate = PI / 3.0 # max yaw rate
        max_speed = 3.5 # max speed
        max_episode_length = 500
        poisson_radius_origin = 1.5
        poisson_radius_area = 0.5
        poisson_radius_easy_start = 8.0
        flight_lower_bound = [0.0, -7.5, 0.1]
        flight_upper_bound = [15.0, 7.5, 1.6]
        act_max = [0.6, 0.6, 0.2, 1.2]
        act_min = [-0.6, -0.6, -0.2, -1.2]
        goal_num_per_episode = 1
        create_texture = False
        goal_arrive_threshold = 0.5

    class LatentSpaceCfg(BaseConfig):
        vae_dims = 70
        lstm_output_dims = 256
        state_dims = 14
        imput_image_size = [224, 320]
        normalize_obs = False
        use_resnet_vae = True
        use_min_pooling = True
        use_kl_latent_loss = False

    class FeasibilityCfg(BaseConfig):
        stabilization_attitude_ref_omega = [5.5, 5.5, 2.5]
        stabilization_attitude_ref_zeta = [3.8, 3.8, 3.2]
        stabilization_attitude_ref_max = [5.2, 5.2, 3.1]
        stabilization_attitude_ref_max_omege_dot = [5.2, 5.2, 2.8]
        stabilization_thrust_rate = 10.0

    class RLParamsCfg(BaseConfig):
        distance_coeff = -0.003
        vel_coeff = -0.01
        vert_coeff = -0.04
        angular_vel_coeff = -0.005
        input_coeff = -0.001
        yaw_coeff = -0.02
        r_exceed = -8.0
        r_arrive = 8.0
        r_collision = -8.0
        r_timeout = -8.0
        start_penalty_vel = 4.0
        names = ["goal_penalty",
                "speed_penalty",
                "vertical_penalty",
                "angular_penalty",
                "input_penalty",
                "yaw_penalty",
                "total"]

    
    class imu_config:
        use_imu = True
        debug = True
        world_frame = False
        # enable or disable noise and bias. Setting to False will simulate a perfect, noise- and bias-free IMU
        enable_noise = True
        enable_bias = True
        bias_std = [9.782812831313576e-07, 9.782812831313576e-07, 9.782812831313576e-07, 2.6541629581345176e-05, 2.6541629581345176e-05, 2.6541629581345176e-05] # first 3 values for acc bias std, next 3 for gyro bias std
        imu_noise_std = [0.001688956233495657, 0.001688956233495657, 0.001688956233495657, 0.0010679343003532472, 0.0010679343003532472, 0.0010679343003532472] # first 3 vaues for acc noise std, next 3 for gyro noise std
        max_measurement_value = [100.0, 100.0, 100.0, 10.0, 10.0, 10.0] # max measurement value for acc and gyro outputs will be clamped by + & - of these 
        
        max_bias_init_value = [1.0e-03, 1.0e-03, 1.0e-03, 1.0e-03, 1.0e-03, 1.0e-03]  # max bias init value for acc and gyro biases will be sampled within +/- of this range
        gravity_compensation = False # usually the force sensor computes total force including gravity, so set this to False

        # Whatever the value of the position is here, the IMU simulator will return
        # the values of acceleration experienced as if the sensor was placed at (0, 0, 0) translation offset from parent_link.
        pos = [0.0, 0.0, 0.0]
        
        
        # We support perturbing the orientation of the sensor
        orientation_euler_deg = [0.0, 0.0, 0.0] # Nominal orientation w.r.t base_link in degrees
        # The sensor is placed at the given orientation but the measurements are perturbed to simulate the discrepancy in sensor placement
        orientation_perturb_amplitude_deg = [2.0, 2.0, 2.0]

    class sensor_config:
        sensor_type = "camera" # lidar or camera
        num_sensors = 1
        width = 512
        height = 128
        horizontal_fov_deg = 360.0
        vertical_fov_deg = 90.0
        
        min_range = 0.1
        max_range = 50.0
        normalize_range = True # divide by max_range.
        far_out_of_range_value = -1.0 * (1 - int(normalize_range)) + (max_range*int(normalize_range)) # Will be [-1]U(0,1] if normalize_range is True, otherwise will be value set by user in place of -1.0
        near_out_of_range_value = -1.0 * (1 - int(normalize_range)) + (-max_range*int(normalize_range)) # Will be [-1]U(0,1] if normalize_range is True, otherwise will be value set by user in place of -1.0
        
        pixel_dropout_prob = 0.000
        pixel_std_dev_multiplier = 0.008
        ray_dir_noise_magnitude = [0.01, 0.01, 0.01] # unit direction vector with this much perturbation in metres from the ideal value
        segmentation_camera = False
        return_pointcloud = False

        euler_frame_rot = [0.0, 0.0, 0.0]
        
        save_sensor_image_to_disk = False
        save_sensor_image_interval = 5

        randomize_placement = False
        min_translation = [0.07, -0.06, 0.01]
        max_translation = [0.12, 0.03, 0.04]
        min_euler_rotation_deg = [-5.0, -5.0, -5.0]
        max_euler_rotation_deg = [5.0, 5.0, 5.0]

    class lidar_params(sensor_config):
        height = 64
        width = 360
        horizontal_fov_deg = 360
        vertical_fov_deg = 90
        num_cameras = 1
        depth_camera=False
        return_pointcloud = False
        pointcloud_in_world_frame = False
        segmentation_camera = True

        max_range = 15.0
        min_range = 0.2
        normalize_range = True # will be set to false when pointcloud is in world frame
        
        # do not change this.
        normalize_range = False if (return_pointcloud==True and pointcloud_in_world_frame==True) else normalize_range  # divide by max_range. Ignored when pointcloud is in world frame
        
        
        far_out_of_range_value = max_range if normalize_range==True else -1.0 # Will be [-1]U[0,1] if normalize_range is True, otherwise will be value set by user in place of -1.0
        near_out_of_range_value = -max_range if normalize_range==True else -1.0 # Will be [-1]U[0,1] if normalize_range is True, otherwise will be value set by user in place of -1.0
        
        euler_frame_rot = [0.0, 0.0, 0.0]
        
        randomize_placement = True
        min_translation = [0.07, -0.06, 0.01]
        max_translation = [0.12, 0.03, 0.04]
        min_euler_rotation_deg = [-5.0, -5.0, -5.0]
        max_euler_rotation_deg = [5.0, 5.0, 5.0]
        
    class camera_params(sensor_config):
        height = 224 # 270
        width = 320 # 480
        horizontal_fov_deg = 86.0
        # height = 240 
        # width = 320 
        # horizontal_fov_deg = 86.0
        num_cameras = 1
        use_stereo_vision = False
        stereo_ground_truth = False
        baseline = 0.12
        calculate_depth = True # get a depth image and not a range image
        return_pointcloud = False
        pointcloud_in_world_frame = False
        segmentation_camera = False

        max_range = 5.0
        min_range = 0.1

        normalize_range = True # will be set to false when pointcloud is in world frame

        # do not change this.
        normalize_range = False if (return_pointcloud==True and pointcloud_in_world_frame==True) else normalize_range  # divide by max_range. Ignored when pointcloud is in world frame
        
        far_out_of_range_value = max_range if normalize_range==True else -1.0 # Will be [-1]U[0,1] if normalize_range is True, otherwise will be value set by user in place of -1.0
        near_out_of_range_value = -max_range if normalize_range==True else -1.0 # Will be [-1]U[0,1] if normalize_range is True, otherwise will be value set by user in place of -1.0
        
        euler_frame_rot = [-np.pi/2.0, 0.0, -np.pi/2.0]
        
        randomize_placement = False
        min_translation = [0.07, -0.06, 0.01]
        max_translation = [0.12, 0.03, 0.04]
        min_euler_rotation_deg = [-5.0, -5.0, -5.0]
        max_euler_rotation_deg = [5.0, 5.0, 5.0]

    class pcd_camera_params(sensor_config):
        height = 640
        width = 640
        horizontal_fov_deg = 90.0
        max_range = 10.0
        min_range = 0.1

    class sim:
        dt =  0.01
        substeps = 1
        gravity = [0., 0. , -9.81]  # [m/s^2]
        up_axis = 1  # 0 is y, 1 is z

        class physx:
            num_threads = 10
            solver_type = 1  # 0: pgs, 1: tgs
            num_position_iterations = 4
            num_velocity_iterations = 0
            contact_offset = 0.01  # [m]
            rest_offset = 0.0   # [m]
            bounce_threshold_velocity = 0.5 # [m/s]
            max_depenetration_velocity = 1.0
            max_gpu_contact_pairs = 2**23 #2**24 -> needed for 8000 envs and more
            default_buffer_size_multiplier = 5
            contact_collection = 1 # 0: never, 1: last sub-step, 2: all sub-steps (default=2)

    class robot_asset:
        file = "{AERIAL_GYM_ROOT_DIR}/resources/robots/quad/model.urdf"
        name = "aerial_robot"  # actor name
        base_link_name = "base_link"
        disable_gravity = False
        collapse_fixed_joints = True # merge bodies connected by fixed joints.
        fix_base_link = False # fix the base of the robot
        collision_mask = 0 # 1 to disable, 0 to enable...bitwise filter
        replace_cylinder_with_capsule = False # replace collision cylinders with capsules, leads to faster/more stable simulation
        flip_visual_attachments = True # Some .obj meshes must be flipped from y-up to z-up
        density = 0.001
        angular_damping = 0.1
        linear_damping = 0.1
        max_angular_velocity = 100.
        max_linear_velocity = 100.
        armature = 0.001

    class robot_spawning_config:
        offset = [0.6, 0.6, 0.6] # offset from each wall
        min_position_ratio = [0.1, 0.1, 0.1] # min position as a ratio of the bounds after offset
        max_position_ratio = [0.2, 0.9, 0.2] # max position as a ratio of the bounds after offset
        min_euler_angles_absolute = [0.0, 0.0, -np.pi/2.0] # min euler angles
        max_euler_angles_absolute = [0.0, 0.0, np.pi/2.0] # max euler angles
        random_start_yaw = True
    
    class goal_spawning_config:
        offset = [0.6, 0.6, 0.6] # offset from each wall
        min_position_ratio = [0.85, 0.1, 0.1] # min position as a ratio of the bounds after offset
        max_position_ratio = [1.0, 0.9, 0.9] # max position as a ratio of the bounds after offset
        min_euler_angles_absolute = [0.0, 0.0, -np.pi / 2.0] # min euler angles
        max_euler_angles_absolute = [0.0, 0.0, np.pi / 2.0] # max euler angles

    class viewer:
        ref_env = 0
        pos = [-5, -5, 4]  # [m]
        lookat = [0, 0, 0]  # [m]

    class control:
        """
        Control parameters
        controller:
            lee_position_control: command_actions = [x, y, z, yaw] in environment frame scaled between -1 and 1
            lee_velocity_control: command_actions = [vx, vy, vz, yaw_rate] in vehicle frame scaled between -1 and 1
            lee_attitude_control: command_actions = [thrust, roll, pitch, yaw_rate] in vehicle frame scaled between -1 and 1
        kP: gains for position
        kV: gains for velocity
        kR: gains for attitude
        kOmega: gains for angular velocity
        """
        mix_actions = False # Modifies the actions to be within the sensor frustum
        controller = "lee_acceleration_control" # or "lee_velocity_control" or "lee_attitude_control" or "lee_acceleration_control"
        kP = [0.1, 0.1, 0.1] # used for lee_position_control only
        kV = [0.078, 0.09, 0.06] # used for lee_position_control, lee_velocity_control only
        kR = [35.0, 35.0, 20.0] # used for lee_position_control, lee_velocity_control and lee_attitude_control
        kOmega = [12, 12, 10] # used for lee_position_control, lee_velocity_control and lee_attitude_control

        kp_rates = [25.0, 25.0, 15.0]
        # kd_rates = [0.2, 0.2, 0.1]
        # ki_rates = [1.0, 1.0, 0.2]
        kd_rates = [0.5, 0.5, 0.2]
        ki_rates = [1.8, 1.8, 0.8]
        kp_acc = [10.0,  10.0, 10.0]
        kd_acc = [ 6.0,   6.0,  6.0]
        # kp_acc = [1.2, 1.2, 2.0]
        # kd_acc = [1.5, 1.5, 2.4]
        krf_acc = [0.8, 0.8, 0.8]
        p_error_max = [2.0, 2.0, 2.0]
        v_error_max = [2.0, 2.0, 2.0]
        # kD_Omega = [0.2, 0.2, 0.1]
        # kI_Omega = [5.0, 5.0, 1.0]
        # integration_max = [5.0, 5.0, 5.0]
        filter_sampling_frequency = 100.0
        filter_cutoff_frequency = 20.0
        filter_initial_value = 0.0
        iterm_lim = [2.0, 2.0, 2.0]
        scale_input = [1.0, 1.0, 1.0, 1.0] # scale the input to the controller from -1 to 1 for each dimension, scale from -np.pi to np.pi for yaw in the case of position control
        randomize_params = False
        use_reference_model = True
        save_control_data = False
        control_data_path = f"{AERIAL_GYM_ROOT_DIR}/aerial_gym/control_data"

    class logging:
        log_file_name = "log_data"
        config_file_name = "config"
        sampling_step = [2, 2, 2]
        lower_bound = [-0.1, -7.6, 0.0]
        upper_bound = [15.1, 7.6, 4.0]
        trial_nums = 10

    class asset_config:
        folder_path = f"{AERIAL_GYM_ROOT_DIR}/resources/models/environment_assets"
        
        include_asset_type = {
            "panels": False,
            "thin": False,
            "trees": False,
            "objects": False,
            "forest": False,
            "cyberzoo": True,
            "fake_trees": True,
            }
            
        include_env_bound_type = {
            "front_wall": True,
            "left_wall": True,
            "top_wall": False,
            "back_wall": True,
            "right_wall": True,
            "right_wall_flip": False,
            "bottom_wall": True,
            "bottom_wall_forest": False,
            "left_wall_forest": False,
            "right_wall_forest": False,
            "front_wall_forest": False,
            "back_wall_forest": False,
            }

        # env_lower_bound_min = [0.0, -4.0, 0.0] # lower bound for the environment space
        # env_lower_bound_max = [0.0, -4.0, 0.0] # lower bound for the environment space
        # env_upper_bound_min = [8.0, 4.0, 8.0] # upper bound for the environment space
        # env_upper_bound_max = [8.0, 4.0, 8.0] # upper bound for the environment space
        env_lower_bound_min = [0.0, -7.5, 0.0] # lower bound for the environment space
        env_lower_bound_max = [0.0, -7.5, 0.0] # lower bound for the environment space
        env_upper_bound_min = [15.0, 7.5, 16.0] # upper bound for the environment space
        env_upper_bound_max = [15.0, 7.5, 16.0] # upper bound for the environment space
        free_space = 1.8 # free space around the environment bounds

    class asset_state_params(robot_asset):
        num_assets = 1                  # number of assets to include

        min_position_ratio = [0.5, 0.5, 0.5] # min position as a ratio of the bounds
        max_position_ratio = [0.5, 0.5, 0.5] # max position as a ratio of the bounds

        collision_mask = 1

        collapse_fixed_joints = True
        fix_base_link = True
        links_per_asset = 1
        set_whole_body_semantic_mask = False
        set_semantic_mask_per_link = False
        semantic_mask_link_list = [] # For empty list, all links are labeled
        specific_filepath = None # if not None, use this folder instead randomizing
        color = None
        keep_in_env = False

    class panel_asset_params(asset_state_params):
        num_assets = 60

        collision_mask = 1 # objects with the same collision mask will not collide

        min_position_ratio = [0.1, 0.05, 0.05] # max position as a ratio of the bounds
        max_position_ratio = [0.85, 0.95, 0.95] # min position as a ratio of the bounds

        specified_position = [-1000.0, -1000.0, -1000.0] # if > -900, use this value instead of randomizing   the ratios

        min_euler_angles = [0.0, 0.0, -np.pi/3.0] # min euler angles
        max_euler_angles = [0.0, 0.0, np.pi/3.0] # max euler angles

        specified_euler_angle = [-1000.0, -1000.0, -1000.0] # if > -900, use this value instead of randomizing

        keep_in_env = True
                
        collapse_fixed_joints = True
        links_per_asset = 1
        set_whole_body_semantic_mask = True
        semantic_id = -1
        set_semantic_mask_per_link = False
        semantic_mask_link_list = [] ## If nothing is specified, all links are labeled
        color = [170,66,66]

    class cyberzoo_asset_params(asset_state_params):
        num_assets = 40

        collision_mask = 1
        min_position_ratio = [0.1, 0.05, 0.05] # max position as a ratio of the bounds
        max_position_ratio = [0.9, 0.9, 0.9] # min position as a ratio of the bounds
        specified_position = [-1000.0, -1000.0, -1000.0] # if > -900, use this value instead of randomizing the ratios

        min_euler_angles = [-0.5, -0.5, -np.pi] # min euler angles
        max_euler_angles = [0.5, 0.5, np.pi] # max euler angles

        specified_euler_angle = [-1000.0, -1000.0, -1000.0] # if > -900, use this value instead of randomizing

        links_per_asset = 1
        set_whole_body_semantic_mask = False
        set_semantic_mask_per_link = False
        semantic_id = CYBERZOO_SEMANTIC_ID

    class fake_trees_asset_params(asset_state_params):
        num_assets = 10

        collision_mask = 1
        min_position_ratio = [0.1, 0.05, 0.05] # max position as a ratio of the bounds
        max_position_ratio = [0.9, 0.9, 0.9] # min position as a ratio of the bounds
        specified_position = [-1000.0, -1000.0, -1000.0] # if > -900, use this value instead of randomizing the ratios

        min_euler_angles = [0.0, 0.0, -np.pi] # min euler angles
        max_euler_angles = [0.0, 0.0, np.pi] # max euler angles

        specified_euler_angle = [-1000.0, -1000.0, -1000.0] # if > -900, use this value instead of randomizing

        links_per_asset = 1
        set_whole_body_semantic_mask = False
        set_semantic_mask_per_link = False
        semantic_id = FAKE_TREE_SEMANTIC_ID

    class thin_asset_params(asset_state_params):
        num_assets = 15

        collision_mask = 1 # objects with the same collision mask will not collide

        min_position_ratio = [0.1, 0.05, 0.05] # max position as a ratio of the bounds
        max_position_ratio = [0.85, 0.95, 0.95] # min position as a ratio of the bounds

        specified_position = [-1000.0, -1000.0, -1000.0] # if > -900, use this value instead of randomizing   the ratios

        min_euler_angles = [-np.pi, -np.pi, -np.pi] # min euler angles
        max_euler_angles = [np.pi, np.pi, np.pi] # max euler angles

        specified_euler_angle = [-1000.0, -1000.0, -1000.0] # if > -900, use this value instead of randomizing
                
        collapse_fixed_joints = True
        links_per_asset = 1
        set_whole_body_semantic_mask = True
        semantic_id = THIN_SEMANTIC_ID
        set_semantic_mask_per_link = False
        semantic_mask_link_list = [] ## If nothing is specified, all links are labeled
        color = [170,66,66]

    class tree_asset_params(asset_state_params):
        num_assets = 15
        collision_mask = 1 # objects with the same collision mask will not collide

        min_position_ratio = [0.2, 0.05, 0.05] # max position as a ratio of the bounds
        max_position_ratio = [0.9, 0.9, 0.9] # min position as a ratio of the bounds

        specified_position = [-1000.0, -1000.0, -1000.0] # if > -900, use this value instead of randomizing   the ratios

        min_euler_angles = [0, -np.pi/6.0, -np.pi] # min euler angles
        max_euler_angles = [0, np.pi/6.0, np.pi] # max euler angles

        specified_euler_angle = [-1000.0, -1000.0, -1000.0] # if > -900, use this value instead of randomizing

        collapse_fixed_joints = True
        links_per_asset = 1
        set_whole_body_semantic_mask = False
        set_semantic_mask_per_link = True
        semantic_mask_link_list = [] ## If nothing is specified, all links are labeled
        semantic_id = TREE_SEMANTIC_ID
        color = [70,200,100]

    class forest_asset_params(asset_state_params):
        num_assets = 15
        collision_mask = 1

        min_position_ratio = [0.2, 0.05, 0.0] # max position as a ratio of the bounds
        max_position_ratio = [0.9, 0.9, 0.9] # min position as a ratio of the bounds
        specified_position = [-1000.0, -1000.0, -1000.0] # if > -900, use this value instead of randomizing   the ratios
        min_euler_angles = [-np.pi/2.0, 0.0, 0.0] # min euler angles
        max_euler_angles = [-np.pi/2.0, 0.0, 0.0] # max euler angles
        specified_euler_angle = [-1000.0, -1000.0, -1000.0] # if > -900, use this value instead of randomizing
        collapse_fixed_joints = True
        links_per_asset = 1
        set_whole_body_semantic_mask = False
        set_semantic_mask_per_link = True
        semantic_mask_link_list = [] ## If nothing is specified, all links are labeled
        semantic_id = TREE_SEMANTIC_ID
        color = [70,200,100]

    class object_asset_params(asset_state_params):
        num_assets = 50
        
        min_position_ratio = [0.1, 0.05, 0.05] # max position as a ratio of the bounds
        max_position_ratio = [0.9, 0.9, 0.9] # min position as a ratio of the bounds

        specified_position = [-1000.0, -1000.0, -1000.0] # if > -900, use this value instead of randomizing the ratios

        min_euler_angles = [0, 0, -np.pi] # min euler angles
        max_euler_angles = [0, 0, np.pi] # max euler angles

        specified_euler_angle = [-1000.0, -1000.0, -1000.0] # if > -900, use this value instead of randomizing

        links_per_asset = 1
        set_whole_body_semantic_mask = False
        set_semantic_mask_per_link = False
        semantic_id = OBJECT_SEMANTIC_ID

        # color = [80,255,100]

    class left_wall(asset_state_params):
        num_assets = 1

        collision_mask = 1 # objects with the same collision mask will not collide
        
        min_position_ratio = [0.5, 1.0, 0.5] # min position as a ratio of the bounds
        max_position_ratio = [0.5, 1.0, 0.5] # max position as a ratio of the bounds

        specified_position = [-1000.0, -1000.0, -1000.0] # if > -900, use this value instead of randomizing the ratios

        min_euler_angles = [0.0, 0.0, 0.0] # min euler angles
        max_euler_angles = [0.0, 0.0, 0.0] # max euler angles

        specified_euler_angle = [-1000.0, -1000.0, -1000.0] # if > -900, use this value instead of randomizing
                
        collapse_fixed_joints = False
        links_per_asset = 1
        specific_filepath = "cube.urdf"
        semantic_id = WALL_SEMANTIC_ID
        color = [100,200,210]

    class left_wall_forest(asset_state_params):
        num_assets = 1

        collision_mask = 1 # objects with the same collision mask will not collide
        
        min_position_ratio = [0.5, 1.02, 0.5] # min position as a ratio of the bounds
        max_position_ratio = [0.5, 1.02, 0.5] # max position as a ratio of the bounds

        specified_position = [-1000.0, -1000.0, -1000.0] # if > -900, use this value instead of randomizing the ratios

        min_euler_angles = [0.0, 0.0, 0.0] # min euler angles
        max_euler_angles = [0.0, 0.0, 0.0] # max euler angles

        specified_euler_angle = [-1000.0, -1000.0, -1000.0] # if > -900, use this value instead of randomizing
                
        collapse_fixed_joints = False
        links_per_asset = 1
        specific_filepath = "cube.urdf"
        semantic_id = WALL_SEMANTIC_ID
        color = [100,200,210]
    
    class right_wall(asset_state_params):
        num_assets = 1

        collision_mask = 1 # objects with the same collision mask will not collide

        min_position_ratio = [0.5, 0.0, 0.5] # min position as a ratio of the bounds
        max_position_ratio = [0.5, 0.0, 0.5] # max position as a ratio of the bounds

        specified_position = [-1000.0, -1000.0, -1000.0] # if > -900, use this value instead of randomizing the ratios

        min_euler_angles = [0.0, 0.0, 0.0] # min euler angles
        max_euler_angles = [0.0, 0.0, 0.0] # max euler angles

        specified_euler_angle = [-1000.0, -1000.0, -1000.0] # if > -900, use this value instead of randomizing
        
        collapse_fixed_joints = False
        links_per_asset = 1
        specific_filepath = "cube.urdf"
        semantic_id = WALL_SEMANTIC_ID
        color = [100,200,210]

    class right_wall_forest(asset_state_params):
        num_assets = 1

        collision_mask = 1 # objects with the same collision mask will not collide

        min_position_ratio = [0.5, -0.02, 0.5] # min position as a ratio of the bounds
        max_position_ratio = [0.5, -0.02, 0.5] # max position as a ratio of the bounds

        specified_position = [-1000.0, -1000.0, -1000.0] # if > -900, use this value instead of randomizing the ratios

        min_euler_angles = [0.0, 0.0, 0.0] # min euler angles
        max_euler_angles = [0.0, 0.0, 0.0] # max euler angles

        specified_euler_angle = [-1000.0, -1000.0, -1000.0] # if > -900, use this value instead of randomizing
        
        collapse_fixed_joints = False
        links_per_asset = 1
        specific_filepath = "cube.urdf"
        semantic_id = WALL_SEMANTIC_ID
        color = [100,200,210]
    
    class right_wall_flip(asset_state_params):
        num_assets = 1

        collision_mask = 1 # objects with the same collision mask will not collide

        min_position_ratio = [0.5, 0.0, 0.5] # min position as a ratio of the bounds
        max_position_ratio = [0.5, 0.0, 0.5] # max position as a ratio of the bounds

        specified_position = [-1000.0, -1000.0, -1000.0] # if > -900, use this value instead of randomizing the ratios

        min_euler_angles = [0.0, 0.0, 0.0] # min euler angles
        max_euler_angles = [0.0, 0.0, 0.0] # max euler angles

        specified_euler_angle = [-1000.0, -1000.0, -1000.0] # if > -900, use this value instead of randomizing
        
        collapse_fixed_joints = False
        links_per_asset = 1
        specific_filepath = "cube.urdf"
        semantic_id = WALL_SEMANTIC_ID
        color = [100,200,210]
    
    class top_wall(asset_state_params):
        num_assets = 1

        collision_mask = 1 # objects with the same collision mask will not collide

        min_position_ratio = [0.5, 0.5, 1.0] # min position as a ratio of the bounds
        max_position_ratio = [0.5, 0.5, 1.0] # max position as a ratio of the bounds

        specified_position = [-1000.0, -1000.0, -1000.0] # if > -900, use this value instead of randomizing the ratios

        min_euler_angles = [0.0, 0.0, 0.0] # min euler angles
        max_euler_angles = [0.0, 0.0, 0.0] # max euler angles

        specified_euler_angle = [-1000.0, -1000.0, -1000.0] # if > -900, use this value instead of randomizing

        collapse_fixed_joints = False
        links_per_asset = 1
        specific_filepath = "cube.urdf"
        semantic_id = WALL_SEMANTIC_ID
        color = [100,200,210]
    
    class bottom_wall(asset_state_params):
        num_assets = 1

        collision_mask = 1 # objects with the same collision mask will not collide

        min_position_ratio = [0.5, 0.5, 0.0] # min position as a ratio of the bounds
        max_position_ratio = [0.5, 0.5, 0.0] # max position as a ratio of the bounds

        specified_position = [-1000.0, -1000.0, -1000.0] # if > -900, use this value instead of randomizing the ratios

        min_euler_angles = [0.0, 0.0, 0.0] # min euler angles
        max_euler_angles = [0.0, 0.0, 0.0] # max euler angles

        specified_euler_angle = [-1000.0, -1000.0, -1000.0] # if > -900, use this value instead of randomizing

        collapse_fixed_joints = False
        links_per_asset = 1
        specific_filepath = "cube.urdf"
        semantic_id = WALL_SEMANTIC_ID
        color = [100,150,150]
    
    class bottom_wall_forest(asset_state_params):
        num_assets = 1

        collision_mask = 1 # objects with the same collision mask will not collide

        min_position_ratio = [0.5, 0.5, 0.0] # min position as a ratio of the bounds
        max_position_ratio = [0.5, 0.5, 0.0] # max position as a ratio of the bounds

        specified_position = [-1000.0, -1000.0, -1000.0] # if > -900, use this value instead of randomizing the ratios

        min_euler_angles = [0.0, 0.0, 0.0] # min euler angles
        max_euler_angles = [0.0, 0.0, 0.0] # max euler angles

        specified_euler_angle = [-1000.0, -1000.0, -1000.0] # if > -900, use this value instead of randomizing

        collapse_fixed_joints = False
        links_per_asset = 1
        specific_filepath = "cube.urdf"
        semantic_id = WALL_SEMANTIC_ID
        color = [100,150,150]

    class front_wall(asset_state_params):
        num_assets = 1

        collision_mask = 1 # objects with the same collision mask will not collide


        min_position_ratio = [1.0, 0.5, 0.5] # min position as a ratio of the bounds
        max_position_ratio = [1.0, 0.5, 0.5] # max position as a ratio of the bounds

        specified_position = [-1000.0, -1000.0, -1000.0] # if > -900, use this value instead of randomizing the ratios

        min_euler_angles = [0.0, 0.0, 0.0] # min euler angles
        max_euler_angles = [0.0, 0.0, 0.0] # max euler angles

        specified_euler_angle = [-1000.0, -1000.0, -1000.0] # if > -900, use this value instead of randomizing

        collapse_fixed_joints = False
        links_per_asset = 1
        specific_filepath = "cube.urdf"
        semantic_id = WALL_SEMANTIC_ID
        color = [100,200,210]

    class front_wall_forest(asset_state_params):
        num_assets = 1

        collision_mask = 1 # objects with the same collision mask will not collide


        min_position_ratio = [1.02, 0.5, 0.5] # min position as a ratio of the bounds
        max_position_ratio = [1.02, 0.5, 0.5] # max position as a ratio of the bounds

        specified_position = [-1000.0, -1000.0, -1000.0] # if > -900, use this value instead of randomizing the ratios

        min_euler_angles = [0.0, 0.0, 0.0] # min euler angles
        max_euler_angles = [0.0, 0.0, 0.0] # max euler angles

        specified_euler_angle = [-1000.0, -1000.0, -1000.0] # if > -900, use this value instead of randomizing

        collapse_fixed_joints = False
        links_per_asset = 1
        specific_filepath = "cube.urdf"
        semantic_id = WALL_SEMANTIC_ID
        color = [100,200,210]
    
    
    class back_wall(asset_state_params):
        num_assets = 1

        collision_mask = 1 # objects with the same collision mask will not collide
        
        min_position_ratio = [0.0, 0.5, 0.5] # min position as a ratio of the bounds
        max_position_ratio = [0.0, 0.5, 0.5] # max position as a ratio of the bounds

        specified_position = [-1000.0, -1000.0, -1000.0] # if > -900, use this value instead of randomizing the ratios

        min_euler_angles = [0.0, 0.0, 0.0] # min euler angles
        max_euler_angles = [0.0, 0.0, 0.0] # max euler angles

        specified_euler_angle = [-1000.0, -1000.0, -1000.0] # if > -900, use this value instead of randomizing
        
        collapse_fixed_joints = False
        links_per_asset = 1
        specific_filepath = "cube.urdf"
        semantic_id = WALL_SEMANTIC_ID
        color = [100,200,210]
            
    class back_wall_forest(asset_state_params):
        num_assets = 1

        collision_mask = 1 # objects with the same collision mask will not collide
        
        min_position_ratio = [-0.02, 0.5, 0.5] # min position as a ratio of the bounds
        max_position_ratio = [-0.02, 0.5, 0.5] # max position as a ratio of the bounds

        specified_position = [-1000.0, -1000.0, -1000.0] # if > -900, use this value instead of randomizing the ratios

        min_euler_angles = [0.0, 0.0, 0.0] # min euler angles
        max_euler_angles = [0.0, 0.0, 0.0] # max euler angles

        specified_euler_angle = [-1000.0, -1000.0, -1000.0] # if > -900, use this value instead of randomizing
        
        collapse_fixed_joints = False
        links_per_asset = 1
        specific_filepath = "cube.urdf"
        semantic_id = WALL_SEMANTIC_ID
        color = [100,200,210]

    class env_disturbance_application_probability:
        enable = False
        distribution = "bernoulli"
        dist_params = {"probs": 0.12}
        transform_after_sampling = False
        
    class env_force_perturbations_noise:
        enable = True
        distribution = "normal"
        dist_params = {"loc": 0.0, "scale": 1.0}
        transform_after_sampling = True
            
    class env_torque_perturbations_noise:
        enable = True
        distribution = "normal"
        dist_params = {"loc": 0.0, "scale": 1.0}
        transform_after_sampling = True

    class velocity_measurement_noise:
        enable = True
        distribution = "normal"
        dist_params = {"loc": 0.0, "scale": 1.0}
        transform_after_sampling = True

    class angle_measurement_noise:
        enable = True
        distribution = "normal"
        dist_params = {"loc": 0.0, "scale": 1.0}
        transform_after_sampling = True

    class angular_velocity_measurement_noise:
        enable = True
        distribution = "normal"
        dist_params = {"loc": 0.0, "scale": 1.0}
        transform_after_sampling = True
    

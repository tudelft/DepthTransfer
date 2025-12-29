from .base_config import BaseConfig

import numpy as np
from aerial_gym import AERIAL_GYM_ROOT_DIR

from dataclasses import dataclass
import os 

THIN_SEMANTIC_ID = 1
TREE_SEMANTIC_ID = 2
OBJECT_SEMANTIC_ID = 3
CYBERZOO_SEMANTIC_ID = 4
WALL_SEMANTIC_ID = 8

PI = np.pi

class DroneRacingTaskCfg(BaseConfig):
    seed = -1
    class env:
        num_envs = 128
        num_maps = 1
        num_observations = 0  # 3 vehicle_frame_vel, 1 roll, 1 pitch, 3 angular_vels, 3 unit_direction_vec_to_goal, 1 distance_to_goal = 3 + 1 + 1 + 3 + 3 + 1 = 12
        get_privileged_obs = False # if True the states of all entitites in the environment will be returned as privileged observations, otherwise None will be returned
        num_actions = 4
        env_spacing = 40.0
        num_control_steps_per_env_step = 10 # number of control & physics steps between camera renders
        enable_isaacgym_cameras = True # enable onboard cameras
        manual_camera_trigger = False # trigger camera captures manually
        reset_on_collision = True # reset environment when contact force on quadrotor is above a threshold
        create_ground_plane = False # create a ground plane
        use_warp_rendering = False # use warp rendering
        debug = False # enable debug mode
        logging_sanity_check = False # enable logging sanity check
        log_file_name = "old_setup_but_with_body_frames_new_sf_config_gamma_0985_try2_rollout32_action_penalties_sampled_latent.txt" # log file name
        log_runs = False # log runs
        randomize_goal_position = True # randomize goal position
        max_episode_length = 250
        poisson_radius_origin = 1.1
        poisson_radius_area = 1.2
        poisson_radius_easy_start = 4.0
        # flight_lower_bound = [0.0, -4.0, 0.1]
        # flight_upper_bound = [8.0, 4.0, 2.0]
        act_max = [0.6, 0.6, 0.4, 1.2]
        act_min = [-0.6, -0.6, -0.4, -1.2]
        goal_num_per_episode = 2
        disableObstacleManager = False
        enableDebugVis = False

    class envCreator:
        env_size = 40.0
        backstage_z_offset = 20.0
        ground_color = [ 0.25, 0.25, 0.25 ]
        ground_len_z = 0.3
        gate_bar_len_x = [ 0.15 ]  # thick
        gate_bar_len_y = [ 2.0 ]  # length
        gate_bar_len_z = [ 0.225 ]  # wide
        gate_color = [ 1.0, 0.5, 0.3 ]
        disable_tqdm = False
        # random boxes, params: [size_x, size_y, size_z]
        num_box_actors = 5
        num_box_assets = 5
        box_params_min = [ 0.3, 0.3, 0.3 ]
        box_params_max = [ 2.0, 2.0, 2.0 ]
        box_color = [ 0.12156862745098039, 0.4666666666666667, 0.7058823529411765 ]
        # random capsules, params: [radius, length]
        num_capsule_actors = 5
        num_capsule_assets = 5
        capsule_params_min = [ 0.3, 0.3 ]
        capsule_params_max = [ 1.0, 1.0 ]
        capsule_color = [ 0.7294117647058823, 0.21176470588235294, 0.3411764705882353 ]
        # random cuboid wireframes, params: [size_x, size_y, size_z, weight]
        num_cuboid_wireframe_actors = 0
        num_cuboid_wireframe_assets = 0
        cuboid_wireframe_params_min = [ 0.3, 0.3, 0.3, 0.2 ]
        cuboid_wireframe_params_max = [ 2.0, 2.0, 2.0, 0.4 ]
        cuboid_wireframe_color = [ 0.5803921568627451, 0.403921568627451, 0.7411764705882353 ]
        # random cylinders, params: [radius, length]
        num_cylinder_actors = 5
        num_cylinder_assets = 5
        cylinder_params_min = [ 0.1, 0.2 ]
        cylinder_params_max = [ 1.0, 2.0 ]
        cylinder_color = [ 0.5490196078431373, 0.33725490196078434, 0.29411764705882354 ]
        # random hollow cuboids, params: [length_x, inner_length_y, inner_length_z, diff_length_y, diff_length_z]
        num_hollow_cuboid_actors = 5
        num_hollow_cuboid_assets = 5
        hollow_cuboid_params_min = [ 0.10, 0.5, 0.5, 0.2, 0.2 ]
        hollow_cuboid_params_max = [ 0.25, 1.4, 1.4, 0.6, 0.6 ]
        hollow_cuboid_color = [ 0.8901960784313725, 0.4666666666666667, 0.7607843137254902 ]
        # random spheres, params: [radius]
        num_sphere_actors = 5
        num_sphere_assets = 5
        sphere_params_min = [ 0.3 ]
        sphere_params_max = [ 1.0 ]
        sphere_color = [ 0.7372549019607844, 0.7411764705882353, 0.13333333333333333 ]
        # random trees, params: none
        num_tree_actors = 0
        num_tree_assets = 0
        tree_color = [ 0.4196078431372549, 0.5411764705882353, 0.47843137254901963 ]
        # random walls, params: [size_x, size_y, size_z]
        num_wall_actors = 5
        num_wall_assets = 5
        wall_params_min = [ 0.2, 1.5, 1.5 ]
        wall_params_max = [ 0.2, 2.5, 2.5 ]
        wall_color = [ 0.09019607843137255, 0.7450980392156863, 0.8117647058823529 ]

    class LatentSpaceCfg(BaseConfig):
        vae_dims = 64
        lstm_output_dims = 256
        state_dims = 14
        imput_image_size = 256
        normalize_obs = False

    class FeasibilityCfg(BaseConfig):
        stabilization_attitude_ref_omega = [5.5, 5.5, 2.5]
        stabilization_attitude_ref_zeta = [3.8, 3.8, 3.2]
        stabilization_attitude_ref_max = [5.2, 5.2, 3.1]
        stabilization_attitude_ref_max_omege_dot = [5.2, 5.2, 2.8]
        stabilization_thrust_rate = 10.0

    class RLParamsCfg(BaseConfig):
        distance_coeff = -0.0005
        vel_coeff = -0.001
        vert_coeff = -0.001
        angular_vel_coeff = -0.006
        input_coeff = -0.001
        yaw_coeff = -0.02
        r_exceed = -10.0
        r_arrive = 8.0
        r_collision = -10.0
        r_timeout = -10.0
        start_penalty_vel = 8.0
        names = ["goal_penalty",
                "speed_penalty",
                "vertical_penalty",
                "angular_penalty",
                "input_penalty",
                "yaw_penalty",
                "total"]
        
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

    class camera_params(sensor_config):
        height = 256 # 270
        width = 256 # 480
        horizontal_fov_deg = 70.0
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

        max_range = 12.0
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
        kd_acc = [6.0,   6.0,  6.0]
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
        use_reference_model = False
        save_control_data = False
        control_data_path = f"{AERIAL_GYM_ROOT_DIR}/aerial_gym/control_data"

    class way_point_generator:
        num_waypoints = 4
        num_gate_x_lens = 1 # check envCreate.gate_bar_len_x
        num_gate_weights = 1 # check envCreate.gate_bar_len_z
        gate_weight_max = 0.225  # check envCreator.gate_bar_len_z
        fixed_waypoint_id = 1
        fixed_waypoint_position = [ 0.0, 0.0, 20.0 ]

    class initRandOpt:
        class randWaypointOptions:
            wp_size_min = 1.2  # check gate_bar_len_y
            wp_size_max = 2.0
            init_roll_max = 0.5
            init_pitch_max = 0.5
            init_yaw_max = 3.14
            psi_max = 1.57
            theta_max = 0.79
            alpha_max = 3.14
            gamma_max = 0.5
            r_min = 0.0
            r_max = 20.0
            force_gate_flag = 1
            same_track = False
        class randObstacleOptions:
            extra_clearance = 1.42
            orbit_density = 0.05
            tree_density = 0.05
            wall_density = 0.05
            std_dev_scale = 1.0
            gnd_distance_min = 1.0
            gnd_distance_max = 10.0
        class randDroneOptions:
            next_wp_id_max = 1
            dist_along_line_min = 0.0
            dist_along_line_max = 0.1
            drone_rotation_x_max = 3.14
            dist_to_line_max = 1.0
            lin_vel_x_max = 1.0  # m/s
            lin_vel_y_max = 1.0
            lin_vel_z_max = 1.0
            ang_vel_x_max = 0.1  # rad/s
            ang_vel_y_max = 0.1
            ang_vel_z_max = 0.1
            aileron_max = 0.25
            elevator_max = 0.25
            rudder_max = 0.25
            throttle_min = -1.0
            throttle_max = -0.5

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

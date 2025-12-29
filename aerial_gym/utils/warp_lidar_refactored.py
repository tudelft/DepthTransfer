import time

import matplotlib.pyplot as plt
import torch
import numpy as np
import nvtx
import warp as wp

wp.init()
wp.config.fast_math = True


NO_HIT_RAY_VAL = wp.constant(1000.0)
NO_HIT_SEGMENTATION_VAL = wp.constant(wp.int32(-1))

@wp.kernel
def draw_optimized_kernel_pointcloud(mesh_ids: wp.array(dtype=wp.uint64),
        lidar_pos_array: wp.array(dtype=wp.vec3),
        lidar_quat_array: wp.array(dtype=wp.quat),
        ray_vectors: wp.array2d(dtype=wp.vec3),
        # ray_noise_magnitude: wp.array(dtype=float),
        far_plane: float,
        pixels: wp.array(dtype=wp.vec3,ndim=3),
        pointcloud_in_world_frame: bool
        ):
    
    env_id, scan_line, point_index = wp.tid()
    cam_id = env_id # TODO: handle more cams per env
    mesh = mesh_ids[env_id]
    lidar_position = lidar_pos_array[cam_id]
    lidar_quaternion = lidar_quat_array[cam_id]
    ray_origin = lidar_position
    # perturb ray_vectors with uniform noise
    ray_dir = ray_vectors[scan_line, point_index] # + sampled_vec3_noise
    ray_dir = wp.normalize(ray_dir)
    ray_direction_world = wp.normalize(wp.quat_rotate(lidar_quaternion, ray_dir))
    t = float(0.0)
    u = float(0.0)
    v = float(0.0)
    sign = float(0.0)
    n = wp.vec3()
    f = int(0)
    dist = NO_HIT_RAY_VAL
    if wp.mesh_query_ray(mesh, ray_origin, ray_direction_world, far_plane, t, u, v, sign, n, f):
        dist = t
    if pointcloud_in_world_frame:
        pixels[env_id, scan_line, point_index] = ray_origin + dist * ray_direction_world
    else:
        pixels[env_id, scan_line, point_index] = dist * ray_dir


@wp.kernel
def draw_optimized_kernel_pointcloud_segmentation(mesh_ids: wp.array(dtype=wp.uint64),
        lidar_pos_array: wp.array(dtype=wp.vec3),
        lidar_quat_array: wp.array(dtype=wp.quat),
        ray_vectors: wp.array2d(dtype=wp.vec3),
        # ray_noise_magnitude: wp.array(dtype=float),
        far_plane: float,
        pixels: wp.array(dtype=wp.vec3,ndim=3),
        segmentation_pixels: wp.array(dtype=wp.int32, ndim=3),
        pointcloud_in_world_frame: bool
        ):
    
    env_id, scan_line, point_index = wp.tid()
    cam_id = env_id # TODO: handle more cams per env
    mesh = mesh_ids[env_id]
    lidar_position = lidar_pos_array[cam_id]
    lidar_quaternion = lidar_quat_array[cam_id]
    ray_origin = lidar_position
    # perturb ray_vectors with uniform noise
    ray_dir = ray_vectors[scan_line, point_index] # + sampled_vec3_noise
    ray_dir = wp.normalize(ray_dir)
    ray_direction_world = wp.normalize(wp.quat_rotate(lidar_quaternion, ray_dir))
    t = float(0.0)
    u = float(0.0)
    v = float(0.0)
    sign = float(0.0)
    n = wp.vec3()
    f = int(0)
    dist = NO_HIT_RAY_VAL
    if wp.mesh_query_ray(mesh, ray_origin, ray_direction_world, far_plane, t, u, v, sign, n, f):
        dist = t
        mesh_obj = wp.mesh_get(mesh)
        face_index = mesh_obj.indices[f*3]
        segmentation_value = wp.int32(mesh_obj.velocities[face_index][0])
    if pointcloud_in_world_frame:
        pixels[env_id, scan_line, point_index] = ray_origin + dist * ray_direction_world
    else:
        pixels[env_id, scan_line, point_index] = dist * ray_dir
    segmentation_pixels[env_id, scan_line, point_index] = segmentation_value

@wp.kernel
def draw_optimized_kernel_range_segmentation(mesh_ids: wp.array(dtype=wp.uint64),
        lidar_pos_array: wp.array(dtype=wp.vec3),
        lidar_quat_array: wp.array(dtype=wp.quat),
        ray_vectors: wp.array2d(dtype=wp.vec3),
        # ray_noise_magnitude: wp.array(dtype=float),
        far_plane: float,
        pixels: wp.array(dtype=float,ndim=3),
        segmentation_pixels: wp.array(dtype=wp.int32, ndim=3)
        ):
    env_id, scan_line, point_index = wp.tid()
    cam_id = env_id # TODO: handle more cams per env
    mesh = mesh_ids[env_id]
    lidar_position = lidar_pos_array[cam_id]
    lidar_quaternion = lidar_quat_array[cam_id]
    ray_origin = lidar_position
    # perturb ray_vectors with uniform noise
    ray_dir = ray_vectors[scan_line, point_index] # + sampled_vec3_noise
    ray_dir = wp.normalize(ray_dir)
    ray_direction_world = wp.normalize(wp.quat_rotate(lidar_quaternion, ray_dir))
    t = float(0.0)
    u = float(0.0)
    v = float(0.0)
    sign = float(0.0)
    n = wp.vec3()
    f = int(0)
    dist = NO_HIT_RAY_VAL 
    segmentation_value = NO_HIT_SEGMENTATION_VAL
    if wp.mesh_query_ray(mesh, ray_origin, ray_direction_world, far_plane, t, u, v, sign, n, f):
        dist = t
        mesh_obj = wp.mesh_get(mesh)
        face_index = mesh_obj.indices[f*3]
        segmentation_value = wp.int32(mesh_obj.velocities[face_index][0])
    pixels[env_id, scan_line, point_index] = dist
    segmentation_pixels[env_id, scan_line, point_index] = segmentation_value

@wp.kernel
def draw_optimized_kernel_range(mesh_ids: wp.array(dtype=wp.uint64),
        lidar_pos_array: wp.array(dtype=wp.vec3),
        lidar_quat_array: wp.array(dtype=wp.quat),
        ray_vectors: wp.array2d(dtype=wp.vec3),
        # ray_noise_magnitude: wp.array(dtype=float),
        far_plane: float,
        pixels: wp.array(dtype=float,ndim=3)
        ):
    env_id, scan_line, point_index = wp.tid()
    cam_id = env_id # TODO: handle more cams per env
    mesh = mesh_ids[env_id]
    lidar_position = lidar_pos_array[cam_id]
    lidar_quaternion = lidar_quat_array[cam_id]
    ray_origin = lidar_position
    # perturb ray_vectors with uniform noise
    ray_dir = ray_vectors[scan_line, point_index] # + sampled_vec3_noise
    ray_dir = wp.normalize(ray_dir)
    ray_direction_world = wp.normalize(wp.quat_rotate(lidar_quaternion, ray_dir))
    t = float(0.0)
    u = float(0.0)
    v = float(0.0)
    sign = float(0.0)
    n = wp.vec3()
    f = int(0)
    dist = NO_HIT_RAY_VAL 
    if wp.mesh_query_ray(mesh, ray_origin, ray_direction_world, far_plane, t, u, v, sign, n, f):
        dist = t
    pixels[env_id, scan_line, point_index] = dist


class WarpLidarRefactored:
    def __init__(self, num_envs, mesh_id_list, pixels, segmentation_pixels, config, device="cuda:0"):
        self.cfg = config
        self.num_envs = num_envs
        self.num_sensors = self.cfg.num_sensors
        if self.num_sensors > 1:
            raise NotImplementedError("Multiple cameras not supported yet")
        self.num_scan_lines = self.cfg.height
        self.num_points_per_line = self.cfg.width
        self.horizontal_fov = np.radians(self.cfg.horizontal_fov_deg)
        self.vertical_fov = np.radians(self.cfg.vertical_fov_deg)
        if self.vertical_fov > np.pi:
            raise ValueError("Vertical FOV must be less than pi")
        self.far_plane = self.cfg.max_range
        self.device = device
        
        # init buffers. None when uninitialized
        if self.cfg.return_pointcloud:
            self.pixels = wp.from_torch(pixels,dtype=wp.vec3)
            self.pointcloud_in_world_frame = self.cfg.pointcloud_in_world_frame
        else:
            self.pixels = wp.from_torch(pixels,dtype=wp.float32)
        if self.cfg.segmentation_camera == True:
            self.segmentation_pixels = wp.from_torch(segmentation_pixels,dtype=wp.int32)
        
        self.lidar_position_array = None
        self.lidar_quat_array = None
        self.mesh_ids_array = wp.array(mesh_id_list, dtype=wp.uint64)
        self.graph = None

        # populate a 2D torch array with the ray vectors that are 2d arrays of wp.vec3
        ray_vectors = torch.zeros((self.num_scan_lines, self.num_points_per_line, 3), dtype=torch.float32, device=self.device, requires_grad=False)
        for i in range(self.num_scan_lines):
            for j in range(self.num_points_per_line):
                # Check if this makes sense
                ray_vectors[i, j, 0] = np.cos(self.horizontal_fov * (j / self.num_points_per_line - 0.5))
                ray_vectors[i, j, 1] = np.sin(self.horizontal_fov * (j / self.num_points_per_line - 0.5))
                # Different subtraction order here because we want to have top rays at the top of the image
                ray_vectors[i, j, 2] = np.sin(self.vertical_fov * (0.5 - i / self.num_scan_lines))

        # normalize ray_vectors
        ray_vectors = ray_vectors / torch.norm(ray_vectors, dim=2, keepdim=True)

        # recast as 2D warp array of vec3
        self.ray_vectors = wp.from_torch(ray_vectors, dtype=wp.vec3)

    def create_render_graph_pointcloud(self, debug=False):
        if not debug:
            print(f"creating render graph")
            wp.capture_begin(device=self.device)
        if self.cfg.segmentation_camera == True:
            wp.launch(
                kernel=draw_optimized_kernel_pointcloud_segmentation,
                dim=(self.num_envs, self.num_scan_lines, self.num_points_per_line),
                inputs=[self.mesh_ids_array, self.lidar_position_array, self.lidar_quat_array, self.ray_vectors, self.far_plane, self.pixels, self.segmentation_pixels, self.pointcloud_in_world_frame],
                device=self.device)

        else:
            wp.launch(
                kernel=draw_optimized_kernel_pointcloud,
                dim=(self.num_envs, self.num_scan_lines, self.num_points_per_line),
                inputs=[self.mesh_ids_array, self.lidar_position_array, self.lidar_quat_array, self.ray_vectors, self.far_plane, self.pixels, self.pointcloud_in_world_frame],
                device=self.device)
        if not debug:
            self.graph = wp.capture_end(device=self.device)
    
    def create_render_graph_range(self, debug=False):
        if not debug:
            print(f"creating render graph")
            wp.capture_begin(device=self.device)
        if self.cfg.segmentation_camera == True:
            wp.launch(
                kernel=draw_optimized_kernel_range_segmentation,
                dim=(self.num_envs, self.num_scan_lines, self.num_points_per_line),
                inputs=[self.mesh_ids_array, self.lidar_position_array, self.lidar_quat_array, self.ray_vectors, self.far_plane, self.pixels, self.segmentation_pixels],
                device=self.device)
        else:
            wp.launch(
                kernel=draw_optimized_kernel_range,
                dim=(self.num_envs, self.num_scan_lines, self.num_points_per_line),
                inputs=[self.mesh_ids_array, self.lidar_position_array, self.lidar_quat_array, self.ray_vectors, self.far_plane, self.pixels],
                device=self.device)
        if not debug:
            self.graph = wp.capture_end(device=self.device)

    def set_pose_tensor(self, lidar_positions, lidar_orientations):
        self.lidar_position_array = wp.from_torch(lidar_positions,dtype=wp.vec3)
        self.lidar_quat_array = wp.from_torch(lidar_orientations,dtype=wp.quat)


    @nvtx.annotate()
    def capture(self, debug=False):
        if self.graph is None:
            if self.cfg.return_pointcloud:
                self.create_render_graph_pointcloud(debug)
            else:
                self.create_render_graph_range(debug)

        if self.graph is not None:
            wp.capture_launch(self.graph)

        return wp.to_torch(self.pixels)

import time

import matplotlib.pyplot as plt
import torch
import numpy as np
import nvtx
import warp as wp

wp.init()
wp.config.fast_math = True


NO_HIT_RAY_VAL = wp.constant(150.0)

@wp.kernel
def draw_optimized_kernel(mesh_ids: wp.array(dtype=wp.uint64),
        lidar_pos_array: wp.array(dtype=wp.vec3),
        lidar_quat_array: wp.array(dtype=wp.quat),
        ray_vectors: wp.array2d(dtype=wp.vec3),
        far_plane: float,
        pixels: wp.array(dtype=float,ndim=3)
        ):
    
    env_id, scan_line, point_index = wp.tid()

    # env_id = tid + kernel_size * i + kernel_size * iters_per_thread * kernel_id
    cam_id = env_id # TODO: handle more cams per env
    mesh = mesh_ids[env_id]
    
    lidar_position = lidar_pos_array[cam_id]
    lidar_quaternion = lidar_quat_array[cam_id]

    noise_magnitude = 0.01

    ray_origin = lidar_position
    # perturb ray_vectors with uniform noise
    ray_dir = ray_vectors[scan_line, point_index] #+ wp.vec3(wp.random_uniform(-noise_magnitude, noise_magnitude), wp.random_uniform(-noise_magnitude, noise_magnitude), wp.random_uniform(-noise_magnitude, noise_magnitude))
    ray_direction = wp.normalize(wp.quat_rotate(lidar_quaternion, ray_dir))

    t = float(0.0)
    u = float(0.0)
    v = float(0.0)
    sign = float(0.0)
    n = wp.vec3()
    f = int(0)

    dist = NO_HIT_RAY_VAL 
    if wp.mesh_query_ray(mesh, ray_origin, ray_direction, far_plane, t, u, v, sign, n, f):
        dist = t
    
    pixels[env_id, scan_line, point_index] = dist


class WarpLidarV1:
    def __init__(self, num_envs, mesh_id_list, num_scan_lines=64, num_points_per_line=512, horizontal_fov_deg=360.0, vertical_fov_deg=90.0, far_plane=150.0, device="cuda", num_cameras=1):
        self.num_envs = num_envs
        self.num_cameras = num_cameras
        self.num_scan_lines = num_scan_lines
        self.num_points_per_line = num_points_per_line
        self.horizontal_fov = horizontal_fov_deg * np.pi / 180.0
        self.vertical_fov = vertical_fov_deg * np.pi / 180.0
        self.far_plane = far_plane  # metres
        self.device = device
        self.render_counter = 0
        self.debug_save_dir = "./tmp_images"

        # init buffers. None when uninitialized
        self.pixels = wp.zeros((self.num_envs,self.num_scan_lines,self.num_points_per_line), dtype=float, device=self.device)
        
        self.lidar_position_array = None
        self.lidar_quat_array = None
        self.mesh_ids_array = wp.array(mesh_id_list, dtype=wp.uint64)
        self.graph = None

        # populate a 2D torch array with the ray vectors that are 2d arrays of wp.vec3
        ray_vectors = torch.zeros((self.num_scan_lines, self.num_points_per_line, 3), dtype=torch.float32, device=self.device, requires_grad=False)
        for i in range(self.num_scan_lines):
            for j in range(self.num_points_per_line):
                # Check if this makes sense
                ray_vectors[num_scan_lines - i - 1, j, 0] = np.cos(self.horizontal_fov * (j / self.num_points_per_line - 0.5))
                ray_vectors[num_scan_lines - i - 1, j, 1] = np.sin(self.horizontal_fov * (j / self.num_points_per_line - 0.5))
                ray_vectors[num_scan_lines - i - 1, j, 2] = np.sin(self.vertical_fov * (i / self.num_scan_lines - 0.5))

        # normalize ray_vectors
        ray_vectors = ray_vectors / torch.norm(ray_vectors, dim=2, keepdim=True)

        # recast as 2D warp array of vec3
        self.ray_vectors = wp.from_torch(ray_vectors, dtype=wp.vec3)

    def create_render_graph(self, debug=False):

        # assert self.num_envs % (iters_per_thread * num_kernels) == 0, f"number of envs must be divisble by ({iters_per_thread}*{num_kernels})!"
        if not debug:
            # print(f"creating render graph with {num_kernels} kernels and {iters_per_thread} iterations per thread")
            wp.capture_begin(device=self.device)
        wp.launch(
            kernel=draw_optimized_kernel,
            dim=(self.num_envs, self.num_scan_lines, self.num_points_per_line),
            inputs=[self.mesh_ids_array, self.lidar_position_array, self.lidar_quat_array, self.ray_vectors, self.far_plane, self.pixels],
            device=self.device)
        if not debug:
            self.graph = wp.capture_end(device=self.device)

    def update(self, lidar_positions, lidar_orientations):
        self.lidar_position_array = wp.from_torch(lidar_positions,dtype=wp.vec3)
        self.lidar_quat_array = wp.from_torch(lidar_orientations,dtype=wp.quat)


    @nvtx.annotate()
    def render_optimized(self, debug=False, iters_per_thread=1, num_kernels=1):

        if self.graph is None:
            self.create_render_graph(debug)

        if self.graph is not None:
            wp.capture_launch(self.graph)

        return wp.to_torch(self.pixels)

    def _debug_save_imgs(self, selected_envs):
        for j in range(min(4,self.num_envs)):
            pixels_array = wp.to_torch(self.pixels)[:, :, :, 0]
            img = pixels_array[selected_envs[j]].cpu().numpy() # .reshape((self.height, self.width, 3))
            save_str = f"{self.debug_save_dir}/test_cam_{self.render_counter % self.num_cameras}_{self.render_counter // self.num_cameras}_{selected_envs[j]}.png"
            plt.imsave(save_str, img, cmap="Greys")

        self.render_counter += 1

from pathlib import Path
import subprocess

import time
import os
import tempfile
import shutil
import time

import matplotlib.pyplot as plt
import matplotlib


from trimesh import transformations as tmtf
import trimesh as tm
import pytorch3d.transforms as p3d_transforms
import torch
import numpy as np
import nvtx
import warp as wp

wp.init()
#wp.config.mode = "debug"
#wp.config.print_launches = True
#wp.config.verify_cuda = True
wp.config.fast_math = True


NO_HIT_RAY_VAL = wp.constant(20.0)

@wp.kernel
def draw_optimized_kernel(mesh_ids: wp.array(dtype=wp.uint64),
        kernel_id: int,
        cam_poss: wp.array(dtype=wp.vec3),
        cam_quats: wp.array(dtype=wp.quat),
        K_inv: wp.mat44,  
        width: int,
        height: int,
        far_plane: float,
        pixels: wp.array(dtype=float,ndim=3),
        iters_per_thread: int,
        kernel_size: int,
        c_x: int,
        c_y: int,
        calculate_depth: bool):
    
    img_id, tid = wp.tid()

    x = img_id%width
    y = img_id//width

    for i in range(iters_per_thread):
        env_id = tid + kernel_size * i + kernel_size * iters_per_thread * kernel_id
        cam_id = env_id # TODO: handle more cams per env
        mesh = mesh_ids[env_id]
        
        cam_pos = cam_poss[cam_id]
        cam_quat = cam_quats[cam_id]

        cam_coords = wp.vec3(float(x),float(y), 1.0) # this only converts the frame from warp's z-axis front to Isaac Gym's x-axis front

        cam_coords_principal = wp.vec3(float(c_x), float(c_y), 1.0) # get the vector of principal axis

        # transform to uv [-1,1]
        uv = wp.transform_vector(K_inv,cam_coords)

        uv_principal = wp.transform_vector(K_inv, cam_coords_principal) # uv for principal axis

        # compute camera ray
        
        # cam origin in world space
        ro = cam_pos
        
        # tf the direction from camera to world space and normalize
        rd = wp.normalize(wp.quat_rotate(cam_quat,uv))

        rd_principal = wp.normalize(wp.quat_rotate(cam_quat, uv_principal)) # ray direction of principal axis

        t = float(0.0)
        u = float(0.0)
        v = float(0.0)
        sign = float(0.0)
        n = wp.vec3()
        f = int(0)

        multiplier = 1.0
        if calculate_depth:
            multiplier = wp.dot(rd, rd_principal) # multiplier to project each ray on principal axis for depth instead of range

        dist = NO_HIT_RAY_VAL 
        if wp.mesh_query_ray(mesh, ro, rd, far_plane/multiplier, t, u, v, sign, n, f):
            dist = multiplier*t
        
        pixels[env_id,y,x] = dist


class WarpCamV2:

    def __init__(self, num_envs, meshes, mesh_id_list, width, height, horizontal_fov, far_plane, device, num_cameras=1, calculate_depth=True):
        self.num_envs = num_envs
        self.num_cameras = num_cameras
        self.width = width
        self.height = height
        self.horizontal_fov = horizontal_fov
        self.far_plane = far_plane
        self.device = device
        self.render_counter = 0
        self.debug_save_dir = "./tmp_images"
        self.calculate_depth = calculate_depth

        # Calculate camera params
        W = self.width
        H = self.height
        (u_0,v_0) = (W / 2 , H / 2)
        aspect_ratio = W / H 
        f = W / 2 * 1 / np.tan(np.radians(self.horizontal_fov)/2)
        vertical_fov = np.degrees(2 * np.arctan(H/(2*f)))
        alpha_u = u_0 / np.tan(np.radians(self.horizontal_fov)/2) 
        alpha_v = v_0 / np.tan(np.radians(vertical_fov)/2)
        # simple pinhole model
        self.K = wp.mat44(  alpha_u,    0.0,    u_0, 0.0,\
                            0.0,        alpha_v,v_0, 0.0,\
                            0.0,        0.0,    1.0, 0.0,\
                            0.0,        0.0,    0.0, 1.0,)
        # print(self.K)
        self.K_inv = wp.inverse(self.K)

        self.c_x = int(u_0)
        self.c_y = int(v_0)



        # init buffers. None when uninitialized
        self.pixels = wp.zeros((self.num_envs,self.height,self.width), dtype=float, device=self.device)
        
        self.cam_poss = None
        self.cam_quats = None
        # self.meshes = wp.array(meshes, dtype=wp.Mesh)
        self.mesh_ids_array = wp.array(mesh_id_list, dtype=wp.uint64)
        self.graph = None


    def create_render_graph(self, iters_per_thread, num_kernels, debug=False):

        assert self.num_envs % (iters_per_thread * num_kernels) == 0, f"number of envs must be divisble by ({iters_per_thread}*{num_kernels})!"
        if not debug:
            print(f"creating render graph with {num_kernels} kernels and {iters_per_thread} iterations per thread")
            wp.capture_begin(device=self.device)
        #with wp.ScopedTimer("render"):
        for i in range(num_kernels):
            wp.launch(
                kernel=draw_optimized_kernel,
                dim=(self.width*self.height, self.num_envs//(num_kernels*iters_per_thread)),
                inputs=[self.mesh_ids_array, i, self.cam_poss, self.cam_quats, self.K_inv, self.width, self.height, self.far_plane, self.pixels, iters_per_thread, self.num_envs//(num_kernels*iters_per_thread), self.c_x, self.c_y, self.calculate_depth],
                device=self.device)
        if not debug:
            self.graph = wp.capture_end(device=self.device)

    def update(self, cam_poss, cam_quats):
        # if self.cam_poss is None:
        self.cam_poss = wp.from_torch(cam_poss,dtype=wp.vec3)
        # if self.cam_quats is None:
        self.cam_quats = wp.from_torch(cam_quats,dtype=wp.quat)


    @nvtx.annotate()
    def render_optimized(self, debug=False, iters_per_thread=1, num_kernels=1):

        if self.graph is None:
            self.create_render_graph(iters_per_thread, num_kernels,debug)

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

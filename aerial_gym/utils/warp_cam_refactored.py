import time
import numpy as np
import nvtx
import warp as wp

wp.init()
wp.config.fast_math = True


NO_HIT_RAY_VAL = wp.constant(1000.0)
NO_HIT_SEGMENTATION_VAL = wp.constant(wp.int32(-1))


@wp.kernel
def draw_optimized_kernel_pointcloud_segmentation(mesh_ids: wp.array(dtype=wp.uint64),
        cam_poss: wp.array(dtype=wp.vec3),
        cam_quats: wp.array(dtype=wp.quat),
        K_inv: wp.mat44,
        far_plane: float,
        pixels: wp.array(dtype=wp.vec3,ndim=3),
        segmentation_pixels: wp.array(dtype=wp.int32, ndim=3),
        c_x: int,
        c_y: int,
        pointcloud_in_world_frame: bool):
    
    env_id, x, y = wp.tid()
    cam_id = env_id # TODO: handle more cams per env
    mesh = mesh_ids[env_id]
    cam_pos = cam_poss[cam_id]
    cam_quat = cam_quats[cam_id]
    cam_coords = wp.vec3(float(x),float(y), 1.0) # this only converts the frame from warp's z-axis front to Isaac Gym's x-axis front
    cam_coords_principal = wp.vec3(float(c_x), float(c_y), 1.0) # get the vector of principal axis
    # transform to uv [-1,1]
    uv = wp.normalize(wp.transform_vector(K_inv,cam_coords))
    uv_principal = wp.normalize(wp.transform_vector(K_inv, cam_coords_principal)) # uv for principal axis
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
    dist = NO_HIT_RAY_VAL
    segmentation_value = NO_HIT_SEGMENTATION_VAL
    if wp.mesh_query_ray(mesh, ro, rd, far_plane, t, u, v, sign, n, f):
        dist = t
        mesh_obj = wp.mesh_get(mesh)
        face_index = mesh_obj.indices[f*3]
        segmentation_value = wp.int32(mesh_obj.velocities[face_index][0])
    if pointcloud_in_world_frame:
        pixels[env_id,y,x] = ro + dist * rd
    else:
        pixels[env_id,y,x] = dist * uv
    segmentation_pixels[env_id,y,x] = segmentation_value


@wp.kernel
def draw_optimized_kernel_pointcloud(mesh_ids: wp.array(dtype=wp.uint64),
        cam_poss: wp.array(dtype=wp.vec3),
        cam_quats: wp.array(dtype=wp.quat),
        K_inv: wp.mat44,
        far_plane: float,
        pixels: wp.array(dtype=wp.vec3,ndim=3),
        c_x: int,
        c_y: int,
        pointcloud_in_world_frame: bool):
    
    env_id, x, y = wp.tid()
    cam_id = env_id # TODO: handle more cams per env
    mesh = mesh_ids[env_id]
    cam_pos = cam_poss[cam_id]
    cam_quat = cam_quats[cam_id]
    cam_coords = wp.vec3(float(x),float(y), 1.0) # this only converts the frame from warp's z-axis front to Isaac Gym's x-axis front
    cam_coords_principal = wp.vec3(float(c_x), float(c_y), 1.0) # get the vector of principal axis
    # transform to uv [-1,1]
    uv = wp.normalize(wp.transform_vector(K_inv,cam_coords))
    uv_principal = wp.normalize(wp.transform_vector(K_inv, cam_coords_principal)) # uv for principal axis
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
    dist = NO_HIT_RAY_VAL 
    if wp.mesh_query_ray(mesh, ro, rd, far_plane, t, u, v, sign, n, f):
        dist = t
    if pointcloud_in_world_frame:
        pixels[env_id,y,x] = ro + dist * rd
    else:
        pixels[env_id,y,x] = dist * uv

@wp.kernel
def draw_optimized_kernel_depth_range(mesh_ids: wp.array(dtype=wp.uint64),
        cam_poss: wp.array(dtype=wp.vec3),
        cam_quats: wp.array(dtype=wp.quat),
        K_inv: wp.mat44,
        far_plane: float,
        pixels: wp.array(dtype=float,ndim=3),
        c_x: int,
        c_y: int,
        calculate_depth: bool):
    
    env_id, x, y = wp.tid()
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

@wp.kernel
def draw_optimized_kernel_depth_range_segmentation(mesh_ids: wp.array(dtype=wp.uint64),
        cam_poss: wp.array(dtype=wp.vec3),
        cam_quats: wp.array(dtype=wp.quat),
        K_inv: wp.mat44,
        far_plane: float,
        pixels: wp.array(dtype=float,ndim=3),
        segmentation_pixels: wp.array(dtype=wp.int32, ndim=3),
        c_x: int,
        c_y: int,
        calculate_depth: bool):
    
    env_id, x, y = wp.tid()
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
    segmentation_value = NO_HIT_SEGMENTATION_VAL
    if wp.mesh_query_ray(mesh, ro, rd, far_plane/multiplier, t, u, v, sign, n, f):
        dist = multiplier*t
        mesh_obj = wp.mesh_get(mesh)
        face_index = mesh_obj.indices[f*3]
        segmentation_value = wp.int32(mesh_obj.velocities[face_index][0])
    
    pixels[env_id,y,x] = dist
    segmentation_pixels[env_id,y,x] = segmentation_value


class WarpCamRefactored:
    def __init__(self, num_envs, mesh_id_list, pixels, segmentation_pixels, config, device="cuda:0"):
        self.cfg = config
        self.num_envs = num_envs
        self.num_sensors = self.cfg.num_sensors
        if self.num_sensors > 1:
            self.num_cameras = self.num_sensors

        self.width = self.cfg.width
        self.height = self.cfg.height

        self.horizontal_fov = np.radians(self.cfg.horizontal_fov_deg)
        # Calculate camera params
        W = self.width
        H = self.height
        (u_0,v_0) = (W / 2 , H / 2)
        f = W / 2 * 1 / np.tan(self.horizontal_fov/2)

        vertical_fov = 2 * np.arctan(H/(2*f))
        alpha_u = u_0 / np.tan(self.horizontal_fov/2) 
        alpha_v = v_0 / np.tan(vertical_fov/2)

        # print vertical and horizontal fov in both degrees and radians
        print(f"vertical fov: {np.degrees(vertical_fov)} degrees, {vertical_fov} radians")
        print(f"horizontal fov: {np.degrees(self.horizontal_fov)} degrees, {self.horizontal_fov} radians")

        # simple pinhole model
        self.K = wp.mat44(  alpha_u,    0.0,    u_0, 0.0,\
                            0.0,        alpha_v,v_0, 0.0,\
                            0.0,        0.0,    1.0, 0.0,\
                            0.0,        0.0,    0.0, 1.0,)
        # print(self.K)
        self.K_inv = wp.inverse(self.K)

        self.c_x = int(u_0)
        self.c_y = int(v_0)



        self.far_plane = self.cfg.max_range
        self.calculate_depth = self.cfg.calculate_depth
        self.device = device
        
        # init buffers. None when uninitialized
        if self.cfg.return_pointcloud:
            self.pixels = wp.from_torch(pixels,dtype=wp.vec3)
            self.pointcloud_in_world_frame = self.cfg.pointcloud_in_world_frame
        else:
            self.pixels = wp.from_torch(pixels,dtype=wp.float32)
        if self.cfg.segmentation_camera == True:
            self.segmentation_pixels = wp.from_torch(segmentation_pixels,dtype=wp.int32)
        
        self.cam_poss = None
        self.cam_quats = None
        self.mesh_ids_array = wp.array(mesh_id_list, dtype=wp.uint64)
        self.graph = None


    def create_render_graph_pointcloud(self,debug=False):
        if not debug:
            print(f"creating render graph")
            wp.capture_begin(device=self.device)
        #with wp.ScopedTimer("render"):
        if self.cfg.segmentation_camera == True:
            wp.launch(
            kernel=draw_optimized_kernel_pointcloud_segmentation,
            dim=(self.num_envs, self.width, self.height),
            inputs=[self.mesh_ids_array, self.cam_poss, self.cam_quats, self.K_inv, self.far_plane, self.pixels, self.segmentation_pixels, self.c_x, self.c_y, self.pointcloud_in_world_frame],
            device=self.device)
        else:
            wp.launch(
                kernel=draw_optimized_kernel_pointcloud,
                dim=(self.num_envs, self.width, self.height),
                inputs=[self.mesh_ids_array, self.cam_poss, self.cam_quats, self.K_inv, self.far_plane, self.pixels, self.c_x, self.c_y, self.pointcloud_in_world_frame],
                device=self.device)
        if not debug:
            self.graph = wp.capture_end(device=self.device)

    def create_render_graph_depth_range(self,debug=False):
        if not debug:
            print(f"creating render graph")
            wp.capture_begin(device=self.device)
        #with wp.ScopedTimer("render"):
        if self.cfg.segmentation_camera == True:
            wp.launch(
                kernel=draw_optimized_kernel_depth_range_segmentation,
                dim=(self.num_envs, self.width, self.height),
                inputs=[self.mesh_ids_array, self.cam_poss, self.cam_quats, self.K_inv, self.far_plane, self.pixels, self.segmentation_pixels, self.c_x, self.c_y, self.calculate_depth],
                device=self.device)
        else:
            wp.launch(
                kernel=draw_optimized_kernel_depth_range,
                dim=(self.num_envs, self.width, self.height),
                inputs=[self.mesh_ids_array, self.cam_poss, self.cam_quats, self.K_inv, self.far_plane, self.pixels, self.c_x, self.c_y, self.calculate_depth],
                device=self.device)
        if not debug:
            self.graph = wp.capture_end(device=self.device)

    def set_pose_tensor(self, cam_poss, cam_quats):
        self.cam_poss = wp.from_torch(cam_poss,dtype=wp.vec3)
        self.cam_quats = wp.from_torch(cam_quats,dtype=wp.quat)


    @nvtx.annotate()
    def capture(self, debug=False):
        if self.graph is None:
            if self.cfg.return_pointcloud:
                self.create_render_graph_pointcloud(debug=debug)
            else:
                self.create_render_graph_depth_range(debug=debug)
        
        if self.graph is not None:
            wp.capture_launch(self.graph)
        
        return wp.to_torch(self.pixels)

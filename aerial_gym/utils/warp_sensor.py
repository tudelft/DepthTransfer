import torch
from aerial_gym.utils.warp_cam_refactored import WarpCamRefactored
from aerial_gym.utils.warp_lidar_refactored import WarpLidarRefactored

import warp as wp
    
    
class WarpSensor:
    def __init__(self, num_envs, mesh_id_list, pixels, segmentation_pixels, config, device="cuda:0"):
        self.cfg = config
        self.num_envs = num_envs
        self.mesh_id_list = mesh_id_list
        self.pixels = pixels
        self.refit_graph = None
        self.segmentation_pixels = segmentation_pixels
        self.device = device
        if self.cfg.sensor_type == "lidar":
            self.sensor = WarpLidarRefactored(num_envs=self.num_envs, mesh_id_list=self.mesh_id_list, pixels=self.pixels, segmentation_pixels=self.segmentation_pixels, config=self.cfg)
        elif self.cfg.sensor_type == "camera":
            self.sensor = WarpCamRefactored(num_envs=self.num_envs, mesh_id_list=self.mesh_id_list, pixels=self.pixels, segmentation_pixels=self.segmentation_pixels, config=self.cfg)
        else:
            raise NotImplementedError
    
    def initialize_sensor(self):
        self.capture()

    def capture(self):
        self.sensor.capture()
        
        # self.apply_noise()
        if self.cfg.return_pointcloud == True:
            # if pointcloud is to be in the world frame, we need to transform it
            if self.cfg.pointcloud_in_world_frame == False:
                self.pixels[self.pixels.norm(dim=3, keepdim=True).expand(-1, -1, -1, 3) > self.cfg.max_range] = self.cfg.far_out_of_range_value
                self.pixels[self.pixels.norm(dim=3, keepdim=True).expand(-1, -1, -1, 3) < self.cfg.min_range] = self.cfg.near_out_of_range_value
        else:
            self.pixels[self.pixels > self.cfg.max_range] = self.cfg.far_out_of_range_value
            self.pixels[self.pixels < self.cfg.min_range] = self.cfg.near_out_of_range_value

        if self.cfg.normalize_range and self.cfg.pointcloud_in_world_frame == False:
            self.pixels[:] = self.pixels/self.cfg.max_range

    
    def apply_noise(self):
        self.pixels[:] = torch.normal(mean=self.pixels, std=self.cfg.pixel_std_dev_multiplier * self.pixels)
        self.pixels[torch.bernoulli(torch.ones_like(self.pixels)*self.cfg.pixel_dropout_prob) > 0] = self.cfg.near_out_of_range_value

    def set_pose_tensor(self, positions, orientations):
        self.sensor.set_pose_tensor(positions, orientations)
    
    def refit_meshes(self, mesh_array, env_ids):
        # TODO See if below can be graph-captured
        # if self.refit_graph is None:
        #     wp.capture_begin(device=self.device)
        #     for i in range(self.num_envs):
        #         if refit_flag[i]:
        #             mesh_array[i].refit()
        #     self.refit_graph = wp.capture_end(device=self.device)
        # else:
        #     wp.capture_launch(self.refit_graph)
        for i in env_ids:
            mesh_array[i].refit()
        return


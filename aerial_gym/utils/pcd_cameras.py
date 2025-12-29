from isaacgym import gymapi
from isaacgym import gymtorch
import torch
from typing import Dict, Any, Optional, List, Tuple

# define an emum for the camera type
class PCD_CAMERA_TYPE:
    FRONT = 0
    BACK = 1
    LEFT = 2
    RIGHT = 3
    TOP = 4
    BOTTOM = 5

class PCDCameraManager:
    def __init__(self, gym, sim, pcd_camera_param):
        self.gym = gym
        self.sim = sim
        self.pcd_camera_param = pcd_camera_param
        self.extra_front_depth_tensors: List[torch.Tensor] = []
        self.extra_back_depth_tensors: List[torch.Tensor] = []
        self.extra_left_depth_tensors: List[torch.Tensor] = []
        self.extra_right_depth_tensors: List[torch.Tensor] = []
        # self.extra_top_depth_tensors: List[torch.Tensor] = []
        # self.extra_bottom_depth_tensors: List[torch.Tensor] = []

    def create_pcd_cameras_single_drone(self, env, drone_actor):
        pc_camera_props = gymapi.CameraProperties()
        pc_camera_props.enable_tensors = True
        pc_camera_props.width = self.pcd_camera_param.width
        pc_camera_props.height = self.pcd_camera_param.height
        pc_camera_props.far_plane = self.pcd_camera_param.max_range
        pc_camera_props.horizontal_fov = self.pcd_camera_param.horizontal_fov_deg

        pcd_front_cam = self.gym.create_camera_sensor(env, pc_camera_props)
        pcd_back_cam = self.gym.create_camera_sensor(env, pc_camera_props)
        pcd_left_cam = self.gym.create_camera_sensor(env, pc_camera_props)
        pcd_right_cam = self.gym.create_camera_sensor(env, pc_camera_props)
        # pcd_top_cam = self.gym.create_camera_sensor(env, pc_camera_props)
        # pcd_bottom_cam = self.gym.create_camera_sensor(env, pc_camera_props)

        pcd_front_cam_body_tf: gymapi.Transform = gymapi.Transform()
        pcd_back_cam_body_tf: gymapi.Transform = gymapi.Transform()
        pcd_left_cam_body_tf: gymapi.Transform = gymapi.Transform()
        pcd_right_cam_body_tf: gymapi.Transform = gymapi.Transform()
        # pcd_top_cam_body_tf: gymapi.Transform = gymapi.Transform()
        # pcd_bottom_cam_body_tf: gymapi.Transform = gymapi.Transform()

        pcd_back_cam_body_tf.r = gymapi.Quat.from_axis_angle(gymapi.Vec3(0, 0, 1), torch.pi)
        pcd_left_cam_body_tf.r = gymapi.Quat.from_axis_angle(gymapi.Vec3(0, 0, 1), torch.pi / 2)
        pcd_right_cam_body_tf.r = gymapi.Quat.from_axis_angle(gymapi.Vec3(0, 0, 1), -torch.pi / 2)
        # pcd_top_cam_body_tf.r = gymapi.Quat.from_axis_angle(gymapi.Vec3(0, 1, 0), -torch.pi / 2)
        # pcd_bottom_cam_body_tf.r = gymapi.Quat.from_axis_angle(gymapi.Vec3(0, 1, 0), torch.pi / 2)

        pcd_front_cam_body_tf.p = gymapi.Vec3(0.15, 0, 0.0)
        pcd_back_cam_body_tf.p = gymapi.Vec3(-0.15, 0, 0.0)
        pcd_left_cam_body_tf.p = gymapi.Vec3(0, 0.15, 0.0)
        pcd_right_cam_body_tf.p = gymapi.Vec3(0, -0.15, 0.0)
        # pcd_top_cam_body_tf.p = gymapi.Vec3(0, 0, 0.05)
        # pcd_bottom_cam_body_tf.p = gymapi.Vec3(0, 0, -0.05)


        self.gym.attach_camera_to_body(pcd_front_cam, env, drone_actor, 
                                       pcd_front_cam_body_tf, gymapi.FOLLOW_TRANSFORM)
        self.gym.attach_camera_to_body(pcd_back_cam, env, drone_actor,
                                        pcd_back_cam_body_tf, gymapi.FOLLOW_TRANSFORM)
        self.gym.attach_camera_to_body(pcd_left_cam, env, drone_actor,
                                        pcd_left_cam_body_tf, gymapi.FOLLOW_TRANSFORM)
        self.gym.attach_camera_to_body(pcd_right_cam, env, drone_actor,
                                        pcd_right_cam_body_tf, gymapi.FOLLOW_TRANSFORM)
        # self.gym.attach_camera_to_body(pcd_top_cam, env, drone_actor,
        #                                 pcd_top_cam_body_tf, gymapi.FOLLOW_TRANSFORM)
        # self.gym.attach_camera_to_body(pcd_bottom_cam, env, drone_actor,
        #                                 pcd_bottom_cam_body_tf, gymapi.FOLLOW_TRANSFORM)
        self.extra_front_depth_tensors.append(
                gymtorch.wrap_tensor(
                self.gym.get_camera_image_gpu_tensor(
                    self.sim, env, pcd_front_cam, gymapi.IMAGE_DEPTH
                )
            )
        )
        self.extra_back_depth_tensors.append(
                gymtorch.wrap_tensor(
                self.gym.get_camera_image_gpu_tensor(
                    self.sim, env, pcd_back_cam, gymapi.IMAGE_DEPTH
                )
            )
        )
        self.extra_left_depth_tensors.append(
                gymtorch.wrap_tensor(
                self.gym.get_camera_image_gpu_tensor(
                    self.sim, env, pcd_left_cam, gymapi.IMAGE_DEPTH
                )
            )
        )
        self.extra_right_depth_tensors.append(
                gymtorch.wrap_tensor(
                self.gym.get_camera_image_gpu_tensor(
                    self.sim, env, pcd_right_cam, gymapi.IMAGE_DEPTH
                )
            )
        )
        # self.extra_top_depth_tensors.append(
        #         gymtorch.wrap_tensor(
        #         self.gym.get_camera_image_gpu_tensor(
        #             self.sim, env, pcd_top_cam, gymapi.IMAGE_DEPTH
        #         )
        #     )
        # )
        # self.extra_bottom_depth_tensors.append(
        #         gymtorch.wrap_tensor(
        #         self.gym.get_camera_image_gpu_tensor(
        #             self.sim, env, pcd_bottom_cam, gymapi.IMAGE_DEPTH
        #         )
        #     )
        # )

            
    def get_pcd_camera_data(self):
        pcd_front_depth = -torch.stack(self.extra_front_depth_tensors)
        pcd_back_depth = -torch.stack(self.extra_back_depth_tensors)
        pcd_left_depth = -torch.stack(self.extra_left_depth_tensors)
        pcd_right_depth = -torch.stack(self.extra_right_depth_tensors)
        # pcd_top_depth = -torch.stack(self.extra_top_depth_tensors)
        # pcd_bottom_depth = -torch.stack(self.extra_bottom_depth_tensors)
        # pcd_front_depth = torch.clamp(pcd_front_depth, min=0.0, max=self.pcd_camera_param.max_range)
        # pcd_back_depth = torch.clamp(pcd_back_depth, min=0.0, max=self.pcd_camera_param.max_range)
        # pcd_left_depth = torch.clamp(pcd_left_depth, min=0.0, max=self.pcd_camera_param.max_range)
        # pcd_right_depth = torch.clamp(pcd_right_depth, min=0.0, max=self.pcd_camera_param.max_range)
        # pcd_top_depth = torch.clamp(pcd_top_depth, min=0.0, max=self.pcd_camera_param.max_range)
        # pcd_bottom_depth = torch.clamp(pcd_bottom_depth, min=0.0, max=self.pcd_camera_param.max_range)
        pcd_depth =torch.stack(
            [pcd_front_depth, pcd_back_depth, pcd_left_depth, pcd_right_depth],
            dim=1
        )
        return pcd_depth.nan_to_num_(posinf=0.0).cpu()
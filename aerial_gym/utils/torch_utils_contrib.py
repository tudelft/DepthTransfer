
import torch
import numpy as np

@torch.jit.script
def copysign_3d(a, b):
    # type: (float, Tensor) -> Tensor
    a = torch.tensor(a, device=b.device, dtype=torch.float).repeat(b.shape[0], b.shape[1])
    return torch.abs(a) * torch.sign(b)

@torch.jit.script
def get_euler_xyz_3d(q):
    qx, qy, qz, qw = 0, 1, 2, 3
    # roll (x-axis rotation)
    sinr_cosp = 2.0 * (q[:, :, qw] * q[:, :, qx] + q[:, :, qy] * q[:, :, qz])
    cosr_cosp = q[:, :, qw] * q[:, :, qw] - q[:, :, qx] * \
        q[:, :, qx] - q[:, :, qy] * q[:, :, qy] + q[:, :, qz] * q[:, :, qz]
    roll = torch.atan2(sinr_cosp, cosr_cosp)

    # pitch (y-axis rotation)
    sinp = 2.0 * (q[:, :, qw] * q[:, :, qy] - q[:, :, qz] * q[:, :, qx])
    pitch = torch.where(torch.abs(sinp) >= 1, copysign_3d(
        np.pi / 2.0, sinp), torch.asin(sinp))

    # yaw (z-axis rotation)
    siny_cosp = 2.0 * (q[:, :, qw] * q[:, :, qz] + q[:, :, qx] * q[:, :, qy])
    cosy_cosp = q[:, :, qw] * q[:, :, qw] + q[:, :, qx] * \
        q[:, :, qx] - q[:, :, qy] * q[:, :, qy] - q[:, :, qz] * q[:, :, qz]
    yaw = torch.atan2(siny_cosp, cosy_cosp)

    return roll % (2*np.pi), pitch % (2*np.pi), yaw % (2*np.pi)

@torch.jit.script
def quat_rotate_inverse_3d(q, v):
    q_temp = q.clone()
    v_temp = v.clone()
    shape = q_temp.shape
    q_w = q_temp[:, :, -1].view(shape[0]*shape[1], 1)
    q_vec = q_temp[:, :, :3].view(shape[0]*shape[1], 3)
    a = (v_temp.view(shape[0]*shape[1], 3)) * (2.0 * q_w ** 2 - 1.0)
    b = torch.cross(q_vec, (v_temp.view(shape[0]*shape[1], 3)), dim=-1) * q_w * 2.0
    c = q_vec * \
        torch.bmm(q_vec.view(shape[0]*shape[1], 1, 3), v_temp.view(
            shape[0]*shape[1], 3, 1)).squeeze(-1) * 2.0
    return (a - b + c).view(shape[0], shape[1], 3)
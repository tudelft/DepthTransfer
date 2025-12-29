from isaacgym.torch_utils import *
import torch

class TensorIMUV2:
    def __init__(self, num_envs, dt, 
                    config, gravity, device = torch.device("cuda")):
        '''
        num_envs: number of environments
        dt: time step
        config: config file
        gravity: gravity vector
        device: device to run the simulation on

        NOTE: This imu_simulator does NOT measure the effects of coriolis force or acceleration due to rotating frames.
        The values obtained for the sensor are those that are measured at offset (0, 0, 0) from the parent link.
        The orientation in the physics engine is set to (0,0,0,1), i.e. no rotation.
        This means that we are free to transform the sensor to any orientation we want w.r.t sensor_parent_link and transform the values appropriately.
        '''

        self.num_envs = num_envs
        self.dt = dt
        self.cfg = config
        self.debug = self.cfg.debug
        self.device = device
        self.gravity_compensation = self.cfg.gravity_compensation

        # first 3 values for acc bias std, next 3 for gyro bias std
        self.bias_std = torch.tensor(self.cfg.bias_std, device=self.device, requires_grad=False).expand(self.num_envs, -1)
        # first 3 vaues for acc noise std, next 3 for gyro noise std
        self.imu_noise_std = torch.tensor(self.cfg.imu_noise_std, device=self.device, requires_grad=False).expand(self.num_envs, -1)

        self.max_measurement_value = torch.tensor(self.cfg.max_measurement_value, device=self.device, requires_grad=False).expand(self.num_envs, -1)
        self.max_bias_init_value = torch.tensor(self.cfg.max_bias_init_value, device=self.device, requires_grad=False)

        self.bias = torch.zeros((self.num_envs,6), device=self.device, requires_grad=False)
        self.noise = torch.zeros((self.num_envs,6), device=self.device, requires_grad=False)

        self.imu_meas = torch.zeros((self.num_envs,6), device=self.device, requires_grad=False)

        orientation_vec = torch.deg2rad(torch.tensor(self.cfg.orientation_euler_deg, device=self.device, requires_grad=False))

        # Nominal sensor orientation value that can be perturbed
        self.sensor_orientation_quat = quat_from_euler_xyz(orientation_vec[0], orientation_vec[1], orientation_vec[2]).expand(self.num_envs, -1)
        self.sensor_quats = self.sensor_orientation_quat.clone()
        self.sensor_orientation_perturb_rad_amplitude = torch.deg2rad(torch.tensor(self.cfg.orientation_perturb_amplitude_deg, device=self.device, requires_grad=False)).expand(self.num_envs, -1)

        self.g_world = torch.tensor(gravity, device=self.device, requires_grad=False).expand(self.num_envs, -1)

        self.g_world = self.g_world * (1 - int(self.gravity_compensation))

        self.enable_noise = int(self.cfg.enable_noise)
        self.enable_bias = int(self.cfg.enable_bias)
    
    
    def sample_noise(self):
        self.noise = torch.randn((self.num_envs,6), device=self.device) * self.imu_noise_std
    
    def update_bias(self):
        self.bias_update_rate = torch.randn((self.num_envs,6), device=self.device) * self.bias_std
        self.bias += self.bias_update_rate * self.dt


    def print_params(self):
        for name, value in vars(self).items():
            # if tensor print dtype otherwise print type
            if isinstance(value, torch.Tensor):
                print(name, value.dtype)
            else:
                print(name, type(value))

    def update(self, robot_quat, accel_t, ang_rate_t, world_frame=False):

        '''
        world_frame: if accel_t and ang_rate_t are in world frame or not
        '''
        if world_frame:
            acceleration = quat_rotate_inverse(quat_mul(robot_quat, self.sensor_quats), (accel_t - self.g_world))
            ang_rate = quat_rotate_inverse(quat_mul(robot_quat, self.sensor_quats), ang_rate_t)
        else:
            # Rotate the acceleration and angular rate from true sensor frame to perturbed sensor frame
            acceleration = quat_rotate_inverse(self.sensor_quats, accel_t) - quat_rotate_inverse(quat_mul(robot_quat, self.sensor_quats), self.g_world)
            ang_rate = quat_rotate_inverse(self.sensor_quats, ang_rate_t)

        self.sample_noise()
        self.update_bias()

        accel_meas = acceleration + self.enable_bias * self.bias[:,:3] + self.enable_noise * self.noise[:,:3]
        ang_rate_meas = ang_rate + self.enable_bias * self.bias[:,3:] + self.enable_noise * self.noise[:,3:]

        # clamp the measurements from acceleroemter and gyro to max values
        accel_meas = tensor_clamp(accel_meas, -self.max_measurement_value[:, 0:3], self.max_measurement_value[:, 0:3])
        ang_rate_meas = tensor_clamp(ang_rate_meas, -self.max_measurement_value[:, 3:], self.max_measurement_value[:, 3:])

        return accel_meas, ang_rate_meas


    # TODO @mihirk check how to reset biases properly with @mohit @morten
    def reset(self):
        self.bias.zero_()
        self.bias[:] =  self.max_bias_init_value *  (2.0*(torch.rand_like(self.bias) - 0.5))
        # uniformly sample the perturbation of sensor mounting orientation from the nominal orientation
        sampled_orientation_perturbation_rad = 2.0 * (torch.rand((self.num_envs,3), device=self.device) - 0.5) * self.sensor_orientation_perturb_rad_amplitude
        sampled_orientation_perturbation_quat = quat_from_euler_xyz(sampled_orientation_perturbation_rad[:,0], sampled_orientation_perturbation_rad[:,1], sampled_orientation_perturbation_rad[:,2])
        self.sensor_quats[:] = quat_mul(self.sensor_orientation_quat, sampled_orientation_perturbation_quat)
    
    def reset_idx(self, env_ids):
        self.bias[env_ids, :] = (self.max_bias_init_value *  (2.0*(torch.rand_like(self.bias) - 0.5)))[env_ids, :]
        # uniformly sample the perturbation of sensor mounting orientation from the nominal orientation
        sampled_orientation_perturbation_rad = 2.0 * (torch.rand((self.num_envs,3), device=self.device) - 0.5) * self.sensor_orientation_perturb_rad_amplitude
        sampled_orientation_perturbation_quat = quat_from_euler_xyz(sampled_orientation_perturbation_rad[:,0], sampled_orientation_perturbation_rad[:,1], sampled_orientation_perturbation_rad[:,2])
        self.sensor_quats[env_ids, :] = quat_mul(self.sensor_orientation_quat, sampled_orientation_perturbation_quat)[env_ids, :]


##############################################################################################################
# Helper functions
##############################################################################################################

@torch.jit.script
def quat_rotate(q, v):
    shape = q.shape
    q_w = q[:, -1]
    q_vec = q[:, :3]
    a = v * (2.0 * q_w ** 2 - 1.0).unsqueeze(-1)
    b = torch.cross(q_vec, v, dim=-1) * q_w.unsqueeze(-1) * 2.0
    c = q_vec * \
        torch.bmm(q_vec.view(shape[0], 1, 3), v.view(
            shape[0], 3, 1)).squeeze(-1) * 2.0
    return a + b + c

@torch.jit.script
def quat_axis(q, axis=0):
    # type: (Tensor, int) -> Tensor
    basis_vec = torch.zeros(q.shape[0], 3, device=q.device)
    basis_vec[:, axis] = 1
    return quat_rotate(q, basis_vec)
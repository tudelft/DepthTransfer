import torch 
import pytorch3d.transforms as p3d_transforms

def float_angle_nomalize(angle):
    return angle - 2 * torch.pi * torch.floor((angle + torch.pi) / (2 * torch.pi))


class ReferenceBase:
    def __init__(self, config, num_envs, reference_steps, device):
        self.device = device
        self.num_envs = num_envs
        self.reference_steps = reference_steps
        self.reference_steps_rest = reference_steps
        self.euler_angles = torch.zeros(3, dtype=torch.float32, device=self.device, requires_grad=False).tile((num_envs, 1))
        self.rates = torch.zeros(3, dtype=torch.float32, device=self.device, requires_grad=False).tile((num_envs, 1))
        self.accel = torch.zeros(3, dtype=torch.float32, device=self.device, requires_grad=False).tile((num_envs, 1))

        self.linear_accel = torch.zeros(3, dtype=torch.float32, device=self.device, requires_grad=False).tile((num_envs, 1))
        self.linear_vel = torch.zeros(3, dtype=torch.float32, device=self.device, requires_grad=False).tile((num_envs, 1))
        self.overshoot_vel = torch.zeros(3, dtype=torch.float32, device=self.device, requires_grad=False).tile((num_envs, 1))
        self.overshoot_accel = torch.zeros(3, dtype=torch.float32, device=self.device, requires_grad=False).tile((num_envs, 1))
        self.linear_pos = torch.zeros(3, dtype=torch.float32, device=self.device, requires_grad=False).tile((num_envs, 1))
        self.linear_norm_thrust = torch.zeros(1, dtype=torch.float32, device=self.device, requires_grad=False).tile((num_envs, 1))

        self.thrust = torch.ones(num_envs, dtype=torch.float32, device=self.device, requires_grad=False) * 9.81
        self.thrust_rate = torch.zeros(num_envs, dtype=torch.float32, device=self.device, requires_grad=False)

        self.model_omega = torch.tensor(config.stabilization_attitude_ref_omega, dtype=torch.float32, device=self.device, requires_grad=False).tile((num_envs, 1))
        self.model_zeta = torch.tensor(config.stabilization_attitude_ref_zeta, dtype=torch.float32, device=self.device, requires_grad=False).tile((num_envs, 1))
        self.saturation_max_rate = torch.tensor(config.stabilization_attitude_ref_max, dtype=torch.float32, device=self.device, requires_grad=False).tile((num_envs, 1))
        self.saturation_max_accel = torch.tensor(config.stabilization_attitude_ref_max_omege_dot, dtype=torch.float32, device=self.device, requires_grad=False).tile((num_envs, 1))
        self.stabilization_thrust_rate_max = torch.tensor(config.stabilization_thrust_rate, dtype=torch.float32, device=self.device, requires_grad=False).tile(num_envs)

        self.model_omega_current = self.model_omega.clone()
        self.model_zeta_current = self.model_zeta.clone()
        self.saturation_max_rate_current = self.saturation_max_rate.clone()
        self.saturation_max_accel_current = self.saturation_max_accel.clone()
        self.stabilization_thrust_rate_max_current = self.stabilization_thrust_rate_max.clone()

    def randomize_params(self, env_ids):
        self.model_omega_current[env_ids] = (self.model_omega + ((torch.rand_like(self.model_omega) - 0.5) * 2) * 0.8)[env_ids]
        self.model_zeta_current[env_ids] = (self.model_zeta + ((torch.rand_like(self.model_zeta) - 0.5) * 2) * 0.8)[env_ids]
        self.saturation_max_rate_current[env_ids] = (self.saturation_max_rate + ((torch.rand_like(self.saturation_max_rate) - 0.5) * 2) * 0.8)[env_ids]
        self.saturation_max_accel_current[env_ids] = (self.saturation_max_accel + ((torch.rand_like(self.saturation_max_accel) - 0.5) * 2) * 0.8)[env_ids]
        self.stabilization_thrust_rate_max_current[env_ids] = (self.stabilization_thrust_rate_max + ((torch.rand_like(self.stabilization_thrust_rate_max) - 0.5) * 2) * 1.6)[env_ids]

    def attitude_ref_euler_float_update(self, sp_eulers, dt, thrust):
        self.update_euler_angles(dt)
        self.euler_angles[:, 2] = float_angle_nomalize(self.euler_angles[:, 2])

        # update thrust rate
        rest_time = self.reference_steps_rest * dt
        self.thrust_rate = (thrust - self.thrust) * (1.0/rest_time)
        self.thrust_rate = torch.where(torch.abs(self.thrust_rate) > self.stabilization_thrust_rate_max_current, torch.sign(self.thrust_rate) * self.stabilization_thrust_rate_max_current, self.thrust_rate)
        # updae thrust
        self.thrust += (self.thrust_rate * dt)
        # self.thrust = thrust

        # integrate reference rotational speed
        delta_accel = self.accel * dt
        self.rates += delta_accel
        # compute reference attitude error
        attitude_error = self.euler_angles - sp_eulers
        attitude_error[:, 2] = float_angle_nomalize(attitude_error[:, 2])

        # compute reference angular accelerations
        self.accel = -2.0 * self.model_omega_current * self.model_zeta_current * self.rates - self.model_omega_current**2 * attitude_error
        # saturate angular accelerations
        self.attitude_ref_float_saturate_naive()

        thrust_body = torch.zeros((self.thrust.shape[0], 3), device=self.device)
        thrust_body[:, 2] = self.thrust
        self.linear_accel = torch.bmm(p3d_transforms.euler_angles_to_matrix(self.euler_angles[:, [2, 1, 0]], "ZYX"), thrust_body.unsqueeze(-1)).squeeze(-1)
        self.linear_accel[:, 2] = self.thrust - 9.81
        # self.linear_accel = (self.linear_accel - torch.tensor([0, 0, 9.81], dtype=torch.float32, device=self.device))
        self.overshoot_accel = self.linear_accel.clone()
        self.overshoot_accel[:, 2] = self.overshoot_accel[:, 2]*1.0

        self.linear_pos = self.linear_pos + self.overshoot_vel * dt + 0.5 * self.overshoot_accel * dt**2
        
        self.linear_vel = self.linear_vel + self.linear_accel * dt
        self.overshoot_vel = self.overshoot_vel + self.overshoot_accel * dt

        # update rest reference steps
        self.reference_steps_rest -= 1
        if self.reference_steps_rest == 0:
            self.reference_steps_rest = self.reference_steps

    def attitude_ref_float_saturate_naive(self):
        self.accel = torch.where(self.accel > self.saturation_max_accel_current, self.saturation_max_accel_current, self.accel)
        self.accel = torch.where(self.accel < -self.saturation_max_accel_current, -self.saturation_max_accel_current, self.accel)

        self.rates = torch.where(self.rates >= self.saturation_max_rate_current, self.saturation_max_rate_current, self.rates)
        self.accel = torch.where((self.rates >= self.saturation_max_rate_current) & (self.accel > 0.0), 0.0, self.accel)
        self.rates = torch.where(self.rates <= -self.saturation_max_rate_current, -self.saturation_max_rate_current, self.rates)
        self.accel = torch.where((self.rates <= -self.saturation_max_rate_current) & (self.accel < 0.0), 0.0, self.accel)

    def update_euler_angles(self, dt):
        rotation_matrix = p3d_transforms.matrix_to_quaternion(p3d_transforms.euler_angles_to_matrix(self.euler_angles[..., [2, 1, 0]], "ZYX"))
        rotation_matrix_dot = self.quaternion_derivative(rotation_matrix, self.rates)
        rotation_matrix = rotation_matrix + rotation_matrix_dot * dt
        rotation_matrix = rotation_matrix / torch.norm(rotation_matrix, dim=1, keepdim=True)
        self.euler_angles = p3d_transforms.matrix_to_euler_angles(p3d_transforms.quaternion_to_matrix(rotation_matrix), "ZYX")[..., [2, 1, 0]]

    @staticmethod
    def quaternion_derivative(q, omega):
        q_dot = torch.zeros_like(q)
        q_dot[:, 0] = -0.5 * (omega[:, 0] * q[:, 1] + omega[:, 1] * q[:, 2] + omega[:, 2] * q[:, 3])
        q_dot[:, 1] = 0.5 * (omega[:, 0] * q[:, 0] + omega[:, 2] * q[:, 2] - omega[:, 1] * q[:, 3])
        q_dot[:, 2] = 0.5 * (omega[:, 1] * q[:, 0] - omega[:, 2] * q[:, 1] + omega[:, 0] * q[:, 3])
        q_dot[:, 3] = 0.5 * (omega[:, 2] * q[:, 0] + omega[:, 1] * q[:, 1] - omega[:, 0] * q[:, 2])
        return q_dot

    def reset_reference(self, env_ids, euler_angles, rates, linear_pos, linear_vel, linear_accel):
        self.randomize_params(env_ids)
        self.euler_angles[env_ids] = euler_angles
        self.rates[env_ids] = rates
        self.accel[env_ids] = torch.zeros(3, dtype=torch.float32, device=self.device)
        self.linear_pos[env_ids] = linear_pos
        self.linear_vel[env_ids] = linear_vel
        self.linear_accel[env_ids] = linear_accel
        self.overshoot_vel[env_ids] = linear_vel
        self.overshoot_accel[env_ids] = linear_accel
        self.thrust = torch.ones(self.num_envs, dtype=torch.float32, device=self.device, requires_grad=False) * 9.81

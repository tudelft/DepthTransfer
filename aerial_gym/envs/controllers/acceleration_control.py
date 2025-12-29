import torch
import pytorch3d.transforms as p3d_transforms
from aerial_gym.utils.math import * 
from aerial_gym.envs.reference.reference_base import ReferenceBase
from aerial_gym.envs.controllers.low_pass_filter import LowPassFilter

class LeeAccelerationController:
    def __init__(self, inertia, control_config, num_envs, device):
        self.device = device
        self.inertia = inertia
        self.control_config = control_config
        if self.control_config.use_reference_model:
            self.Kp_rates = torch.tensor(control_config.kp_rates, dtype=torch.float32, device=self.device, requires_grad=False).tile((num_envs, 1))
            self.Kd_rates = torch.tensor(control_config.kd_rates, dtype=torch.float32, device=self.device, requires_grad=False).tile((num_envs, 1))
            self.Ki_rates = torch.tensor(control_config.ki_rates, dtype=torch.float32, device=self.device, requires_grad=False).tile((num_envs, 1))
            self.filter_sampling_frequency = control_config.filter_sampling_frequency
            self.filter_cutoff_frequency = control_config.filter_cutoff_frequency
            self.filter_initial_value = control_config.filter_initial_value

            self.Kp_acc = torch.tensor(control_config.kp_acc, dtype=torch.float32, device=self.device, requires_grad=False).tile((num_envs, 1))
            self.Kd_acc = torch.tensor(control_config.kd_acc, dtype=torch.float32, device=self.device, requires_grad=False).tile((num_envs, 1))
            self.Krf_acc = torch.tensor(control_config.krf_acc, dtype=torch.float32, device=self.device, requires_grad=False).tile((num_envs, 1))
            self.KR = torch.tensor(control_config.kR, dtype=torch.float32, device=self.device, requires_grad=False).tile((num_envs, 1))

            self.euler_setpoints = torch.zeros((num_envs, 3), device=self.device, requires_grad=False)
            self.filter = LowPassFilter(num_envs, 3, self.filter_cutoff_frequency, self.filter_sampling_frequency, self.filter_initial_value, self.device)
            self.int_err_ang_vel = torch.zeros(num_envs, 3, device=self.device, requires_grad=False)
            self.int_max = torch.tensor(control_config.iterm_lim, dtype=torch.float32, device=self.device, requires_grad=False).tile((num_envs, 1))
            self.T_att = torch.zeros((num_envs, 3, 3), device=self.device, requires_grad=False)
            self.T_att[:, 0, 0] = control_config.kR[0]
            self.T_att[:, 1, 1] = control_config.kR[1]
            self.T_att[:, 2, 2] = control_config.kR[2]
        else:
            self.K_rot_tensor_current = torch.tensor(control_config.kR, dtype=torch.float32, device=self.device).tile((num_envs, 1))
            self.K_angvel_tensor_current = torch.tensor(control_config.kOmega, dtype=torch.float32, device=self.device).tile((num_envs, 1))
    
    def randomize_params(self, env_ids=None):
        pass
        if env_ids is None:
            env_ids = torch.arange(self.K_vel_tensor.shape[0])
        self.K_rot_tensor_current[env_ids] = (self.K_rot_tensor + ((torch.rand_like(self.K_rot_tensor) - 0.5) * 2) * 0.8)[env_ids]
        self.K_angvel_tensor_current[env_ids] = (self.K_angvel_tensor + ((torch.rand_like(self.K_angvel_tensor) - 0.5) * 2) * 0.2)[env_ids]
        # # here pay special attention to yaw_rate. TODO fix this to not be particular

    def get_euler_setpoints_from_acc(self, robot_state, command_actions, reference: ReferenceBase, dt):
        # perform calculation for transformation matrices
        # rotation_matrices = p3d_transforms.quaternion_to_matrix(
        #     robot_state[:, [6, 3, 4, 5]])
        
        # Compute desired accelerations
        accel_command = command_actions[:, :3]
        accel_command[:, 2] += 1
        forces_command = accel_command
        # thrust_command = torch.sum(forces_command * rotation_matrices[:, :, 2], dim=1) * 9.81
        thrust_command = forces_command[:, 2] * 9.81
        c_phi_s_theta = forces_command[:, 0]
        s_phi = -forces_command[:, 1]
        c_phi_c_theta = forces_command[:, 2]

        pitch_setpoint = torch.atan2(c_phi_s_theta, c_phi_c_theta)
        roll_setpoint = torch.atan2(s_phi, torch.sqrt(
            c_phi_c_theta**2 + c_phi_s_theta**2))
        yaw_setpoint = self.euler_setpoints[:, 2] + command_actions[:, 3] * dt

        self.euler_setpoints[:, 0] = roll_setpoint
        self.euler_setpoints[:, 1] = pitch_setpoint
        self.euler_setpoints[:, 2] = yaw_setpoint
        return self.euler_setpoints, thrust_command
    
    def update_command(self, robot_state, reference: ReferenceBase):
        rotation_matrices = p3d_transforms.quaternion_to_matrix(
            robot_state[:, [6, 3, 4, 5]])
        rotation_matrices_transpose = torch.transpose(rotation_matrices, 1, 2)

        pos_err = reference.linear_pos - robot_state[:, :3]
        vel_err = reference.overshoot_vel - robot_state[:, 7:10]
        accel_command = (self.Kp_acc * pos_err + self.Kd_acc * vel_err + self.Krf_acc * reference.overshoot_accel
                        + torch.tensor([0, 0, 9.81], dtype=torch.float32, device=self.device))
        # print(pos_err[0, :], vel_err[0, :])
        thrust_command = torch.sum(accel_command * rotation_matrices[:, :, 2], dim=1)
        # thrust_command = reference.thrust

        vehicle_frame_euler = torch.zeros_like(reference.euler_angles)
        vehicle_frame_euler[:, 2] = reference.euler_angles[:, 2]
        vehicle_frame_transforms = p3d_transforms.euler_angles_to_matrix(
            vehicle_frame_euler[:, [2, 1, 0]], "ZYX")
        vehicle_frame_transforms_transpose = torch.transpose(vehicle_frame_transforms, 1, 2)

        accel_command = torch.bmm(vehicle_frame_transforms_transpose, accel_command.unsqueeze(2)).squeeze(2)
        
        c_phi_s_theta = accel_command[:, 0]
        s_phi = -accel_command[:, 1]
        c_phi_c_theta = accel_command[:, 2]
        euler_setpoints = torch.zeros_like(reference.euler_angles)
        euler_setpoints[:, 0] = torch.atan2(s_phi, torch.sqrt(c_phi_c_theta**2 + c_phi_s_theta**2))
        euler_setpoints[:, 1] = torch.atan2(c_phi_s_theta, c_phi_c_theta)
        euler_setpoints[:, 2] = reference.euler_angles[:, 2]

        q_des = p3d_transforms.matrix_to_quaternion(p3d_transforms.euler_angles_to_matrix(euler_setpoints[:, [2, 1, 0]], "ZYX"))
        rate_cmd = self.tiltPrioritizedControl(robot_state[:, [6, 3, 4, 5]], q_des)
        # torque command
        omega_wolrd = robot_state[:, 10:13]
        omega_body = torch.bmm(rotation_matrices_transpose, omega_wolrd.unsqueeze(2)).squeeze(2)
        omega_f = self.filter.add(omega_body)
        omega_f_dot = self.filter.derivative()

        omega_error = rate_cmd - omega_f

        # integral of error rate, and limit the integral amount
        self.int_err_ang_vel += omega_error
        self.int_err_ang_vel.clamp_(-self.int_max, self.int_max)
        alpha_cmd = self.Kp_rates * omega_error - self.Kd_rates * omega_f_dot + self.Ki_rates * self.int_err_ang_vel
        torque = torch.bmm(self.inertia, alpha_cmd.unsqueeze(2)).squeeze(2) + torch.cross(omega_body, torch.bmm(self.inertia, omega_body.unsqueeze(2)).squeeze(2), dim=1)
        return thrust_command, torque, omega_error
    
    def tiltPrioritizedControl(self, q, q_des):
        q_e = p3d_transforms.quaternion_multiply(p3d_transforms.quaternion_invert(q), q_des)
        tmp = torch.zeros_like(self.euler_setpoints)
        tmp[:, 0] = q_e[:, 0] * q_e[:, 1] - q_e[:, 2] * q_e[:, 3]
        tmp[:, 1] = q_e[:, 0] * q_e[:, 2] + q_e[:, 1] * q_e[:, 3]
        tmp[:, 2] = q_e[:, 3]
        tmp[:, 2] = torch.where(q_e[:, 0] > 0, tmp[:, 2], -tmp[:, 2])
        tmp_a = 2.0 / torch.sqrt(q_e[:, 0]**2 + q_e[:, 3]**2)
        tmp_a = tmp_a.unsqueeze(1).repeat(1, 3)
        rate_cmd = tmp_a * torch.bmm(self.T_att, tmp.unsqueeze(2)).squeeze(2)
        return rate_cmd

    def __call__(self, robot_state, command_actions):
        """
        Lee acceleration controller
        :param robot_state: tensor of shape (num_envs, 13) with state of the robot
        :param command_actions: tensor of shape (num_envs, 4) with desired velocity setpoint in vehicle frame and yaw_rate command in vehicle frame
        :return: m*g normalized thrust and interial normalized torques
        """
        # perform calculation for transformation matrices
        rotation_matrices = p3d_transforms.quaternion_to_matrix(
            robot_state[:, [6, 3, 4, 5]])
        rotation_matrix_transpose = torch.transpose(rotation_matrices, 1, 2)
        euler_angles = p3d_transforms.matrix_to_euler_angles(
            rotation_matrices, "ZYX")[:, [2, 1, 0]]

        # Compute desired accelerations
        accel_command = command_actions[:, :3]
        accel_command[:, 2] += 1

        forces_command = accel_command
        thrust_command = torch.sum(forces_command * rotation_matrices[:, :, 2], dim=1)

        c_phi_s_theta = forces_command[:, 0]
        s_phi = -forces_command[:, 1]
        c_phi_c_theta = forces_command[:, 2]

        # Calculate euler setpoints
        pitch_setpoint = torch.atan2(c_phi_s_theta, c_phi_c_theta)
        roll_setpoint = torch.atan2(s_phi, torch.sqrt(
            c_phi_c_theta**2 + c_phi_s_theta**2))
        yaw_setpoint = euler_angles[:, 2]


        euler_setpoints = torch.zeros_like(euler_angles)
        euler_setpoints[:, 0] = roll_setpoint
        euler_setpoints[:, 1] = pitch_setpoint
        euler_setpoints[:, 2] = yaw_setpoint

        # perform computation on calculated values
        rotation_matrix_desired = p3d_transforms.euler_angles_to_matrix(
            euler_setpoints[:, [2, 1, 0]], "ZYX")
        rotation_matrix_desired_transpose = torch.transpose(
            rotation_matrix_desired, 1, 2)
        rot_err_mat = torch.bmm(rotation_matrix_desired_transpose, rotation_matrices) - \
            torch.bmm(rotation_matrix_transpose, rotation_matrix_desired)
        rot_err = 0.5 * compute_vee_map(rot_err_mat)

        rotmat_euler_to_body_rates = torch.zeros_like(rotation_matrices)

        s_pitch = torch.sin(euler_angles[:, 1])
        c_pitch = torch.cos(euler_angles[:, 1])

        s_roll = torch.sin(euler_angles[:, 0])
        c_roll = torch.cos(euler_angles[:, 0])

        rotmat_euler_to_body_rates[:, 0, 0] = 1.0
        rotmat_euler_to_body_rates[:, 1, 1] = c_roll
        rotmat_euler_to_body_rates[:, 0, 2] = -s_pitch
        rotmat_euler_to_body_rates[:, 2, 1] = -s_roll
        rotmat_euler_to_body_rates[:, 1, 2] = s_roll * c_pitch
        rotmat_euler_to_body_rates[:, 2, 2] = c_roll * c_pitch

        euler_angle_rates = torch.zeros_like(euler_angles)
        euler_angle_rates[:, 2] = command_actions[:, 3]

        omega_desired_body = torch.bmm(rotmat_euler_to_body_rates, euler_angle_rates.unsqueeze(2)).squeeze(2)

        # omega_des_body = [0, 0, yaw_rate] ## approximated body_rate as yaw_rate
        # omega_body = R_t @ omega_world
        # angvel_err = omega_body - R_t @ R_des @ omega_des_body
        # Refer to Lee et. al. (2010) for details (https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=5717652)

        desired_angvel_body_frame = torch.bmm(rotation_matrix_transpose, torch.bmm(
            rotation_matrix_desired, omega_desired_body.unsqueeze(2))).squeeze(2)

        actual_angvel_body_frame = torch.bmm(
            rotation_matrix_transpose, robot_state[:, 10:13].unsqueeze(2)).squeeze(2)
        angvel_err = actual_angvel_body_frame - desired_angvel_body_frame

        torque = - self.K_rot_tensor_current * rot_err - self.K_angvel_tensor_current * angvel_err + torch.cross(robot_state[:, 10:13],robot_state[:, 10:13], dim=1)
        return thrust_command, torque
    
    def reset(self, eulers, env_ids=None):
        self.filter.reset(env_ids)
        self.euler_setpoints[env_ids] = eulers
        self.int_err_ang_vel[env_ids] = torch.zeros(3, device=self.device, requires_grad=False)
        return True

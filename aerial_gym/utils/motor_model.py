import torch
from isaacgym.torch_utils import tensor_clamp


class MotorModel():
    def __init__(self, num_envs, dt, config, device="cuda:0"):
        self.num_envs = num_envs
        self.dt = dt
        self.cfg = config
        self.device = device
        self.current_motor_thrust = torch.zeros((self.num_envs, self.cfg.num_motors_per_robot), device=self.device)
        self.output_wrench = torch.zeros((self.num_envs, 6), device=self.device)

        self.num_motors_per_robot = self.cfg.num_motors_per_robot
        self.motor_time_constants = self.cfg.motor_time_constant_min + torch.rand((self.num_envs, self.num_motors_per_robot), device=self.device) * (self.cfg.motor_time_constant_max - self.cfg.motor_time_constant_min)
        self.motor_thrust_rate = torch.zeros((self.num_envs, self.num_motors_per_robot), device=self.device)
        self.max_thrust = self.cfg.max_thrust
        self.min_thrust = self.cfg.min_thrust
        
        self.total_force = torch.zeros((self.num_envs, 3), device=self.device)
        self.total_torque = torch.zeros((self.num_envs, 3), device=self.device)

        self.force_torque_allocation_matrix = torch.tensor(self.cfg.allocation_matrix, device=self.device, dtype=torch.float32)

        alloc_matrix_rank = torch.linalg.matrix_rank(self.force_torque_allocation_matrix)
        if alloc_matrix_rank < 6:
            print("WARNING: allocation matrix is not full rank. Rank: {}".format(alloc_matrix_rank))

        self.force_torque_allocation_matrix = self.force_torque_allocation_matrix.expand(self.num_envs, -1, -1)

        self.inv_force_torque_allocation_matrix = torch.linalg.pinv(torch.tensor(self.cfg.allocation_matrix, device=self.device, dtype=torch.float32)).expand(self.num_envs, -1, -1)
    
    def update_reference_wrench(self, ref_wrench):
        ref_motor_thrusts = torch.bmm(self.inv_force_torque_allocation_matrix, ref_wrench.unsqueeze(-1)).squeeze(-1)
        self.update_motor_thrusts(ref_motor_thrusts)
        self.output_wrench[:] = torch.bmm(self.force_torque_allocation_matrix, self.current_motor_thrust.unsqueeze(-1)).squeeze(-1)
        return self.output_wrench
    
    def update_motor_thrusts(self, ref_thrust):
        ref_thrust = torch.clamp(ref_thrust, self.min_thrust, self.max_thrust)
        self.motor_thrust_rate[:] = (1./self.motor_time_constants) * (ref_thrust - self.current_motor_thrust)
        self.motor_thrust_rate[:] = torch.clamp(self.motor_thrust_rate, -self.cfg.max_thrust_rate, self.cfg.max_thrust_rate)
        self.current_motor_thrust[:] = self.current_motor_thrust + self.dt * self.motor_thrust_rate
        return self.current_motor_thrust
    
    def reset_idx(self, env_ids):
        self.motor_time_constants[env_ids] = self.cfg.motor_time_constant_min + torch.rand((len(env_ids), self.num_motors_per_robot), device=self.device) * (self.cfg.motor_time_constant_max - self.cfg.motor_time_constant_min)
        self.current_motor_thrust[env_ids] = 0 #torch.zeros((self.num_envs, self.num_motors_per_robot), device=self.device)

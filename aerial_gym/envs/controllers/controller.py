import torch

from aerial_gym.envs.controllers.attitude_control import LeeAttitudeContoller
from aerial_gym.envs.controllers.position_control import LeePositionController
from aerial_gym.envs.controllers.velocity_control import LeeVelocityController
from aerial_gym.envs.controllers.acceleration_control import LeeAccelerationController
from aerial_gym.envs.controllers.velocity_steeing_angle_controller import LeeVelocitySteeringAngleController


control_class_dict = {
    "lee_position_control": LeePositionController,
    "lee_velocity_control": LeeVelocityController,
    "lee_attitude_control": LeeAttitudeContoller,
    "lee_acceleration_control": LeeAccelerationController,
    "lee_velocity_steering_angle_control": LeeVelocitySteeringAngleController
}

class Controller:
    def __init__(self, inertia, control_config, num_envs, device):
        self.control_config = control_config
        self.device = device
        self.controller_name = control_config.controller
        self.kP = torch.tensor(control_config.kP, dtype=torch.float32, device=self.device).tile((num_envs, 1))
        self.kV = torch.tensor(control_config.kV, dtype=torch.float32, device=self.device).tile((num_envs, 1))
        self.kOmega = torch.tensor(control_config.kOmega, dtype=torch.float32, device=self.device).tile((num_envs, 1))
        self.kR = torch.tensor(control_config.kR, dtype=torch.float32, device=self.device).tile((num_envs, 1))
        self.filter_sampling_frequency = control_config.filter_sampling_frequency
        self.filter_cutoff_frequency = control_config.filter_cutoff_frequency
        self.filter_initial_value = control_config.filter_initial_value

        self.scale_input = torch.tensor(control_config.scale_input, dtype=torch.float32, device=self.device)

        if self.control_config.controller not in control_class_dict:
            raise ValueError("Invalid controller name: {}".format(self.control_config.controller))
        else:
            if control_class_dict[self.controller_name] is LeeAttitudeContoller:
                self.controller = LeeAttitudeContoller(self.kR, self.kOmega)
            elif control_class_dict[self.controller_name] is LeePositionController:
                self.controller = LeePositionController(self.kP, self.kV, self.kR, self.kOmega)
            elif control_class_dict[self.controller_name] is LeeVelocityController:
                self.controller = LeeVelocityController(self.kV, self.kR, self.kOmega)
            elif control_class_dict[self.controller_name] is LeeAccelerationController:
                self.controller = LeeAccelerationController(inertia, control_config, num_envs, device)
            elif control_class_dict[self.controller_name] is LeeVelocitySteeringAngleController:
                self.controller = LeeVelocitySteeringAngleController(self.kV, self.kR, self.kOmega)
            else:
                raise ValueError("Invalid controller name: {}".format(self.control_config.controller))
            
    def randomize_params(self, env_ids):
        self.controller.randomize_params(env_ids)

    def get_euler_setpoints_from_acc(self, robot_state, command_actions, reference, dt):
        return self.controller.get_euler_setpoints_from_acc(robot_state, command_actions, reference, dt)
    
    def reset(self, eulers, env_ids):
        return self.controller.reset(eulers, env_ids)

    def update_command(self, robot_state, reference):
        return self.controller.update_command(robot_state, reference)

    def __call__(self, robot_state, command_actions):
        # check if controller name matches class in dict
        scaled_input = command_actions * self.scale_input
        return self.controller(robot_state, scaled_input)

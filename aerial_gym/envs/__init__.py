from aerial_gym.envs.base.mavrl_task_config import MAVRLTaskCfg
from aerial_gym.envs.base.zoo_task_config import ZooTaskCfg
from aerial_gym.envs.base.mavrl_task_agile_config import MAVRLTaskAgileCfg
from aerial_gym.envs.base.mavrl_task import MAVRLTask
from aerial_gym.envs.base.drone_racing_task import DroneRacingTask
from aerial_gym.envs.base.drone_racing_task_config import DroneRacingTaskCfg
import os

from aerial_gym.utils.task_registry import task_registry

task_registry.register("mavrl_task", MAVRLTask, MAVRLTaskCfg())
task_registry.register("mavrl_zoo", MAVRLTask, ZooTaskCfg())
task_registry.register("drone_racing", DroneRacingTask, DroneRacingTaskCfg())
task_registry.register("mavrl_task_agile", MAVRLTask, MAVRLTaskAgileCfg())
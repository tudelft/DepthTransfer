import torch
import pickle as pkl
import numpy as np


class ORACLERolloutLogger():
    def __init__(self, num_envs, const_action_steps, depth_image_steps, state_action_steps, device, state_count, action_count, depth_image_shape, pickle_prefix):
        self.num_envs = num_envs
        self.const_action_steps = const_action_steps
        self.depth_image_steps = depth_image_steps
        self.state_action_steps = state_action_steps
        self.device = device
        self.state_count = state_count
        self.action_count = action_count
        self.pickle_prefix = pickle_prefix
        self.pickle_counter = 0
        self.depth_image_shape = depth_image_shape
        self.init_tensors()
    
    def init_tensors(self):
        self.state_tensors = np.zeros(
            (self.num_envs, self.const_action_steps, self.depth_image_steps, self.state_action_steps, self.state_count))
        self.action_tensors = np.zeros(
            (self.num_envs, self.const_action_steps, self.depth_image_steps, self.state_action_steps, self.action_count))
        self.collision_tensors = np.zeros(
            (self.num_envs, self.const_action_steps, self.depth_image_steps, self.state_action_steps))
        self.external_force_tensors = np.zeros(
            (self.num_envs, self.const_action_steps, self.depth_image_steps, self.state_action_steps, 3))
        self.depth_image_tensors = np.zeros(
            (self.num_envs, self.const_action_steps, self.depth_image_steps, self.depth_image_shape[0], self.depth_image_shape[1]), dtype=np.uint16)
        
    def reset_tensors(self):
        self.state_tensors[:] = 0.0
        self.action_tensors[:] = 0.0
        self.collision_tensors[:] = 0.0
        self.external_force_tensors[:] = 0.0
        self.depth_image_tensors[:] = 0.0
    
    def set_image_tensors(self, const_action_step, depth_image_step, image_tensors):
        self.depth_image_tensors[:, const_action_step, depth_image_step] = (1000.0*np.clip(image_tensors.cpu().numpy(), 0.0, 1.0)).astype(np.uint16)
    
    def update_tensors_per_timestep(self, const_action_step, depth_image_step, state_action_step, state_obs, collision_label, external_force_label, action):
        self.state_tensors[:, const_action_step, depth_image_step, state_action_step] = state_obs.cpu().numpy()
        self.collision_tensors[:, const_action_step, depth_image_step, state_action_step] = collision_label.cpu().numpy()
        self.external_force_tensors[:, const_action_step, depth_image_step, state_action_step] = external_force_label.cpu().numpy()
        self.action_tensors[:, const_action_step, depth_image_step, state_action_step] = action.cpu().numpy()
    
    def save_rollout_as_pickle(self):
        print("Saving rollout as pickle")
        rollout_dict = {
            "rollout_metadata": {
                "num_envs": self.num_envs,
                "const_action_steps": self.const_action_steps,
                "depth_image_steps": self.depth_image_steps,
                "state_action_steps": self.state_action_steps,
                "state_count": self.state_count,
                "action_count": self.action_count,
                "device": self.device,
                "depth_image_shape": self.depth_image_shape
            },
            "actions": self.action_tensors,
            "collisions": self.collision_tensors,
            "observations": self.state_tensors,
            "forces": self.external_force_tensors,
            "depth_images": self.depth_image_tensors
        }

        print("Writing to pickle file")
        # write dict as pickle file

        with open(self.pickle_prefix + "_" + str(self.pickle_counter) + ".pkl", "wb") as f:
            pkl.dump(rollout_dict, f)

        print("Done writing to pickle file")
        self.pickle_counter += 1

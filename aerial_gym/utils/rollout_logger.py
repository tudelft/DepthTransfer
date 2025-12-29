import torch
import pickle as pkl


class ORACLERolloutLogger():
    def __init__(self, num_envs=128, states=13, rollout_timesteps=25, device="cuda:0", terminate_on_collision=False, pickle_prefix="sample_experiment"):
        self.num_envs = num_envs
        self.rollout_timesteps = rollout_timesteps
        self.terminate_on_collision = terminate_on_collision
        self.device = device
        self.state_count = states
        self.num_updates = 0
        self.pickle_prefix = pickle_prefix
        self.pickle_counter = 0
        self.init_tensors()

    def init_tensors(self):
        print((self.num_envs, self.rollout_timesteps, 4))
        self.actions = torch.zeros(
            (self.num_envs, self.rollout_timesteps, 4), device=self.device)
        self.collision_label = torch.zeros(
            (self.num_envs, self.rollout_timesteps), device=self.device)
        self.external_force_label = torch.zeros(
            (self.num_envs, self.rollout_timesteps, 3), device=self.device)
        self.state_obs = torch.zeros(
            (self.num_envs, self.rollout_timesteps, self.state_count))

    def reset_tensors(self):
        self.actions.zero_()
        self.collision_label.zero_()
        self.external_force_label.zero_()
        self.state_obs.zero_()

    def set_image_tensors(self, image_tensors):
        self.depth_images = image_tensors

    def update_tensors_per_timestep(self, timestep, action, state_obs, collision_label=0, external_force_label=0):
        if timestep >= self.rollout_timesteps:
            print("Number of updates are greater than timesteps for which this class is designed. Please see what's wrong")
            return
        self.actions[:, timestep] = action
        self.collision_label[:, timestep] = collision_label
        self.state_obs[:, timestep] = state_obs
        self.external_force_label[:, timestep] = external_force_label
        self.num_updates += 1
        if self.num_updates == self.rollout_timesteps:
            print("Number of updates are greater than timesteps for which this class is designed. Please see what's wrong")

    def convert_to_numpy_arrays(self):
        self.np_actions = self.actions.cpu().numpy()
        self.np_collision_label = self.collision_label.cpu().numpy()
        self.np_external_force_label = self.external_force_label.cpu().numpy()
        self.np_state_obs = self.state_obs.cpu().numpy()
        try:
            self.np_depth_images = self.depth_images.cpu().numpy()
        except:
            print("Depth images not set. Not converting to numpy array")
            self.np_depth_images = None

    def save_rollout_as_pickle(self):
        print("Saving rollout as pickle")
        self.convert_to_numpy_arrays()
        print("Converted to numpy arrays")
        rollout_dict = {
            "rollout_metadata": {
                "rollout_timesteps": self.rollout_timesteps,
                "num_envs": self.num_envs,
                "num_states": self.state_count,
                "num_actions": self.actions.shape[-1],
                "device": self.device,
                "terminate_on_collision": self.terminate_on_collision
            },
            "actions": self.np_actions,
            "collisions": self.np_collision_label,
            "observations": self.np_state_obs,
            "forces": self.np_external_force_label,
            "depth_images": self.np_depth_images
        }

        print("Writing to pickle file")
        # write dict as pickle file

        with open(self.pickle_prefix + "_" + str(self.pickle_counter) + ".pkl", "wb") as f:
            pkl.dump(rollout_dict, f)

        print("Done writing to pickle file")
        self.pickle_counter += 1

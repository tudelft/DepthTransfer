from typing import Any
import torch

distribution_type_to_function_mapping = {
    "bernoulli": torch.distributions.bernoulli.Bernoulli,
    "binomial": torch.distributions.binomial.Binomial,
    "normal": torch.distributions.normal.Normal,
    "poisson": torch.distributions.poisson.Poisson,
    "uniform": torch.distributions.uniform.Uniform,
}


class Sampler:
    # init function with a parameter list based on type of noise
    def __init__(self, enable, distribution, dist_params, transform_after_sampling, size=None, device="cuda:0"):
        self.enable = enable
        self.distribution = distribution
        self.dist_params = dist_params
        self.transform_after_sampling = transform_after_sampling
        self.device = device
        self.enable_multiplier = 1.0 if self.enable else 0.0
        if self.device is not "cpu":
            # create a tensor out of the parameters
            for key, val in self.dist_params.items():
                self.dist_params[key] = torch.tensor(val, device=self.device)
        try:
            self.distribution_function = distribution_type_to_function_mapping[distribution]
        except KeyError:
            raise ValueError(f"Noise type {distribution} not supported.")

        self.distribution = self.distribution_function(**self.dist_params)

        if size is not None:
            self.distribution = self.distribution.expand(size)

        # define lambda as transform function
        if distribution == "normal":
            self.transform = transform_standard_normal_to_normal
        elif distribution == "uniform":
            self.transform = transform_uniform_to_value
        else:
            self.transform = lambda x: x

    # sample function to sample from the distribution
    def sample(self, *args):
        if self.transform_after_sampling == False:
            return self.enable_multiplier*self.distribution.sample()
        else:
            return self.enable_multiplier*self.transform(self.distribution.sample(), *args)

# torch jit

# @torch.jit.script
def transform_standard_normal_to_normal( sample, loc=0.0, scale=1.0):
    return sample * scale + loc


# @torch.jit.script
def transform_uniform_to_value(sample, low=0.0, high=1.0):
    return sample * (high - low) + low

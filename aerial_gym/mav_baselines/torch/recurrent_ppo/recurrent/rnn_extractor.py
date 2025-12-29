from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from typing import Dict, List, Tuple, Type, Union

from gymnasium import spaces
import torch as th
from torch import nn
import time

from stable_baselines3.common.preprocessing import get_flattened_obs_dim, is_image_space
from stable_baselines3.common.type_aliases import TensorDict
from stable_baselines3.common.utils import get_device

from aerial_gym.mav_baselines.torch.controlNet.ldm.modules.diffusionmodules.model import Encoder as ResnetEncoder
from aerial_gym.mav_baselines.torch.controlNet.ldm.modules.diffusionmodules.model import Decoder as ResnetDecoder
from aerial_gym.mav_baselines.torch.controlNet.ldm.modules.distributions.distributions import DiagonalGaussianDistribution


class Encoder(BaseFeaturesExtractor):
    def __init__(self, observation_space: spaces.Box, features_dim: int = 64):

        image_observation_space = None
        if isinstance(observation_space, spaces.Dict):
            for key, subspace in observation_space.spaces.items():
                if is_image_space(subspace):
                    image_observation_space = subspace
        elif is_image_space(observation_space):
            image_observation_space = observation_space
        else:
            raise ValueError("No image space in the observation space")
        super(Encoder, self).__init__(image_observation_space, features_dim)
        n_input_channels = image_observation_space.shape[0]
        self.conv1 = nn.Conv2d(n_input_channels, 8, kernel_size=4, stride=2)
        self.conv2 = nn.Conv2d(8, 16, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(16, 32, kernel_size=4, stride=2)
        self.conv4 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv5 = nn.Conv2d(64, 128, kernel_size=4, stride=2)
        self.conv6 = nn.Conv2d(128, 256, kernel_size=4, stride=2)
        # Compute shape by doing one forward pass
        # with th.no_grad():
        #     n_flatten = self.conv6(self.conv5(self.conv4(self.conv3(self.conv2(self.conv1(th.as_tensor(image_observation_space.sample()[None][:, :1, :, :]).float())))))).shape
        self.linear = nn.Linear(2*2*256, features_dim)
        self.fc_logsigma = nn.Linear(2*2*256, features_dim)

    def forward(self, observations: th.Tensor) -> th.Tensor:
        x = nn.functional.relu(self.conv1(observations))
        x = nn.functional.relu(self.conv2(x))
        x = nn.functional.relu(self.conv3(x))
        x = nn.functional.relu(self.conv4(x))
        x = nn.functional.relu(self.conv5(x))
        x = nn.functional.relu(self.conv6(x))
        mu = self.linear(x.view(observations.size(0), -1))
        # logsigma = self.fc_logsigma(x.view(observations.size(0), -1))
        # sigma = logsigma.exp()
        # eps = th.randn_like(sigma)
        # z = eps.mul(sigma).add_(mu)
        return mu
    
class Decoder(nn.Module):
    def __init__(self, observation_space: spaces.Box, lstm_hidden_dim: int = 64) -> None:
        super(Decoder, self).__init__()
        if isinstance(observation_space, spaces.Dict):
            for key, subspace in observation_space.spaces.items():
                if is_image_space(subspace):
                    image_observation_space = subspace
            n_input_channels = image_observation_space.shape[0]
        else:
            n_input_channels = 256
        self.fc = nn.Linear(lstm_hidden_dim, 2*2*256)
        self.deconv1 = nn.ConvTranspose2d(2*2*256, 128, kernel_size=5, stride=2)
        self.deconv2 = nn.ConvTranspose2d(128, 64, kernel_size=5, stride=2)
        self.deconv3 = nn.ConvTranspose2d(64, 32, kernel_size=6, stride=2)
        self.deconv4 = nn.ConvTranspose2d(32, 16, kernel_size=4, stride=2)
        self.deconv5 = nn.ConvTranspose2d(16, 8, kernel_size=5, stride=2)
        self.deconv6 = nn.ConvTranspose2d(8, n_input_channels, kernel_size=4, stride=2)

    def forward(self, latent)-> th.Tensor:
        x = nn.functional.relu(self.fc(latent))
        x = x.unsqueeze(-1).unsqueeze(-1)
        x = nn.functional.relu(self.deconv1(x))
        x = nn.functional.relu(self.deconv2(x))
        x = nn.functional.relu(self.deconv3(x))
        x = nn.functional.relu(self.deconv4(x))
        x = nn.functional.relu(self.deconv5(x))
        reconstruction = th.sigmoid(self.deconv6(x))
        return reconstruction

    
class Encoder320(BaseFeaturesExtractor):
    def __init__(self, observation_space: spaces.Box, features_dim: int=64, ngf: int=64):

        image_observation_space = None
        if isinstance(observation_space, spaces.Dict):
            for key, subspace in observation_space.spaces.items():
                if is_image_space(subspace):
                    image_observation_space = subspace
        elif is_image_space(observation_space):
            image_observation_space = observation_space
        else:
            raise ValueError("No image space in the observation space")
        super(Encoder320, self).__init__(image_observation_space, features_dim)
        n_input_channels = image_observation_space.shape[0]
        self.encoder = nn.Sequential(
            nn.Conv2d(n_input_channels, ngf, kernel_size=4, stride=2, padding=1),
            nn.ReLU(True),
            nn.Conv2d(ngf, ngf*2, kernel_size=4, stride=2, padding=1),
            nn.ReLU(True),
            nn.Conv2d(ngf*2, ngf*4, kernel_size=4, stride=2, padding=1),
            nn.ReLU(True),
            nn.Conv2d(ngf*4, ngf*8, kernel_size=4, stride=2, padding=1),
            nn.ReLU(True),
            nn.Conv2d(ngf*8, ngf*8, kernel_size=4, stride=2, padding=1),
            nn.ReLU(True),
            nn.Conv2d(ngf*8, ngf*8, kernel_size=(5, 4), stride=2, padding=1),
            nn.ReLU(True),
            nn.Conv2d(ngf*8, ngf*8, kernel_size=(3, 5), stride=2),
            nn.ReLU(True)
        )
        self.mu_linear = nn.Linear(ngf*8, features_dim)
        self.logvar_linear = nn.Linear(ngf*8, features_dim)

    def forward(self, observations: th.Tensor, sample: bool=True) -> th.Tensor:
        x = self.encoder(observations)
        x = x.view(x.size(0), -1)
        mu = self.mu_linear(x)
        logvar = self.logvar_linear(x)
        if sample:
            z = th.randn_like(mu).mul(th.exp(logvar * 0.5)).add_(mu)
            return z
        else:
            return mu

class Decoder320(nn.Module):
    def __init__(self, observation_space: spaces.Box, features_dim: int=64, ngf: int=64) -> None:
        super(Decoder320, self).__init__()
        if isinstance(observation_space, spaces.Dict):
            for key, subspace in observation_space.spaces.items():
                if is_image_space(subspace):
                    image_observation_space = subspace
            n_input_channels = image_observation_space.shape[0]
        else:
            n_input_channels = 256

        self.fc_decoder = nn.Linear(features_dim, ngf*8)
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(ngf*8, ngf*8, kernel_size=(3, 5), stride=2),
            nn.ReLU(True),
            nn.ConvTranspose2d(ngf*8, ngf*8, kernel_size=(5, 4), stride=2, padding=1),
            nn.ReLU(True),
            nn.ConvTranspose2d(ngf*8, ngf*8, kernel_size=4, stride=2, padding=1),
            nn.ReLU(True),
            nn.ConvTranspose2d(ngf*8, ngf*4, kernel_size=4, stride=2, padding=1),
            nn.ReLU(True),
            nn.ConvTranspose2d(ngf*4, ngf*2, kernel_size=4, stride=2, padding=1),
            nn.ReLU(True),
            nn.ConvTranspose2d(ngf*2, ngf, kernel_size=4, stride=2, padding=1),
            nn.ReLU(True),
            nn.ConvTranspose2d(ngf, n_input_channels, kernel_size=4, stride=2, padding=1),
            nn.Sigmoid()
        )

    def forward(self, latent: th.Tensor) -> th.Tensor:
        z = self.fc_decoder(latent)
        z = z.unsqueeze(-1).unsqueeze(-1)
        return self.decoder(z)

class EncoderResnet(BaseFeaturesExtractor):
    def __init__(self, observation_space: spaces.Box, features_dim: int=64, ddconfig=None):

        image_observation_space = None
        if isinstance(observation_space, spaces.Dict):
            for key, subspace in observation_space.spaces.items():
                if is_image_space(subspace):
                    image_observation_space = subspace
        elif is_image_space(observation_space):
            image_observation_space = observation_space
        else:
            raise ValueError("No image space in the observation space")
        super(EncoderResnet, self).__init__(image_observation_space, features_dim)
        self.encoder = ResnetEncoder(**ddconfig)

    def forward(self, observations: th.Tensor, sample: bool=False) -> th.Tensor:
        h = self.encoder(observations)
        posterior = DiagonalGaussianDistribution(h)
        return posterior.sample().view(h.size(0), h.size(2)*h.size(3)) if sample else posterior.mean.view(h.size(0), h.size(2)*h.size(3))
    
class DecoderResnet(nn.Module):
    def __init__(self, ddconfig=None) -> None:
        super(DecoderResnet, self).__init__()
        self.decoder = ResnetDecoder(**ddconfig)

    def forward(self, latent: th.Tensor) -> th.Tensor:
        x = latent.view(latent.size(0), 1, 7, 10)
        return self.decoder(x)

class GradientReversalLayer(th.autograd.Function):
    @staticmethod
    def forward(ctx, x, lambda_):
        ctx.lambda_ = lambda_
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return -grad_output * ctx.lambda_, None
    
class Discriminator(nn.Module):
    # MLP for classification
    def __init__(self, input_dim: int, output_dim: int = 2, hidden_dim: int = 512):
        super(Discriminator, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        self.logsoftmax = nn.LogSoftmax(dim=1)
    
    def forward(self, features: th.Tensor, lamda_: float = 1.0) -> th.Tensor:
        x = GradientReversalLayer.apply(features, lamda_)
        x = th.relu(self.fc1(x))
        x = self.logsoftmax(self.fc2(x))
        return x

class MultiExtractor(BaseFeaturesExtractor):
    """
    Combined feature extractor for Dict observation spaces.
    Builds a feature extractor for each key of the space. Input from each space
    is fed through a separate submodule (CNN or MLP, depending on input shape),
    the output features are concatenated and fed through additional MLP network ("combined").

    :param observation_space:
    :param cnn_output_dim: Number of features to output from each CNN submodule(s). Defaults to
        256 to avoid exploding network sizes.
    """

    def __init__(self, observation_space: spaces.Dict, cnn_output_dim: int = 64):
        # TODO we do not know features-dim here before going over all the items, so put something there. This is dirty!
        super(MultiExtractor, self).__init__(observation_space, features_dim=1)

        extractors = {}

        total_concat_size = 0
        for key, subspace in observation_space.spaces.items():
            if is_image_space(subspace):
                extractors[key] = Encoder(subspace, features_dim=cnn_output_dim)
                total_concat_size += cnn_output_dim
                continue
            else:
                # The observation key is a vector, flatten it if needed
                extractors[key] = nn.Flatten()
                total_concat_size += 6

        self.extractors = nn.ModuleDict(extractors)

        # Update the features dim manually
        self._features_dim = total_concat_size

    def forward(self, observations: TensorDict) -> th.Tensor:
        encoded_tensor_list = []

        for key, extractor in self.extractors.items():
            encoded_tensor_list.append(extractor(observations[key]))
        return th.cat(encoded_tensor_list, dim=1)

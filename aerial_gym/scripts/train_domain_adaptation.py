import argparse
from os.path import join, exists
from os import mkdir

import torch
import torch.utils.data
from torch import optim
from torch.nn import functional as F
from torchvision import transforms
from torchvision.utils import save_image
from aerial_gym.mav_baselines.torch.controlNet.cldm.model import create_model, load_state_dict
from aerial_gym.mav_baselines.torch.controlNet.ldm.modules.losses.autoencoder_kl import AutoEncoderLoss
from aerial_gym.mav_baselines.torch.controlNet.ldm.modules.diffusionmodules.model import Encoder as ResnetEncoder
from aerial_gym.mav_baselines.torch.controlNet.ldm.modules.distributions.distributions import DiagonalGaussianDistribution, normal_kl
from torch.utils.tensorboard import SummaryWriter  # Import TensorBoard Writer
from aerial_gym.mav_baselines.torch.models.vae_320 import save_checkpoint, min_pool2d
from aerial_gym.data.loaders import _RolloutDataset, RosbagDataset
from omegaconf import OmegaConf

cuda = torch.cuda.is_available()
torch.manual_seed(123)
torch.backends.cudnn.benchmark = True
device = torch.device("cuda" if cuda else "cpu")

Input_width = 320
Input_height = 224

def parser():
    parser = argparse.ArgumentParser(description='VAE Trainer')
    parser.add_argument('--batch-size', type=int, default=8, metavar='N',
                        help='input batch size for training (default: 32)')
    parser.add_argument('--epochs', type=int, default=300, metavar='N',
                        help='number of epochs to train (default: 1000)')
    parser.add_argument('--logdir', type=str, default='../exp_vae_320', help='Directory where results are logged')
    parser.add_argument('--noreload', action='store_true',
                        help='Best model is not reloaded if specified')
    parser.add_argument('--nosamples', action='store_true',
                        help='Does not save samples during training if specified')
    parser.add_argument("--trial", type=int, default=1, help="PPO trial number")
    parser.add_argument("--iter", type=int, default=100, help="PPO iter number")
    parser.add_argument("--test_only", type=bool, default=False, help="Test only")
    parser.add_argument("--use_kl_loss", type=bool, default=False, help="Use kl loss")
    parser.add_argument("--use_max_pooling", type=bool, default=True, help="Use max pooling")
    return parser

class DomainAdaptation:
    def __init__(self, args) -> None:
        self.args = args
        self.transform_train = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((Input_height, Input_width)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
        ])
        self.transform_test = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((Input_height, Input_width)),
            transforms.ToTensor(),
        ])
        # load synthetic dataset
        self.dataset_train = _RolloutDataset('../saved/dataset_outdoor_env',
                                                self.transform_train, device=device, train=True)
        self.dataset_test = _RolloutDataset('../saved/dataset_outdoor_env',
                                                self.transform_test, device=device, train=False)
        self.train_loader = torch.utils.data.DataLoader(
            self.dataset_train, batch_size=args.batch_size, shuffle=True, num_workers=8)
        self.test_loader = torch.utils.data.DataLoader(
            self.dataset_test, batch_size=args.batch_size, shuffle=True, num_workers=8)

        self.pre_model = create_model('../mav_baselines/torch/controlNet/models/autoencoder.yaml')
        self.pre_model.to(device)
        self.reload_file = join(join(args.logdir, 'vae_resnet_16ch_sim_stereo_depth_08'), 'vae_400.tar')
        if not args.noreload and exists(self.reload_file):
            print("Reloading model from " + self.reload_file)
            state = torch.load(self.reload_file, map_location=device)
            self.pre_model.load_state_dict(state['state_dict'], strict=True)

        encoder_config = '../mav_baselines/torch/controlNet/models/encoder.yaml'
        self.new_model = ResnetEncoder(**(OmegaConf.load(encoder_config)['ddconfig']))
        # self.new_model = create_model('../mav_baselines/torch/controlNet/models/autoencoder.yaml')
        self.new_model.to(device)
        self.optimizer = optim.Adam(self.new_model.parameters(), lr=1e-3)
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=100, gamma=0.5)
        self.avae_dir = join(args.logdir, 'test_domain_adaptation')
        if not exists(self.avae_dir):
            mkdir(self.avae_dir)
            mkdir(join(self.avae_dir, 'samples'))
        self.cur_best = None
        self.writer = SummaryWriter(log_dir=self.avae_dir)

    def train(self, epoch):
        self.new_model.train()
        self.dataset_train.load_next_buffer()
        train_loss = 0
        for batch_idx, data in enumerate(self.train_loader):
            data = data.to(device)
            self.optimizer.zero_grad()
            if self.args.use_max_pooling:
                data = min_pool2d(data, kernel_size=7, stride=1, padding=3)
            with torch.no_grad():
                recon_batch, posteriors = self.pre_model(data)
            latent = self.new_model(recon_batch)
            recon_posterior = DiagonalGaussianDistribution(latent)
            kl_loss = normal_kl(posteriors.mean, posteriors.logvar, recon_posterior.mean, recon_posterior.logvar)
            kl_loss = torch.sum(kl_loss) / kl_loss.shape[0]
            kl_loss.backward()
            train_loss += kl_loss.item()
            self.optimizer.step()
            if batch_idx % 10 == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, batch_idx * len(data), len(self.train_loader.dataset),
                    100. * batch_idx / len(self.train_loader),
                    kl_loss.item() / len(data)))
        avg_train_loss = train_loss / len(self.train_loader.dataset)
        print('====> Epoch: {} Average loss: {:.4f}'.format(epoch, avg_train_loss))
        # Log training loss to TensorBoard
        self.writer.add_scalar('Loss/train', avg_train_loss, epoch)


    def test(self, epoch):
        self.new_model.eval()
        self.dataset_test.load_next_buffer()
        test_loss = 0
        with torch.no_grad():
            for i, data in enumerate(self.test_loader):
                save_image(data, join(self.avae_dir, 'samples', 'test_' + str(epoch) + '.png'))
                data = data.to(device)
                if self.args.use_max_pooling:
                    data = min_pool2d(data, kernel_size=7, stride=1, padding=3)
                    save_image(data, join(self.avae_dir, 'samples', 'max_pooling_' + str(epoch) + '.png'))
                recon_batch, posteriors = self.pre_model(data)
                save_image(recon_batch, join(self.avae_dir, 'samples', 'recon_' + str(epoch) + '.png'))
                latent = self.new_model(recon_batch)
                recon_posterior = DiagonalGaussianDistribution(latent)
                recon_new = self.pre_model.decode(recon_posterior.sample())
                save_image(recon_new, join(self.avae_dir, 'samples', 'recon_new_' + str(epoch) + '.png'))
                kl_loss = normal_kl(posteriors.mean, posteriors.logvar, recon_posterior.mean, recon_posterior.logvar)
                kl_loss = torch.sum(kl_loss) / kl_loss.shape[0]
                test_loss += kl_loss.item()

        avg_test_loss = test_loss / len(self.test_loader.dataset)
        print('====> Test set loss: {:.4f}'.format(avg_test_loss))
        
        # Log test loss to TensorBoard
        self.writer.add_scalar('Loss/test', avg_test_loss, epoch)
        return test_loss

    def learn(self):
        for epoch in range(1, self.args.epochs + 1):
            if self.args.test_only:
                self.test(epoch)
                continue
            self.train(epoch)
            test_loss = self.test(epoch)
            self.scheduler.step()
            # checkpointing
            best_filename = join(self.avae_dir, 'best.tar')
            filename = join(self.avae_dir, 'checkpoint.tar')
            is_best = not self.cur_best or test_loss < self.cur_best
            if is_best:
                self.cur_best = test_loss
            save_checkpoint({
                'epoch': epoch,
                'state_dict': self.new_model.state_dict(),
                'precision': test_loss,
                'optimizer': self.optimizer.state_dict(),
                'scheduler': self.scheduler.state_dict(),
            }, is_best, filename, best_filename)

        # Close the TensorBoard writer
        self.writer.close()



def main():
    args = parser().parse_args()
    ae = DomainAdaptation(args)
    ae.learn()

if __name__ == '__main__':
    main()
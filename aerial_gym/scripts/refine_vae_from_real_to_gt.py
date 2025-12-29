import argparse
from os.path import join, exists
from os import mkdir
import itertools
import numpy as np
import random

import torch
import torch.utils.data
from torch import optim
from torch.nn import functional as F
from torchvision import transforms
from torchvision.utils import save_image
from aerial_gym.mav_baselines.torch.controlNet.cldm.model import create_model, load_state_dict
from aerial_gym.mav_baselines.torch.controlNet.ldm.modules.losses.autoencoder_kl import AutoEncoderLoss
from aerial_gym.mav_baselines.torch.recurrent_ppo.recurrent.rnn_extractor import Discriminator
from torch.utils.tensorboard import SummaryWriter  # Import TensorBoard Writer
from aerial_gym.mav_baselines.torch.models.vae_320 import save_checkpoint, min_pool2d
from aerial_gym.data.loaders import _RolloutDataset, RosbagDataset

cuda = torch.cuda.is_available()
seed = 15
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
np.random.seed(seed)
random.seed(seed)
torch.backends.cudnn.benchmark = True
device = torch.device("cuda" if cuda else "cpu")

Input_width = 320
Input_height = 224

def parser():
    parser = argparse.ArgumentParser(description='VAE Trainer')
    parser.add_argument('--batch-size', type=int, default=32, metavar='N',
                        help='input batch size for training (default: 32)')
    parser.add_argument('--epochs', type=int, default=501, metavar='N',
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
    parser.add_argument("--zoo_task", type=bool, default=False, help="Use zoo task")
    return parser

class DomainAdaptationFromReal2GT:
    def __init__(self, args) -> None:
        self.args = args
        self.transform_train = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((Input_height, Input_width)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
        ])
        if self.args.zoo_task:
            # load target dataset (real data)
            self.dataset_target = RosbagDataset('../datasets', ['/camera/depth/image_rect_raw', '/camera/infra1/image_rect_raw'], 
                                          transform=self.transform_train, with_gray=False, train=True)
            self.dataset_test = RosbagDataset('../datasets', ['/camera/depth/image_rect_raw', '/camera/infra1/image_rect_raw'], 
                                          transform=self.transform_train, with_gray=False, train=False)
            # load source data
            self.dataset_gt = _RolloutDataset('../saved/zoo_dataset_depth_new',
                                                    self.transform_train, device=device, train=True)
        else:
            # load target dataset
            self.dataset_target = RosbagDataset('../datasets', ['/camera/depth/image_rect_raw', '/camera/infra1/image_rect_raw'], 
                                          transform=self.transform_train, with_gray=False, train=True)
            self.dataset_test = RosbagDataset('../datasets', ['/camera/depth/image_rect_raw', '/camera/infra1/image_rect_raw'], 
                                          transform=self.transform_train, with_gray=False, train=False)
            # load source data
            self.dataset_gt = _RolloutDataset('../saved/dataset_outdoor_env',
                                                    self.transform_train, device=device, train=True)
        
        self.target_loader = torch.utils.data.DataLoader(self.dataset_target, batch_size=args.batch_size, shuffle=True, num_workers=16)
        self.gt_loader = torch.utils.data.DataLoader(self.dataset_gt, batch_size=args.batch_size, shuffle=True, num_workers=16)
        self.test_loader = torch.utils.data.DataLoader(self.dataset_test, batch_size=args.batch_size, shuffle=False, num_workers=16)
        
        self.vae_model = create_model('../mav_baselines/torch/controlNet/models/autoencoder.yaml')
        self.vae_model.to(device)
        # freeze decoder for vae_model
        for param in self.vae_model.decoder.parameters():
            param.requires_grad = False
        if self.args.zoo_task:
            ckpt_file = join(args.logdir, 'vae_resnet_16ch_zoo', 'checkpoint.tar')
        else:
            ckpt_file = join(args.logdir, 'vae_resnet_16ch_outdoor', 'checkpoint.tar')
        ckpt = torch.load(ckpt_file, map_location=device)
        self.vae_model.load_state_dict(ckpt['state_dict'])

        self.sr_model = create_model('../mav_baselines/torch/controlNet/models/autoencoder.yaml')
        self.sr_model.to(device)
        # ckpt_file = join(args.logdir, 'vae_resnet_16ch_outdoor', 'checkpoint.tar')
        # ckpt = torch.load(ckpt_file, map_location=device)
        self.sr_model.load_state_dict(ckpt['state_dict'])
        # freeze the sr_model
        for param in self.sr_model.parameters():
            param.requires_grad = False

        self.discriminator = Discriminator(70)
        # freeze the discriminator
        self.discriminator.to(device)
        self.optimizer_disc = optim.Adam(self.discriminator.parameters(), lr=2e-4, betas=(0.5, 0.9))
        self.scheduler_disc = optim.lr_scheduler.StepLR(self.optimizer_disc, step_size=50, gamma=0.5)

        self.optimizer = optim.Adam(self.vae_model.parameters(), lr=6e-4)
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=50, gamma=0.5)
        self.vae_loss = AutoEncoderLoss()

        self.avae_dir = join(args.logdir, 'vae_resnet_real2sim_forest_new')
        if not exists(self.avae_dir):
            mkdir(self.avae_dir)
            mkdir(join(self.avae_dir, 'samples'))

        self.writer = SummaryWriter(log_dir=self.avae_dir)
        self.global_step = 0
        self.it_d, self.it_g = 0, 0

    
    def test(self, epoch):
        self.vae_model.eval()
        self.discriminator.eval()
        self.dataset_test.load_next_buffer()
        test_loss = 0
        with torch.no_grad():
            for i, data in enumerate(self.test_loader):
                # data = self.dict_to_device(data)
                input = data['depth'].to(device)
                save_image(data['depth'], join(self.avae_dir, 'samples', 'real_depth_' + str(epoch) + '_' + str(i) + '.png'))
                if self.args.use_max_pooling:
                    input = min_pool2d(input, kernel_size=3, stride=1, padding=1)
                    save_image(input, join(self.avae_dir, 'samples', 'max_pooling_' + str(epoch) + '_' + str(i) + '.png'))
                recons, posteriors = self.vae_model(input)
                loss, _ = self.vae_loss(input, recons, posteriors)
                save_image(recons, join(self.avae_dir, 'samples', 'recon_' + str(epoch) + '_' + str(i) + '.png'))
                test_loss += loss.item()
        avg_test_loss = test_loss / len(self.test_loader.dataset)
        print('====> Test set loss: {:.4f}, lenth: {}'.format(avg_test_loss, len(self.test_loader.dataset)))
        # Log test loss to TensorBoard
        self.writer.add_scalar('Loss/test', avg_test_loss, epoch)
        return test_loss

    def train_vae(self, target, source):
        recon_target, posteriors_target = self.vae_model(target)
        with torch.no_grad():
            _, posteriors_source = self.sr_model(source)
        target_loss, _ = self.vae_loss(target, recon_target, posteriors_target)
        domain_features = torch.cat([posteriors_target.mean, posteriors_source.mean], dim=0).view(-1, 70)
        domain_labels = torch.cat([torch.ones(posteriors_target.mean.size(0), device=device), 
                                    torch.zeros(posteriors_source.mean.size(0), device=device)], dim=0).long()
        # shuffle domain_features and domain_labels
        indices = torch.randperm(domain_features.size(0))
        domain_features = domain_features[indices]
        domain_labels = domain_labels[indices]
        domain_probs = self.discriminator(domain_features, 10.0)
        # Domain loss (NLLLoss)
        domain_loss = F.nll_loss(domain_probs, domain_labels)
        return target_loss, domain_loss

    def learn(self):
        for epoch in range(1, self.args.epochs + 1):
            # Load the initial buffers for both datasets
            self.dataset_target.load_next_buffer()
            self.dataset_gt.load_next_buffer()

            # Get the lengths of both DataLoaders
            len_syn = len(self.gt_loader)
            len_real = len(self.target_loader)
            print(f"Length of synthetic data loader: {len_syn}")
            print(f"Length of real data loader: {len_real}")
            # Identify which loader is longer
            if len_syn > len_real:
                longer_loader = 'synthetic'
                shorter_loader = 'real'
            else:
                longer_loader = 'real'
                shorter_loader = 'synthetic'

            # Create iterators for both data loaders
            syn_iter = iter(self.gt_loader)
            real_iter = iter(self.target_loader)

            # Initialize batch indices for each loader
            batch_idx_syn = 0
            batch_idx_real = 0
            train_total_loss = 0
            train_vae_loss = 0
            train_domain_loss = 0
            while True:
                # Load batch from synthetic loader
                try:
                    batch1 = next(syn_iter).to(device)
                except StopIteration:
                    # If synthetic data loader is exhausted, load the next buffer and restart the iterator
                    self.dataset_gt.load_next_buffer()
                    syn_iter = iter(self.gt_loader)
                    batch1 = next(syn_iter).to(device)
                    batch_idx_syn = 0  # Reset the synthetic batch index
                # Load batch from real loader
                try:
                    batch2 = next(real_iter)['depth'].to(device)
                except StopIteration:
                    # If real data loader is exhausted, load the next buffer and restart the iterator
                    self.dataset_target.load_next_buffer()
                    real_iter = iter(self.target_loader)
                    batch2 = next(real_iter)['depth'].to(device)
                    batch_idx_real = 0  # Reset the real batch index

                if batch2.size(0) < self.args.batch_size:
                    batch1 = batch1[:batch2.size(0)]
                elif batch1.size(0) < self.args.batch_size:
                    batch2 = batch2[:batch1.size(0)]

                if self.args.use_max_pooling:
                    source = min_pool2d(batch1, kernel_size=7, stride=1, padding=3)
                    target = min_pool2d(batch2, kernel_size=3, stride=1, padding=1)
                training_target_samples = target
                training_source_samples = source[:training_target_samples.size(0)]

                self.vae_model.train()
                self.discriminator.train()
                self.optimizer.zero_grad()
                if self.it_g % 20 == 0:
                    self.optimizer_disc.zero_grad()
                target_loss, domain_loss = self.train_vae(training_target_samples, training_source_samples)
                total_loss = target_loss + 20 * domain_loss
                train_total_loss += total_loss.item()
                train_vae_loss += target_loss.item()
                train_domain_loss += domain_loss.item()
                total_loss.backward()
                self.optimizer.step()
                if self.it_g % 20 == 0:
                    self.optimizer_disc.step()
                    print('VAE Loss: {:.6f} Domain Loss: {:.6f}'.format(total_loss.item(), domain_loss.item()))
                # Increment the batch indices
                batch_idx_syn += 1
                batch_idx_real += 1
                self.it_g += 1
                # Stop the loop when the longer loader is exhausted
                if (longer_loader == 'synthetic' and batch_idx_syn >= len_syn) or \
                (longer_loader == 'real' and batch_idx_real >= len_real):
                    print(f"Longer data loader ({longer_loader}) exhausted, ending epoch.")
                    break
            average_total_loss = train_total_loss / len_syn if len_syn > len_real else (train_total_loss / len_real)
            average_vae_loss = train_vae_loss / len_syn if len_syn > len_real else (train_vae_loss / len_real)
            average_domain_loss = train_domain_loss / len_syn if len_syn > len_real else (train_domain_loss / len_real)
            self.writer.add_scalar('train/total_loss', average_total_loss, epoch)
            self.writer.add_scalar('train/vae_loss', average_vae_loss, epoch)
            self.writer.add_scalar('train/domain_loss', average_domain_loss, epoch)
            print('====> Epoch: {} Average total loss: {:.4f}'.format(epoch, average_total_loss))
            print('====> Epoch: {} Average vae loss: {:.4f}'.format(epoch, average_vae_loss))
            print('====> Epoch: {} Average domain loss: {:.4f}'.format(epoch, average_domain_loss))

            self.test(epoch)
            self.scheduler.step()
            self.scheduler_disc.step()
            if epoch % 50 == 0:
                filename_vae = join(self.avae_dir, 'vae_' + str(epoch) + '.tar')
                filename_disc = join(self.avae_dir, 'disc_' + str(epoch) + '.tar')
                state_vae = {
                    'epoch': epoch,
                    'state_dict': self.vae_model.state_dict(),
                    'optimizer': self.optimizer.state_dict(),
                    'scheduler': self.scheduler.state_dict(),
                }
                torch.save(state_vae, filename_vae)
                state_disc = {
                    'epoch': epoch,
                    'state_dict': self.discriminator.state_dict(),
                }
                torch.save(state_disc, filename_disc)
        self.writer.close()
                
def main():
    args = parser().parse_args()
    da = DomainAdaptationFromReal2GT(args)
    da.learn()

if __name__ == '__main__':
    main()
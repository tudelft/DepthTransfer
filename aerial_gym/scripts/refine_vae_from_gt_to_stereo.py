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
seed = 12
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
    parser.add_argument("--use_fixed_sr_weight", type=bool, default=True, help="Use fixed source weight")
    return parser

class DomainAdaptationFromGT2Stereo:
    def __init__(self, args) -> None:
        self.args = args
        self.transform_train = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((Input_height, Input_width)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
        ])
        # load target dataset
        self.dataset_target = _RolloutDataset('../saved/dataset_outdoor_env',
                                                self.transform_train, device=device, train=True)
        
        self.dataset_target_test = _RolloutDataset('../saved/dataset_outdoor_env',
                                                self.transform_train, device=device, train=False)
        
        self.target_loader = torch.utils.data.DataLoader(self.dataset_target, batch_size=args.batch_size, shuffle=True, num_workers=8)
        self.target_loader_test = torch.utils.data.DataLoader(self.dataset_target_test, batch_size=args.batch_size, shuffle=True, num_workers=8)
        # load source data
        self.dataset_source = _RolloutDataset('../saved/dataset_outdoor_stereo_new',
                                                self.transform_train, file_num=3, device=device, train=True)
        self.source_loader = torch.utils.data.DataLoader(
            self.dataset_source, batch_size=args.batch_size, shuffle=True, num_workers=16)
        
        self.vae_model = create_model('../mav_baselines/torch/controlNet/models/autoencoder.yaml')
        self.vae_model.to(device)
        ckpt_file = join(args.logdir, 'vae_resnet_16ch_outdoor_stereo', 'checkpoint.tar')
        # ckpt_file = join(args.logdir, 'vae_resnet_from_gt_to_stereo', 'vae_0.tar')
        ckpt = torch.load(ckpt_file, map_location=device)
        self.vae_model.load_state_dict(ckpt['state_dict'])

        if args.use_fixed_sr_weight:
            self.sr_model = create_model('../mav_baselines/torch/controlNet/models/autoencoder.yaml')
            self.sr_model.to(device)
            ckpt_file = join(args.logdir, 'vae_resnet_16ch_outdoor_stereo', 'checkpoint.tar')
            ckpt = torch.load(ckpt_file, map_location=device)
            self.sr_model.load_state_dict(ckpt['state_dict'])
            # freeze the sr_model
            for param in self.sr_model.parameters():
                param.requires_grad = False

        self.discriminator = Discriminator(70)
        # freeze the discriminator
        self.discriminator.to(device)
        self.optimizer_disc = optim.Adam(self.discriminator.parameters(), lr=2e-4, betas=(0.5, 0.9))
        self.scheduler_disc = optim.lr_scheduler.StepLR(self.optimizer_disc, step_size=50, gamma=0.5)

        self.optimizer = optim.Adam(self.vae_model.parameters(), lr=2e-4)
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=50, gamma=0.5)
        self.vae_loss = AutoEncoderLoss()

        self.avae_dir = join(args.logdir, 'vae_resnet_from_gt_to_stereo')
        if not exists(self.avae_dir):
            mkdir(self.avae_dir)
            mkdir(join(self.avae_dir, 'samples'))

        self.writer = SummaryWriter(log_dir=self.avae_dir)
        self.global_step = 0
        self.it_d, self.it_g = 0, 0

    def train_vae(self, epoch):
        self.vae_model.train()
        self.discriminator.train()
        self.dataset_target.load_next_buffer()
        self.dataset_source.load_next_buffer()
        train_loss = 0
        train_vae_loss = 0
        train_domain_loss = 0
        training_samples_num = 0
        for batch_idx, (batch1, batch2) in enumerate(itertools.zip_longest(self.target_loader, self.source_loader, fillvalue=None)):
            batch1 = batch1.to(device)
            batch2 = batch2.to(device)
            if self.args.use_max_pooling:
                target = min_pool2d(batch1, kernel_size=7, stride=1, padding=3)
                source = min_pool2d(batch2, kernel_size=3, stride=1, padding=1)
            training_target_samples = target
            success_num = 0
            if training_target_samples.size(0) == 0:
                continue
            training_source_samples = source[:training_target_samples.size(0)]
            training_samples_num += (training_target_samples.size(0) + training_source_samples.size(0))
            self.optimizer.zero_grad()
            if batch_idx % 5 == 0:
                self.optimizer_disc.zero_grad()

            recon_target, posteriors_target = self.vae_model(training_target_samples)
            if self.args.use_fixed_sr_weight:
                with torch.no_grad():
                    _, posteriors_source = self.sr_model(training_source_samples)
            else:
                _, posteriors_source = self.vae_model(training_source_samples)
            target_loss, _ = self.vae_loss(training_target_samples, recon_target, posteriors_target)
            # source_loss, _ = self.vae_loss(training_source_samples, recon_source, posteriors_source)

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
            # domain_loss = torch.tensor(0.0, device=device)
            # total loss
            # total_loss = target_loss + source_loss -  1.0 * domain_loss
            total_loss = target_loss + 5 * domain_loss
            # total_loss = 1.0 * domain_loss
            total_loss.backward()
            train_vae_loss += target_loss.item()
            train_domain_loss += domain_loss.item()
            train_loss += total_loss.item()
            self.optimizer.step()
            if batch_idx % 5 == 0:
                self.optimizer_disc.step()
                nums = training_target_samples.size(0) + training_source_samples.size(0)
                print('VAE Loss: {:.6f} Domain Loss: {:.6f}, Number of Training samples: {}, fake success sample number: {}'.format(total_loss.item(), 
                                                                    domain_loss.item(), nums, success_num))
        # print the parameters of the discriminator
        # for name, param in self.discriminator.named_parameters():
        #     print(name, param)
        # print(self.vae_model.encoder.mid.block_1.norm1.bias)
        average_train_loss = train_loss / len(self.target_loader.dataset)
        average_vae_loss = train_vae_loss / len(self.target_loader.dataset)
        average_domain_loss = train_domain_loss / len(self.target_loader.dataset)
        self.writer.add_scalar('Loss/average_train_loss', average_train_loss, epoch)
        self.writer.add_scalar('Loss/average_vae_loss', average_vae_loss, epoch)
        self.writer.add_scalar('Loss/average_domain_loss', average_domain_loss, epoch)
        print("training_samples_num: ", training_samples_num)
        print('====> Epoch: {} Average loss: {:.4f} Average vae loss: {:.4f} Average domain loss: {:.4f}'.format(epoch, average_train_loss, average_vae_loss, average_domain_loss))

    def train_classifier(self, epoch):
        self.discriminator.train()
        self.vae_model.eval()

    def test(self, epoch=0):
        self.vae_model.eval()
        self.discriminator.eval()
        self.dataset_target_test.load_next_buffer()
        # self.dataset_source.load_next_buffer()
        # test_loss = 0
        for batch_idx, batch in enumerate(self.target_loader_test):
            batch = batch.to(device)
            save_image(batch, join(self.avae_dir, 'samples', 'raw_' + str(batch_idx) + 'epoch_' + str(epoch) + '.png'))
            recon, posteriors = self.vae_model(batch)
            save_image(recon, join(self.avae_dir, 'samples', 'test_' + str(batch_idx) + 'epoch_' + str(epoch) + '.png'))
            # loss, _ = self.vae_loss(batch, recon, posteriors)
            recon_fake = self.sr_model.decode(posteriors.sample())
            save_image(recon_fake, join(self.avae_dir, 'samples', 'test_fake_' + str(batch_idx) + 'epoch_' + str(epoch) + '.png'))


    def learn(self):
        for epoch in range(self.args.epochs):
            # self.train_classifier(epoch)
            self.train_vae(epoch)
            self.scheduler.step()
            self.scheduler_disc.step()
            # checkpointing
            if epoch % 50 == 0:
                self.test(epoch)
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
    da = DomainAdaptationFromGT2Stereo(args)
    da.learn()

if __name__ == '__main__':
    main()
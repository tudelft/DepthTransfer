import argparse
from os.path import join, exists
from os import mkdir
import itertools

import torch
import torch.utils.data
from torch import optim
from torch.nn import functional as F
from torchvision import transforms
from torchvision.utils import save_image
from aerial_gym.mav_baselines.torch.controlNet.cldm.model import create_model, load_state_dict
from aerial_gym.mav_baselines.torch.controlNet.ldm.modules.losses.autoencoder_kl import AutoEncoderLoss
from torch.utils.tensorboard import SummaryWriter  # Import TensorBoard Writer
from aerial_gym.mav_baselines.torch.models.vae_320 import save_checkpoint, min_pool2d
from aerial_gym.data.loaders import _RolloutDataset, RosbagDataset

cuda = torch.cuda.is_available()
torch.manual_seed(123)
torch.backends.cudnn.benchmark = True
device = torch.device("cuda" if cuda else "cpu")

Input_width = 320
Input_height = 224

def parser():
    parser = argparse.ArgumentParser(description='VAE Trainer')
    parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 32)')
    parser.add_argument('--epochs', type=int, default=500, metavar='N',
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

class AttentionAutoEncoderSimStereo2GT:
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
        # load target dataset
        self.dataset_train = _RolloutDataset('../saved/dataset_outdoor_stereo_new',
                                                self.transform_train, device=device, train=True)
        self.dataset_test = _RolloutDataset('../saved/dataset_outdoor_stereo_new',
                                                self.transform_test, device=device, train=False)
        self.train_loader = torch.utils.data.DataLoader(
            self.dataset_train, batch_size=args.batch_size, shuffle=True, num_workers=8)
        self.test_loader = torch.utils.data.DataLoader(
            self.dataset_test, batch_size=args.batch_size, shuffle=True, num_workers=8)

        # load source data
        self.dataset_gt = _RolloutDataset('../saved/dataset_outdoor_env',
                                                self.transform_train, file_num=3, device=device, train=True)
        self.gt_loader = torch.utils.data.DataLoader(
            self.dataset_gt, batch_size=args.batch_size, shuffle=True, num_workers=8)

        self.model = create_model('../mav_baselines/torch/controlNet/models/autoencoder.yaml')
        self.model.to(device)
        ckpt_file = join(args.logdir, 'vae_resnet_16ch_outdoor', 'checkpoint.tar')
        ckpt = torch.load(ckpt_file, map_location=device)
        self.model.load_state_dict(ckpt['state_dict'])
        # freeze the decoder
        for param in self.model.decoder.parameters():
            param.requires_grad = False

        self.sr_model = create_model('../mav_baselines/torch/controlNet/models/autoencoder.yaml')
        self.sr_model.to(device)
        self.sr_model.load_state_dict(ckpt['state_dict'])
        # freeze the source model
        for param in self.sr_model.parameters():
            param.requires_grad = False

        self.optimizer = optim.Adam(filter(lambda p : p.requires_grad, self.model.parameters()), lr=4e-4)

        # self.loss = AutoEncoderLoss()
        self.loss = create_model('../mav_baselines/torch/controlNet/models/loss.yaml')
        self.loss.to(device)
        self.optimizer_disc = optim.Adam(self.loss.discriminator.parameters(), lr=1e-3, betas=(0.5, 0.9))

        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=100, gamma=0.5)
        self.scheduler_disc = optim.lr_scheduler.StepLR(self.optimizer_disc, step_size=50, gamma=0.5)

        self.avae_dir = join(args.logdir, 'vae_resnet_sim_stereo_to_gt')
        if not exists(self.avae_dir):
            mkdir(self.avae_dir)
            mkdir(join(self.avae_dir, 'samples'))

        self.reload_file = join(join(args.logdir, 'vae_resnet_16ch_sim_stereo_depth'), 'best.tar')
        if not args.noreload and exists(self.reload_file):
            state = torch.load(self.reload_file)
            self.model.load_state_dict(state['state_dict'])
            self.optimizer.load_state_dict(state['optimizer'])
            self.scheduler.load_state_dict(state['scheduler'])
            print("Reloading model at epoch {}"
                ", with test error {}".format(
                state['epoch'], state['precision']))
        self.cur_best = None
        # Initialize TensorBoard writer
        self.writer = SummaryWriter(log_dir=self.avae_dir)
        self.global_step = 0
        self.it_d, self.it_g = 0, 0

    def training_step(self, batch_real, batch_fake, optimizer_id):
        reconstructions_fake, posterior_fake = self.model(batch_fake)
        if optimizer_id == 0:
            # train encoder+decoder+logvar
            aeloss, log = self.loss(batch_fake, reconstructions_fake, posterior_fake, optimizer_id, self.global_step, 
                                        last_layer=self.model.get_last_layer(), split="train")
            return aeloss
        
        if optimizer_id == 1:
            # train the discriminator
            discloss, log = self.loss(batch_fake, reconstructions_fake, posterior_fake, optimizer_id, self.global_step,
                                        inputs_real=batch_real, last_layer=self.model.get_last_layer(), split="train")
            return discloss
        
    def testing_step(self, batch):
        reconstructions, posterior = self.model(batch)
        aeloss, log = self.loss(batch, reconstructions, posterior, 0, self.global_step,
                                        last_layer=self.model.get_last_layer(), split="val")
        return aeloss, reconstructions, log

    def train(self, epoch):
        self.model.train()
        self.dataset_train.load_next_buffer()
        self.dataset_gt.load_next_buffer()
        train_d_loss = 0
        train_g_loss = 0
        len_syn = len(self.dataset_train)
        len_stereo = len(self.dataset_gt)
        for batch_idx, (batch1, batch2) in enumerate(itertools.zip_longest(self.train_loader, self.gt_loader, fillvalue=None)):

            batch1 = batch1.to(device)
            batch2 = batch2.to(device)

            self.optimizer.zero_grad()
            if self.args.use_max_pooling:
                batch1 = min_pool2d(batch1, kernel_size=3, stride=1, padding=1)
                batch2 = min_pool2d(batch2, kernel_size=7, stride=1, padding=3)

            # save_image(batch2, join(self.avae_dir, 'samples', 'train_' + str(epoch) + '.png'))
            with torch.no_grad():
                reconstructions_real, _ = self.sr_model(batch2)
            save_image(reconstructions_real, join(self.avae_dir, 'samples', 'train_' + str(epoch) + '.png'))
            if self.it_g % 20 == 0:
                self.optimizer_disc.zero_grad()
                d_loss = self.training_step(reconstructions_real, batch1, 1)
                d_loss.backward()
                self.optimizer_disc.step()
                train_d_loss += d_loss.item()
                self.it_d += 1

            if self.it_d % 1 == 0:
                self.optimizer.zero_grad()
                g_loss = self.training_step(reconstructions_real, batch1, 0)
                g_loss.backward()
                self.optimizer.step()
                train_g_loss += g_loss.item()
                self.it_g += 1

            self.global_step += 1

        average_d_loss = (20*train_d_loss / len_syn) if len_syn > len_stereo else (20*train_d_loss / len_stereo)
        average_g_loss = (train_g_loss / len_syn) if len_syn > len_stereo else (train_g_loss / len_stereo)
        self.writer.add_scalar('Loss/d/train', average_d_loss, epoch)
        self.writer.add_scalar('Loss/g/train', average_g_loss, epoch)
        print(f'====> Epoch: {epoch} Average D loss: {average_d_loss:.4f}')
        print(f'====> Epoch: {epoch} Average G loss: {average_g_loss:.4f}')

            
    def test(self, epoch):
        self.model.eval()
        self.loss.eval()
        self.dataset_test.load_next_buffer()
        test_loss = 0
        with torch.no_grad():
            for i, data in enumerate(self.test_loader):
                # data = self.dict_to_device(data)
                input = data.to(device)
                save_image(data, join(self.avae_dir, 'samples', 'stereo_depth_' + str(epoch) + '.png'))
                if self.args.use_max_pooling:
                    input = min_pool2d(input, kernel_size=3, stride=1, padding=1)
                    save_image(input, join(self.avae_dir, 'samples', 'max_pooling_' + str(epoch) + '.png'))

                loss, recon_batch, log = self.testing_step(input)
                save_image(recon_batch, join(self.avae_dir, 'samples', 'recon_' + str(epoch) + '.png'))
                test_loss += loss.item()
        print(log)
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
            test_loss = self.test(epoch)
            self.train(epoch)
            self.scheduler.step()
            self.scheduler_disc.step()

            # checkpointing
            if epoch % 100 == 0:
                self.cur_best = test_loss
                filename_vae = join(self.avae_dir, 'vae_' + str(epoch) + '.tar')
                filename_disc = join(self.avae_dir, 'disc_' + str(epoch) + '.tar')
                is_best = not self.cur_best or test_loss < self.cur_best
                if is_best:
                    self.cur_best = test_loss
                state_vae = {
                    'epoch': epoch,
                    'state_dict': self.model.state_dict(),
                    'precision': test_loss,
                    'optimizer': self.optimizer.state_dict(),
                    'scheduler': self.scheduler.state_dict(),
                }
                state_disc = {
                    'epoch': epoch,
                    'state_dict': self.loss.state_dict(),
                    'precision': test_loss,
                    'optimizer': self.optimizer_disc.state_dict(),
                    'scheduler': self.scheduler_disc.state_dict(),
                }
                torch.save(state_vae, filename_vae)
                torch.save(state_disc, filename_disc)

        # Close the TensorBoard writer
        self.writer.close()

    
    def dict_to_device(self, dictionary):
        for key in dictionary:
            dictionary[key] = dictionary[key].to(device)
        return dictionary
                
def main():
    args = parser().parse_args()
    ae = AttentionAutoEncoderSimStereo2GT(args)
    ae.learn()

if __name__ == '__main__':
    main()
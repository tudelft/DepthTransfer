""" Training VAE """
import argparse
from os.path import join, exists
from os import mkdir

from aerial_gym.utils.misc import EarlyStopping
from aerial_gym.utils.misc import ReduceLROnPlateau

from aerial_gym.envs.base.mavrl_task_config import MAVRLTaskCfg
from aerial_gym.data.loaders import _RolloutDataset
from aerial_gym.mav_baselines.torch.models.vae_320 import AutoEncoderNet, min_pool2d

import torch
import torch.utils.data
from torch import optim
from torch.nn import functional as F
from torchvision import transforms
from torchvision.utils import save_image

config = MAVRLTaskCfg()

parser = argparse.ArgumentParser(description='VAE Trainer')
parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                    help='input batch size for training (default: 32)')
parser.add_argument('--epochs', type=int, default=400, metavar='N',
                    help='number of epochs to train (default: 200)')
parser.add_argument('--logdir', type=str,  default='../exp_vae_320', help='Directory where results are logged')
parser.add_argument('--noreload', action='store_true',
                    help='Best model is not reloaded if specified')
parser.add_argument('--nosamples', action='store_true',
                    help='Does not save samples during training if specified')
parser.add_argument("--use_max_pooling", type=bool, default=False, help="Use max pooling")
args = parser.parse_args()
cuda = torch.cuda.is_available()
torch.manual_seed(123)
# Fix numeric divergence due to bug in Cudnn
torch.backends.cudnn.benchmark = True
device = torch.device("cuda" if cuda else "cpu")

class ToPILImageIfNeeded:
    def __call__(self, x):
        if isinstance(x, torch.Tensor):
            return transforms.ToPILImage()(x)
        return x
# from models.vae import VAE
transform_train = transforms.Compose([
    ToPILImageIfNeeded(),
    transforms.Resize((config.LatentSpaceCfg.imput_image_size[0], config.LatentSpaceCfg.imput_image_size[1])),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor()
])

transform_test = transforms.Compose([
    ToPILImageIfNeeded(),
    transforms.Resize((config.LatentSpaceCfg.imput_image_size[0], config.LatentSpaceCfg.imput_image_size[1])),
    transforms.ToTensor()
])

dataset_train = _RolloutDataset('../saved/dataset_outdoor_env',
                                          transform_train, device=device, train=True)
dataset_test = _RolloutDataset('../saved/dataset_outdoor_env',
                                         transform_test, device=device, train=False)
train_loader = torch.utils.data.DataLoader(
    dataset_train, batch_size=args.batch_size, shuffle=True, num_workers=18)
test_loader = torch.utils.data.DataLoader(
    dataset_test, batch_size=args.batch_size, shuffle=True, num_workers=18)

model = AutoEncoderNet(1, config.LatentSpaceCfg.vae_dims).to(device)
optimizer = optim.Adam(model.parameters())
scheduler = ReduceLROnPlateau(optimizer, 'min', factor=0.5, patience=5)
earlystopping = EarlyStopping('min', patience=50)

def inverse_depth(x: torch.Tensor, bound_min=False):
    """ Inverse depth transformation """
    if bound_min:
        x = x * config.camera_params.max_range
        x = torch.clamp(x, min=config.camera_params.min_range)
    else:
        x = x * (config.camera_params.max_range - config.camera_params.min_range) + config.camera_params.min_range
    return (1 / x - 1 / config.camera_params.max_range) / (1 / config.camera_params.min_range - 1 / config.camera_params.max_range)

def log_depth(x: torch.Tensor, bound_min=False):
    """ Log depth transformation """
    if bound_min:
        x = x * config.camera_params.max_range
        x = torch.clamp(x, min=config.camera_params.min_range)
    else:
        x = x * (config.camera_params.max_range - config.camera_params.min_range) + config.camera_params.min_range
    return (1.0 + torch.log10(x)) / 2.0

def recover_depth(x: torch.Tensor):
    """ Recover depth from inverse depth """
    x = 2.0 * x - 1.0
    x = 10.0 ** x
    # print(x)
    return (x - config.camera_params.min_range) / (config.camera_params.max_range - config.camera_params.min_range)

# Reconstruction + KL divergence losses summed over all elements and batch
def loss_function(recon_x, x, mu, logsigma, use_log_depth=False):
    """ VAE loss function """
    if use_log_depth:
        recon_x = log_depth(recon_x, bound_min=True)
        x = log_depth(x, bound_min=True)
    BCE = F.mse_loss(recon_x, x, reduction='sum')
    # see Appendix B from VAE paper:
    # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
    # https://arxiv.org/abs/1312.6114
    # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    KLD = -0.5 * torch.sum(1 + 2 * logsigma - mu.pow(2) - (2 * logsigma).exp())
    return BCE + KLD


def train(epoch):
    """ One training epoch """
    model.train()
    dataset_train.load_next_buffer()
    train_loss = 0
    for batch_idx, data in enumerate(train_loader):
        data = data.to(device)
        optimizer.zero_grad()
        if args.use_max_pooling:
            data = min_pool2d(data, kernel_size=3, stride=1, padding=1)
        recon_batch, mu, logvar = model(data)
        loss = loss_function(recon_batch, data, mu, logvar)
        loss.backward()
        train_loss += loss.item()
        optimizer.step()
        if batch_idx % 20 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader),
                loss.item() / len(data)))

    print('====> Epoch: {} Average loss: {:.4f}'.format(
        epoch, train_loss / len(train_loader.dataset)))


def test():
    """ One test epoch """
    model.eval()
    dataset_test.load_next_buffer()
    test_loss = 0
    with torch.no_grad():
        for data in test_loader:
            save_image(data, join(args.logdir, 'vae', 'samples', 'test_' + str(epoch) + '.png'))
            data = data.to(device)
            if args.use_max_pooling:
                data = min_pool2d(data, kernel_size=3, stride=1, padding=1)
                save_image(data, join(args.logdir, 'vae', 'samples', 'test_' + str(epoch) + '_minpool.png'))
            recon_batch, mu, logvar = model(data)
            save_image(recon_batch, join(args.logdir, 'vae', 'samples', 'recon_' + str(epoch) + '.png'))
            test_loss += loss_function(recon_batch, data, mu, logvar).item()

    test_loss /= len(test_loader.dataset)
    print('====> Test set loss: {:.4f}'.format(test_loss))
    return test_loss

# check vae dir exists, if not, create it
vae_dir = join(args.logdir, 'vae')
if not exists(vae_dir):
    mkdir(vae_dir)
    mkdir(join(vae_dir, 'samples'))

reload_file = join(vae_dir, 'best.tar')
if not args.noreload and exists(reload_file):
    state = torch.load(reload_file)
    print("Reloading model at epoch {}"
          ", with test error {}".format(
              state['epoch'],
              state['precision']))
    model.load_state_dict(state['state_dict'])
    optimizer.load_state_dict(state['optimizer'])
    scheduler.load_state_dict(state['scheduler'])
    earlystopping.load_state_dict(state['earlystopping'])


cur_best = None

def save_checkpoint(state, is_best, filename, best_filename):
    """ Save state in filename. Also save in best_filename if is_best. """
    torch.save(state, filename)
    if is_best:
        torch.save(state, best_filename)

for epoch in range(1, args.epochs + 1):
    train(epoch)
    test_loss = test()
    scheduler.step(test_loss)
    earlystopping.step(test_loss)

    # checkpointing
    best_filename = join(vae_dir, 'best.tar')
    filename = join(vae_dir, 'checkpoint.tar')
    is_best = not cur_best or test_loss < cur_best
    if is_best:
        cur_best = test_loss

    save_checkpoint({
        'epoch': epoch,
        'state_dict': model.state_dict(),
        'precision': test_loss,
        'optimizer': optimizer.state_dict(),
        'scheduler': scheduler.state_dict(),
        'earlystopping': earlystopping.state_dict()
    }, is_best, filename, best_filename)



    # if not args.nosamples:
    #     with torch.no_grad():
    #         sample = torch.randn(64, config.LatentSpaceCfg.vae_dims).to(device)
    #         sample = model.decoder(sample).cpu()
    #         save_image(sample.view(64, 1, config.LatentSpaceCfg.imput_image_size[0], config.LatentSpaceCfg.imput_image_size[1]),
    #                    join(vae_dir, 'samples/sample_' + str(epoch) + '.png'))

    # if earlystopping.stop:
    #     print("End of Training because of early stopping at epoch {}".format(epoch))
    #     break

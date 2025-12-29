import torch
import torch.nn as nn
import torch.nn.functional as F
import functools

def save_checkpoint(state, is_best, filename, best_filename):
    """ Save state in filename. Also save in best_filename if is_best. """
    torch.save(state, filename)
    if is_best:
        torch.save(state, best_filename)

class EncoderDecoderNet(nn.Module):
    def __init__(self, in_channels=1, num_classes=10):
        super(EncoderDecoderNet, self).__init__()
        
        self.encoder_conv1 = self.conv_block(in_channels, 64)
        self.encoder_conv2 = self.conv_block(64, 128)
        self.encoder_conv3 = self.conv_block(128, 256)
        self.encoder_conv4 = self.conv_block(256, 512)
        self.encoder_conv5 = self.conv_block(512, 512)
        self.encoder_conv6 = self.conv_block(512, 512)

        self.unpool = nn.MaxUnpool2d(kernel_size=2, stride=2)

        self.decoder_conv6 = self.conv_block(512, 512)
        self.decoder_conv5 = self.conv_block(512, 512)
        self.decoder_conv4 = self.conv_block(512, 256)
        self.decoder_conv3 = self.conv_block(256, 128)
        self.decoder_conv2 = self.conv_block(128, 64)
        self.decoder_conv1 = self.conv_block(64, in_channels)

        self.final_layer = nn.Conv2d(in_channels, num_classes, kernel_size=1)
        self.softmax = nn.Softmax(dim=1)

    def conv_block(self, in_channels, out_channels, kernel_size=3, padding=1):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU()
        )

    def forward(self, x):
        sizes, indices = [], []
        
        # Encoding
        x = self.encoder_conv1(x); sizes.append(x.size()); x, ind = F.max_pool2d(x, 2, 2, return_indices=True); indices.append(ind)
        x = self.encoder_conv2(x); sizes.append(x.size()); x, ind = F.max_pool2d(x, 2, 2, return_indices=True); indices.append(ind)
        x = self.encoder_conv3(x); sizes.append(x.size()); x, ind = F.max_pool2d(x, 2, 2, return_indices=True); indices.append(ind)
        x = self.encoder_conv4(x); sizes.append(x.size()); x, ind = F.max_pool2d(x, 2, 2, return_indices=True); indices.append(ind)
        x = self.encoder_conv5(x); sizes.append(x.size()); x, ind = F.max_pool2d(x, 2, 2, return_indices=True); indices.append(ind)
        x = self.encoder_conv6(x); sizes.append(x.size()); x, ind = F.max_pool2d(x, 2, 2, return_indices=True); indices.append(ind)
        # Decoding
        x = self.unpool(x, indices.pop(), output_size=sizes.pop())
        x = self.decoder_conv6(x)
        
        x = self.unpool(x, indices.pop(), output_size=sizes.pop())
        x = self.decoder_conv5(x)

        x = self.unpool(x, indices.pop(), output_size=sizes.pop())
        x = self.decoder_conv4(x)

        x = self.unpool(x, indices.pop(), output_size=sizes.pop())
        x = self.decoder_conv3(x)

        x = self.unpool(x, indices.pop(), output_size=sizes.pop())
        x = self.decoder_conv2(x)

        x = self.unpool(x, indices.pop(), output_size=sizes.pop())
        x = self.decoder_conv1(x)

        x = self.final_layer(x)
        return self.softmax(x)
    
class HolePredictor_Unet(nn.Module):
    def __init__(self, input_nc, output_nc, num_classes=2, ngf=64, norm_layer=nn.BatchNorm2d, use_dropout=False) -> None:
        super(HolePredictor_Unet, self).__init__()
        self.segment_end = nn.Conv2d(output_nc, num_classes, kernel_size=1)
        unet_block = HolePredictor_Unet_Block(ngf * 8, ngf * 8, input_nc=None, submodule=None, innermost=True, norm_layer=norm_layer, kernel_size=(3, 5), padding=0, probablistic=True)
        unet_block = HolePredictor_Unet_Block(ngf * 8, ngf * 8, input_nc=None, submodule=unet_block, norm_layer=norm_layer, use_dropout=use_dropout)
        unet_block = HolePredictor_Unet_Block(ngf * 8, ngf * 8, input_nc=None, submodule=unet_block, norm_layer=norm_layer, use_dropout=use_dropout, kernel_size=(5, 4), padding=(0, 1))
        unet_block = HolePredictor_Unet_Block(ngf * 4, ngf * 8, input_nc=None, submodule=unet_block, norm_layer=norm_layer)
        unet_block = HolePredictor_Unet_Block(ngf * 2, ngf * 4, input_nc=None, submodule=unet_block, norm_layer=norm_layer)
        unet_block = HolePredictor_Unet_Block(ngf, ngf * 2, input_nc=None, submodule=unet_block, norm_layer=norm_layer)
        unet_block = HolePredictor_Unet_Block(output_nc, ngf, input_nc=input_nc, submodule=unet_block, outermost=True, norm_layer=norm_layer)

        self.model = unet_block
    
    def forward(self, x):
        return self.segment_end(self.model(x))
    
class HolePredictor_Unet_Block(nn.Module):
    def __init__(self, outer_nc, inner_nc, input_nc=None, submodule=None, outermost=False, innermost=False, 
                 norm_layer=nn.BatchNorm2d, use_dropout=False, kernel_size=4, padding=1, probablistic=False):
        super(HolePredictor_Unet_Block, self).__init__()
        self.outermost = outermost
        if type(norm_layer) is functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d
        if input_nc is None:
            input_nc = outer_nc
        downconv = nn.Conv2d(input_nc, inner_nc, kernel_size=kernel_size, stride=2, padding=padding, bias=use_bias)
        downrelu = nn.LeakyReLU(0.2, True)
        downnorm = norm_layer(inner_nc)
        uprelu = nn.ReLU(True)
        upnorm = norm_layer(outer_nc)
        # if padding is not symmetric, the padding of deconvolution should be reversed
        if outermost:
            upconv = nn.ConvTranspose2d(inner_nc * 2, outer_nc, kernel_size=kernel_size, stride=2, padding=padding)
            down = [downconv]
            up = [uprelu, upconv, nn.Sigmoid()]
            model = down + [submodule] + up
        elif innermost:
            upconv = nn.ConvTranspose2d(inner_nc, outer_nc, kernel_size=kernel_size, stride=2, padding=padding, bias=use_bias)
            down = [downrelu, downconv]
            up = [uprelu, upconv, upnorm]
            # if probablistic:
            #     print("inner_nc: %d, outer_nc: %d" % (inner_nc, outer_nc))
            #     downlinear = nn.Linear(inner_nc, 64)
            #     uplinear = nn.Linear(64, outer_nc)
            #     down = down + [downlinear]
            #     up = [uplinear] + up
            model = down + up
        else:
            upconv = nn.ConvTranspose2d(inner_nc * 2, outer_nc, kernel_size=kernel_size, stride=2, padding=padding, bias=use_bias)
            down = [downrelu, downconv, downnorm]
            up = [uprelu, upconv, upnorm]
            if use_dropout:
                model = down + [submodule] + up + [nn.Dropout(0.5)]
            else:
                model = down + [submodule] + up

        self.model = nn.Sequential(*model)

    def forward(self, x):
        if self.outermost:
            return self.model(x)
        else:
            return torch.cat([x, self.model(x)], 1)
        
class LossBinary:

    def __init__(self, jaccard_weight=0):
        self.nll_loss = nn.BCEWithLogitsLoss()
        self.jaccard_weight = jaccard_weight

    def __call__(self, outputs, targets):
        loss = (1 - self.jaccard_weight) * self.nll_loss(outputs, targets)

        if self.jaccard_weight:
            eps = 1e-15
            jaccard_target = (targets == 1).float()
            jaccard_output = torch.sigmoid(outputs)

            intersection = (jaccard_output * jaccard_target).sum()
            union = jaccard_output.sum() + jaccard_target.sum()

            loss -= self.jaccard_weight * torch.log((intersection + eps) / (union - intersection + eps))

        return loss
    
class VAELoss:
    def __init__(self, beta=1):
        self.beta = beta

    def __call__(self, recon_x, x, mu, logvar):
        BCE = F.mse_loss(recon_x, x, reduction='mean')
        KLD = -0.5 * torch.sum(1 + 2 * logvar - mu.pow(2) - (2 * logvar).exp())
        return BCE
    
class AutoEncoderNet(nn.Module):
    def __init__(self, n_input_channels: int, features_dim: int=64, ngf: int=64):
        super(AutoEncoderNet, self).__init__()
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

    def forward(self, x):
        x = self.encoder(x)
        x = x.view(x.size(0), -1)
        mu = self.mu_linear(x)
        logvar = self.logvar_linear(x)
        z = torch.randn_like(mu).mul(torch.exp(logvar * 0.5)).add_(mu)
        z = self.fc_decoder(z)
        z = z.unsqueeze(-1).unsqueeze(-1)
        return self.decoder(z), mu, logvar


# Function to get normalization layer (for simplicity, we'll use only BatchNorm2d)
def _get_norm_layer_2d(norm):
    if norm == 'batch_norm':
        return nn.BatchNorm2d
    elif norm == 'instance_norm':
        return nn.InstanceNorm2d
    else:
        return nn.Identity

# Define the ConvDiscriminator class
class ConvDiscriminator(nn.Module):
    def __init__(self, input_channels=3, dim=64, n_downsamplings=4, norm='batch_norm'):
        super().__init__()
        Norm = _get_norm_layer_2d(norm)

        def conv_norm_lrelu(in_dim, out_dim, kernel_size=4, stride=2, padding=1):
            return nn.Sequential(
                nn.Conv2d(in_dim, out_dim, kernel_size, stride=stride, padding=padding, bias=False or Norm == nn.Identity),
                Norm(out_dim),
                nn.LeakyReLU(0.2)
            )

        layers = []

        # Initial conv layer
        d = dim
        layers.append(nn.Conv2d(input_channels, d, kernel_size=4, stride=2, padding=1))
        layers.append(nn.LeakyReLU(0.2))

        # Downsampling layers
        for i in range(n_downsamplings - 1):
            d_last = d
            d = min(dim * 2 ** (i + 1), dim * 8)
            layers.append(conv_norm_lrelu(d_last, d, kernel_size=4, stride=2, padding=1))

        # Logit layer
        layers.append(nn.Conv2d(d, 1, kernel_size=4, stride=1, padding=0))

        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)

def get_lsgan_losses_fn():
    mse = torch.nn.MSELoss()

    def d_loss_fn(r_logit, f_logit):
        r_loss = mse(r_logit, torch.ones_like(r_logit))
        f_loss = mse(f_logit, torch.zeros_like(f_logit))
        return r_loss, f_loss

    def g_loss_fn(f_logit):
        f_loss = mse(f_logit, torch.ones_like(f_logit))
        return f_loss

    return d_loss_fn, g_loss_fn

# Define the min pooling function
def min_pool2d(input, kernel_size, stride=None, padding=0, dilation=1):
    neg_input = -input
    max_pool = F.max_pool2d(neg_input, kernel_size, stride, padding, dilation)
    min_pool = -max_pool
    return min_pool


# # Instantiate the discriminator with default settings
# discriminator = ConvDiscriminator(input_channels=1, n_downsamplings=5)

# # Create a sample input with size 240x320 (batch size 1, RGB image with 3 channels)
# input_image = torch.randn(1, 1, 224, 320)

# # Pass the input through the discriminator
# output = discriminator(input_image)

# print(output.shape)

# # Check if GPU is available
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# # Example input (now 240x320)
# x = torch.randn(1, 1, 240, 320).to(device)

# model = AutoEncoderNet(n_input_channels=1, features_dim=64, ngf=64).to(device)
# z, mu, logvar = model(x)
# print(f"Output shape: {z.shape}")
# print(f"Mu shape: {mu.shape}")
# print(f"Logvar shape: {logvar.shape}")

# # Instantiate and run the model
# # model = EncoderDecoderNet(in_channels=1, num_classes=10).to(device)
# # output = model(x)

# model = HolePredictor_Unet(input_nc=1, output_nc=64, num_classes=1).to(device)
# output = model(x)

# # print the weights shape of the model
# for key in model.state_dict():
#     print(key, model.state_dict()[key].shape)

# print(f"Output shape: {output.shape}")

# from torchviz import make_dot
# from torchsummary import summary

# # Instantiate the model
# input_nc = 1
# output_nc = 64
# num_classes = 1
# ngf = 64

# model = HolePredictor_Unet(input_nc, output_nc, num_classes, ngf).to(device)

# # Create dummy input to visualize the model structure
# x = torch.randn(1, input_nc, 240, 320).to(device)
# # Generate a graph of the model
# graph = make_dot(model(x), params=dict(model.named_parameters()))
# graph.render("HolePredictor_Unet", format="png")

# # Summary of the model
# summary(model, (input_nc, 240, 320))

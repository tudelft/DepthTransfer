import torch
import torch.nn as nn

class AutoEncoderLoss(nn.Module):
    def __init__(self, kl_weight=1.0, pixelloss_weight=1.0, logvar_init=0.0):
        super(AutoEncoderLoss, self).__init__()
        self.kl_weight = kl_weight
        self.pixelloss_weight = pixelloss_weight
        # output log variance
        self.logvar = nn.Parameter(torch.ones(size=()) * logvar_init)

    def forward(self, inputs, reconstructions, posteriors, split="train", weights=None):
        rec_loss = torch.abs(inputs.contiguous() - reconstructions.contiguous())
        nll_loss = rec_loss / torch.exp(self.logvar) + self.logvar
        weighted_nll_loss = nll_loss
        if weights is not None:
            weighted_nll_loss = weights*nll_loss
        weighted_nll_loss = torch.sum(weighted_nll_loss) / weighted_nll_loss.shape[0]
        nll_loss = torch.sum(nll_loss) / nll_loss.shape[0]
        kl_loss = posteriors.kl()
        kl_loss = torch.sum(kl_loss) / kl_loss.shape[0]

        loss = weighted_nll_loss + self.kl_weight * kl_loss
        log = {"{}/total_loss".format(split): loss.clone().detach().mean(), "{}/logvar".format(split): self.logvar.detach(),
                "{}/kl_loss".format(split): kl_loss.detach().mean(), "{}/nll_loss".format(split): nll_loss.detach().mean(),
                "{}/rec_loss".format(split): rec_loss.detach().mean(),
                }
        return loss, log
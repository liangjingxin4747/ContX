import torch
import torch.nn as nn
import torch.nn.functional as F
from taming.modules.discriminator.model import NLayerDiscriminator, weights_init


class BCELoss(nn.Module):
    def forward(self, prediction, target):
        loss = F.binary_cross_entropy_with_logits(prediction, target)
        return loss, {}


class BCELossWithQuant(nn.Module):
    def __init__(self, codebook_weight=1.):
        super().__init__()
        self.codebook_weight = codebook_weight

    def forward(self, qloss, target, prediction, split):
        bce_loss = F.binary_cross_entropy_with_logits(prediction, target)
        loss = bce_loss + self.codebook_weight*qloss
        return loss, {"{}/total_loss".format(split): loss.clone().detach().mean(),
                      "{}/bce_loss".format(split): bce_loss.detach().mean(),
                      "{}/quant_loss".format(split): qloss.detach().mean()
                      }


def adopt_weight(weight, global_step, threshold=0, value=0.):
    if global_step < threshold:
        weight = value
    return weight


def hinge_d_loss(logits_real, logits_fake):
    loss_real = torch.mean(F.relu(1. - logits_real))
    loss_fake = torch.mean(F.relu(1. + logits_fake))
    # loss_real = torch.mean(logits_real)
    # loss_fake = torch.mean(logits_fake)
    d_loss = 0.5 * (loss_real + loss_fake)
    return d_loss


class BCELossWithQuant_Discriminator(nn.Module):
    def __init__(self,
                 disc_start,
                 codebook_weight=1.0,
                 disc_in_channels=87,
                 disc_weight=1.0,
                 disc_num_layers=3,
                 use_actnorm=False,
                 disc_ndf=64):
        super().__init__()
        self.codebook_weight = codebook_weight
        self.discriminator = NLayerDiscriminator(input_nc=disc_in_channels,
                                                 n_layers=disc_num_layers,
                                                 use_actnorm=use_actnorm,
                                                 ndf=disc_ndf
                                                 ).apply(weights_init)
        self.discriminator_iter_start = disc_start
        self.disc_loss = hinge_d_loss
        print(f"BCELossWithQuant_Discriminator running with BCEloss.")
        self.disc_weight = disc_weight

    def forward(self, qloss, target, prediction, optimizer_idx, global_step, split):
        d_weight = adopt_weight(self.disc_weight, global_step, threshold=self.discriminator_iter_start)

        batch_size = target.shape[0]

        # now the GAN part
        if optimizer_idx == 0:
            # generator update
            bce_loss = F.binary_cross_entropy_with_logits(prediction, target)
            g_loss = bce_loss + self.codebook_weight * qloss
            d_loss = torch.tensor([0.0]).cuda()
            if d_weight:
                logits_fake = self.discriminator(prediction.contiguous())
                d_loss = -torch.mean(logits_fake)
                g_loss = g_loss + d_loss * d_weight
            log = {"{}/g_total_loss".format(split): g_loss.clone().detach().mean(),
                   "{}/bce_loss".format(split): bce_loss.detach().mean(),
                   "{}/quant_loss".format(split): qloss.detach().mean(),
                   "{}/d_loss".format(split): d_loss.detach().mean()
                   }
            return g_loss, log

        if optimizer_idx == 1:
            # discriminator update
            logits_real = self.discriminator(target.contiguous().detach())
            logits_fake = self.discriminator(prediction.contiguous().detach())
            disc_loss = self.disc_loss(logits_real, logits_fake) * d_weight
            log = {"{}/disc_loss".format(split): disc_loss.clone().detach().mean(),
                   "{}/logits_real".format(split): logits_real.detach().mean(),
                   "{}/logits_fake".format(split): logits_fake.detach().mean()
                   }
            return disc_loss, log

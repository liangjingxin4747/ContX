import functools

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from taming.modules.losses.lpips import LPIPS
from taming.modules.discriminator.model import NLayerDiscriminator, MultiscaleDiscriminator, weights_init


class DummyLoss(nn.Module):
    def __init__(self):
        super().__init__()


def adopt_weight(weight, global_step, threshold=0, value=0.):
    if global_step < threshold:
        weight = value
    return weight


def hinge_d_loss(logits_real, logits_fake):
    loss_real = torch.mean(F.relu(1. - logits_real))
    loss_fake = torch.mean(F.relu(1. + logits_fake))
    d_loss = 0.5 * (loss_real + loss_fake)
    return d_loss


def vanilla_d_loss(logits_real, logits_fake):
    d_loss = 0.5 * (
        torch.mean(torch.nn.functional.softplus(-logits_real)) +
        torch.mean(torch.nn.functional.softplus(logits_fake)))
    return d_loss


class VQLPIPSWithDiscriminator(nn.Module):
    def __init__(self, disc_start, codebook_weight=1.0, pixelloss_weight=1.0,
                 disc_num_layers=3, disc_in_channels=3, disc_factor=1.0, disc_weight=1.0,
                 perceptual_weight=0.75, rec_weight=1.0, div_weight=0.5, use_actnorm=False, 
                 disc_conditional=False, disc_ndf=64, disc_loss="hinge"):
        super().__init__()
        assert disc_loss in ["hinge", "vanilla"]
        self.disc_in_channels = disc_in_channels
        self.codebook_weight = codebook_weight   # 1.0
        self.pixel_weight = pixelloss_weight   # 1.0
        self.perceptual_weight = perceptual_weight   # 1.0
        self.rec_weight = rec_weight
        self.div_weight = div_weight
        # self.perceptual_loss = LPIPS().eval()

        self.discriminator = NLayerDiscriminator(input_nc=disc_in_channels,   # 87
                                                 n_layers=disc_num_layers,   # 3
                                                 use_actnorm=use_actnorm,
                                                 ndf=disc_ndf
                                                 ).apply(weights_init)
        self.discriminator_iter_start = disc_start   # 25000
        if disc_loss == "hinge":   # !
            self.disc_loss = hinge_d_loss
        elif disc_loss == "vanilla":
            self.disc_loss = vanilla_d_loss
        else:
            raise ValueError(f"Unknown GAN loss '{disc_loss}'.")
        print(f"VQLPIPSWithDiscriminator running with {disc_loss} loss.")
        self.disc_factor = disc_factor   # 1.0
        self.discriminator_weight = disc_weight   # 0.2
        self.disc_conditional = disc_conditional   # False
        self.loss_hpy = torch.nn.L1Loss(reduction='mean')
        self.loss_hpz = torch.nn.L1Loss(reduction='mean')
        self.loss_fn_scr = torch.nn.CrossEntropyLoss()

    def calculate_adaptive_weight(self, nll_loss, g_loss, last_layer=None):
        if last_layer is not None:   # !
            nll_grads = torch.autograd.grad(nll_loss, last_layer, retain_graph=True)[0]
            g_grads = torch.autograd.grad(g_loss, last_layer, retain_graph=True)[0]
        else:
            nll_grads = torch.autograd.grad(nll_loss, self.last_layer[0], retain_graph=True)[0]
            g_grads = torch.autograd.grad(g_loss, self.last_layer[0], retain_graph=True)[0]

        d_weight = torch.norm(nll_grads) / (torch.norm(g_grads) + 1e-4)
        d_weight = torch.clamp(d_weight, 0.0, 1e4).detach()
        d_weight = d_weight * self.discriminator_weight
        return d_weight
    
    def softargmax(self, x):   # Tensor(bs,87,h,w)->Tensor(bs,1,h,w)
        bs, c, h, w = x.shape
        indices = torch.arange(c)[None,:,None,None].repeat(bs,1,h,w).to(x.device)   # Tensor(bs,87,h,w)
        x_soft = torch.sum(x.softmax(dim=1) * indices, dim=1, keepdim=True)   # Tensor(bs,1,h,w)
        x_hard = x.argmax(dim=1, keepdim=True)
        return x_soft + (x_hard - x_soft).detach()   # value=x_hard, grad=x_soft
    
    def forward(self, codebook_loss, dis_x, xrec, optimizer_idx,
                global_step, last_layer=None, cond=None, split="train", clip=None, clip_weight=0.):
        # rec_oss = torch.abs(dis_x.contiguous() - reconstructions.contiguous())   # tensor(b,87,h,w)

        # HPy_target = torch.zeros(inputs.shape[0], inputs.shape[1], inputs.shape[2] - 1, inputs.shape[3]).cuda()
        # HPz_target = torch.zeros(inputs.shape[0], inputs.shape[1], inputs.shape[2], inputs.shape[3] - 1).cuda()
        # HPy = inputs.contiguous()[:, :, 1:, :] - inputs.contiguous()[:, :, 0:-1, :]
        # HPz = inputs.contiguous()[:, :, :, 1:] - inputs.contiguous()[:, :, :, 0:-1]
        # lhpy = self.loss_hpy(HPy, HPy_target)
        # lhpz = self.loss_hpz(HPz, HPz_target)
        
        if isinstance(xrec, list):
            reconstructions = xrec[0]   # reconstruction:(bs,87,128,128)   noise:(bs,64,128,128)
            div_loss = (torch.abs(xrec[2] - xrec[3])).mean() / (torch.abs(xrec[0] - xrec[1]).mean() + (torch.abs(xrec[2] - xrec[3])).mean())
            # print(div_loss)
        else:
            reconstructions = xrec
            div_loss = torch.tensor([0.0]).to(xrec.device)
            
        c = dis_x.shape[1]
        ignore, target = torch.max(dis_x, 1)   # tensor(b,h,w)
        
        if self.perceptual_weight > 0:   # !
            p_loss = torch.tensor([0.0])
            rec_loss = self.loss_fn_scr(reconstructions, target)   # Tensor(bs,87,256,256)
        else:
            p_loss = torch.tensor([0.0])

        nll_loss = rec_loss
        # nll_loss = torch.sum(nll_loss) / nll_loss.shape[0]
        nll_loss = torch.mean(nll_loss)   # Tensor()
        
        # now the GAN part
        if optimizer_idx == 0:
            # generator update
            obj = torch.argmax(cond, dim=1, keepdim=True).float()   # (b,1,h,w)
            if not self.disc_conditional:
                logits_fake = self.discriminator(reconstructions.contiguous())   # Tensor(bs,1,30,30)
            else:
                assert self.disc_conditional
                logits_fake = self.discriminator(torch.cat((reconstructions.softmax(dim=1).contiguous(), cond), dim=1))
            g_loss = -torch.mean(logits_fake)
            
            if clip is None:
                clip_loss = torch.tensor([0.0]).to(reconstructions.device)
            else:
                pred_context = self.softargmax(reconstructions)   # (b,1,h,w)
                pred_scene = torch.where(obj == (c-1), pred_context, obj)
                gt_scene = torch.where(obj == (c-1), target.float().unsqueeze(1), obj)
                clip_loss = clip(pred_scene, gt_scene, split='train')   # Tensor(1)
        
            try:
                d_weight = self.calculate_adaptive_weight(nll_loss, g_loss, last_layer=last_layer)
                d_weight = self.discriminator_weight
            except RuntimeError:
                assert not self.training
                d_weight = torch.tensor(0.0)
            # disc_factor = 1.
            disc_factor = adopt_weight(self.disc_factor, global_step, threshold=self.discriminator_iter_start)
            loss = self.div_weight * div_loss + \
                    self.rec_weight * nll_loss + \
                    d_weight * disc_factor * g_loss + \
                    self.codebook_weight * codebook_loss.mean() + \
                    clip_weight * clip_loss

            log = {"{}/total_loss".format(split): loss.clone().detach().mean(),
                   "{}/clip_loss".format(split): clip_loss.clone().detach().mean(),
                   "{}/quant_loss".format(split): codebook_loss.detach().mean(),
                   "{}/nll_loss".format(split): nll_loss.detach().mean(),
                   "{}/rec_loss".format(split): rec_loss.detach().mean(),
                   "{}/div_loss".format(split): div_loss.detach().mean(),
                   "{}/p_loss".format(split): p_loss.detach().mean(),
                   "{}/d_weight".format(split): d_weight,
                   "{}/disc_factor".format(split): torch.tensor(disc_factor),
                   "{}/g_loss".format(split): g_loss.detach().mean(),
                   # "{}/lhpy_loss".format(split): lhpy.detach().mean(),
                   # "{}/lhpz_loss".format(split): lhpz.detach().mean(),
                   }
            return loss, log

        if optimizer_idx == 1:
            # second pass for discriminator update
            obj = torch.argmax(cond, dim=1, keepdim=True)   # (b,1,h,w)
            if not self.disc_conditional:   # !
                # gt_scene = torch.where(obj == (c-1), target[:,None,:,:], obj)
                # pred_context = torch.argmax(reconstructions, dim=1, keepdim=True)
                # pred_scene = torch.where(obj == (c-1), pred_context, obj)
                logits_real = self.discriminator(dis_x.float().contiguous().detach())   # Tensor(bs,1,30,30)
                logits_fake = self.discriminator(reconstructions.float().contiguous().detach())   # Tensor(bs,1,30,30)
            else:
                logits_real = self.discriminator(torch.cat((dis_x.contiguous().detach(), cond), dim=1))
                logits_fake = self.discriminator(torch.cat((reconstructions.softmax(dim=1).contiguous().detach(), cond), dim=1))

            disc_factor = adopt_weight(self.disc_factor, global_step, threshold=self.discriminator_iter_start)
            d_loss = disc_factor * self.disc_loss(logits_real, logits_fake)

            log = {"{}/disc_loss".format(split): d_loss.clone().detach().mean(),
                   "{}/logits_real".format(split): logits_real.detach().mean(),
                   "{}/logits_fake".format(split): logits_fake.detach().mean()
                   }
            return d_loss, log

  
def cal_kl_loss(p, q, reduction='batchmean'):
    p = F.softmax(p, dim=1)
    q = F.softmax(q, dim=1)
    loss = torch.tensor(0.0).to(q.device)
    for p_i, q_i in zip(p, q):
        loss += F.kl_div(q_i.log(), p_i, reduction=reduction)
    return loss / p.shape[0]


class VQLPIPSWithMultiDiscriminator(nn.Module):
    def __init__(self, disc_start, multi_scale, codebook_weight=1.0, pixelloss_weight=1.0, num_D=3,
                 disc_num_layers=3, disc_in_channels=3, disc_factor=1.0, disc_weight=1.0,
                 perceptual_weight=1.0, use_actnorm=False, disc_conditional=False,
                 disc_ndf=64, disc_loss="hinge",
                 use_ganFeat_loss=False, use_lsgan=False):
        super().__init__()
        assert disc_loss in ["hinge", "vanilla"]
        self.codebook_weight = codebook_weight  # 1.0
        self.pixel_weight = pixelloss_weight  # 1.0
        self.perceptual_weight = perceptual_weight  # 1.0

        self.multi_scale = multi_scale
        if self.multi_scale:
            self.discriminator = MultiscaleDiscriminator(input_nc=disc_in_channels,  # 87
                                                         ndf=disc_ndf,
                                                         n_layers=disc_num_layers,  # 3
                                                         norm_layer=functools.partial(nn.InstanceNorm2d, affine=False),
                                                         use_sigmoid=use_lsgan,
                                                         num_D=num_D,
                                                         getIntermFeat=use_ganFeat_loss
                                                         ).apply(weights_init)
            self.criterionGAN = GANLoss(use_lsgan=use_lsgan, tensor=torch.cuda.FloatTensor)
        else:
            self.discriminator = NLayerDiscriminator(input_nc=disc_in_channels,
                                                     n_layers=disc_num_layers,
                                                     use_actnorm=use_actnorm,
                                                     ndf=disc_ndf
                                                     ).apply(weights_init)
        self.discriminator_iter_start = disc_start  # 25000
        if disc_loss == "hinge":  # !
            self.disc_loss = hinge_d_loss
        elif disc_loss == "vanilla":
            self.disc_loss = vanilla_d_loss
        else:
            raise ValueError(f"Unknown GAN loss '{disc_loss}'.")
        print(f"VQLPIPSWithDiscriminator running with {disc_loss} loss.")
        self.disc_factor = disc_factor  # 1.0
        self.discriminator_weight = disc_weight  # 0.2
        self.disc_conditional = disc_conditional  # False
        self.loss_fn_scr = torch.nn.CrossEntropyLoss()


    def calculate_adaptive_weight(self, nll_loss, g_loss, last_layer=None):
        if last_layer is not None:  # !
            nll_grads = torch.autograd.grad(nll_loss, last_layer, retain_graph=True)[0]
            g_grads = torch.autograd.grad(g_loss, last_layer, retain_graph=True)[0]
        else:
            nll_grads = torch.autograd.grad(nll_loss, self.last_layer[0], retain_graph=True)[0]
            g_grads = torch.autograd.grad(g_loss, self.last_layer[0], retain_graph=True)[0]

        d_weight = torch.norm(nll_grads) / (torch.norm(g_grads) + 1e-4)
        d_weight = torch.clamp(d_weight, 0.0, 1e4).detach()
        d_weight = d_weight * self.discriminator_weight
        return d_weight

    def forward(self, codebook_loss, dis_x, reconstructions, optimizer_idx,
                global_step, last_layer=None, cond=None, split="train"):
        if not isinstance(reconstructions, list):
            reconstructions = [reconstructions]

        # now the GAN part
        if optimizer_idx == 0:
            # generator update
            disc_factor = adopt_weight(self.disc_factor, global_step, threshold=self.discriminator_iter_start)

            log = {"{}/quant_loss".format(split): codebook_loss.detach().mean(),
                   "{}/disc_factor".format(split): torch.tensor(disc_factor)}
            total_loss = self.codebook_weight * codebook_loss.mean()

            for i, xrec in enumerate(reconstructions):
                # rec_oss = torch.abs(dis_x.contiguous() - reconstructions.contiguous())   # tensor(b,87,h,w)
                ignore, target = torch.max(dis_x, 1)  # tensor(b,h,w)

                rec_loss = self.loss_fn_scr(xrec, target)  # Tensor(bs,87,256,256)
                nll_loss = torch.mean(rec_loss)   # Tensor

                if cond is None:
                    assert not self.disc_conditional
                    logits_fake = self.discriminator(xrec.contiguous())  # Tensor(bs,1,30,30)
                else:  # !
                    assert self.disc_conditional
                    if self.multi_scale:
                        pred_fake = self.discriminator(xrec.contiguous(), cond)   # list(num_D=2) of list(5)
                        logits_fake = self.criterionGAN(pred_fake, True)   # Tensor() 叠加两个g_loss
                    else:
                        logits_fake = self.discriminator(torch.cat((xrec.contiguous(), cond), dim=1))
                        logits_fake = -torch.mean(logits_fake)
                g_loss = logits_fake

                try:
                    d_weight = self.calculate_adaptive_weight(nll_loss, g_loss, last_layer=last_layer)
                except RuntimeError:
                    assert not self.training
                    d_weight = torch.tensor(0.0)

                loss = nll_loss + d_weight * disc_factor * g_loss
                total_loss += loss

                log.update({"{}/loss_{}".format(split, i): loss.clone().detach().mean(),
                            "{}/nll_loss_{}".format(split, i): nll_loss.detach().mean(),
                            "{}/rec_loss_{}".format(split, i): nll_loss.detach().mean(),
                            "{}/d_weight_{}".format(split, i): d_weight.detach(),
                            "{}/g_loss_{}".format(split, i): g_loss.detach().mean(),
                            })

            log.update({"{}/total_loss".format(split): total_loss.clone().detach().mean()})
            return total_loss, log

        if optimizer_idx == 1:
            # second pass for discriminator update
            total_disc_loss = torch.tensor([0.0]).cuda()
            log = {}
            for i, xrec in enumerate(reconstructions):
                if cond is None:
                    logits_real = self.discriminator(dis_x.contiguous().detach())  # Tensor(bs,1,30,30)
                    logits_fake = self.discriminator(xrec.contiguous().detach())  # Tensor(bs,1,30,30)
                else:  # !
                    assert self.disc_conditional
                    if self.multi_scale:
                        pred_real = self.discriminator(dis_x.contiguous().detach(), cond)   # list(num_D=2) of list(5)
                        pred_fake = self.discriminator(xrec.contiguous().detach(), cond)
                        logits_real = self.criterionGAN(pred_real, True)  # Tensor() 叠加两个g_loss
                        logits_fake = self.criterionGAN(pred_fake, False)
                        d_loss = (logits_real + logits_fake) * 0.5
                    else:
                        logits_real = self.discriminator(torch.cat((dis_x.contiguous().detach(), cond), dim=1))
                        logits_fake = self.discriminator(torch.cat((xrec.contiguous().detach(), cond), dim=1))
                        d_loss = self.disc_loss(logits_real, logits_fake)

                disc_factor = adopt_weight(self.disc_factor, global_step, threshold=self.discriminator_iter_start)
                d_loss = disc_factor * d_loss
                total_disc_loss += d_loss

                log.update({"{}/disc_loss_{}".format(split, i): d_loss.clone().detach().mean(),
                            "{}/logits_real_{}".format(split, i): logits_real.detach().mean(),
                            "{}/logits_fake_{}".format(split, i): logits_fake.detach().mean()
                            })

            log.update({"{}/total_disc_loss".format(split): total_disc_loss.clone().detach().mean()})
            return total_disc_loss, log


class GANLoss(nn.Module):
    def __init__(self, use_lsgan=True, target_real_label=1.0, target_fake_label=0.0,
                 tensor=torch.FloatTensor):
        super(GANLoss, self).__init__()
        self.real_label = target_real_label
        self.fake_label = target_fake_label
        self.real_label_var = None
        self.fake_label_var = None
        self.Tensor = tensor
        if use_lsgan:
            self.loss = nn.MSELoss()
        else:
            self.loss = nn.BCELoss()

    def get_target_tensor(self, input, target_is_real):
        target_tensor = None
        if target_is_real:
            create_label = ((self.real_label_var is None) or
                            (self.real_label_var.numel() != input.numel()))
            if create_label:
                real_tensor = self.Tensor(input.size()).fill_(self.real_label)
                self.real_label_var = Variable(real_tensor, requires_grad=False)
            target_tensor = self.real_label_var
        else:
            create_label = ((self.fake_label_var is None) or
                            (self.fake_label_var.numel() != input.numel()))
            if create_label:
                fake_tensor = self.Tensor(input.size()).fill_(self.fake_label)
                self.fake_label_var = Variable(fake_tensor, requires_grad=False)
            target_tensor = self.fake_label_var
        return target_tensor

    def __call__(self, input, target_is_real):   # list(2) of list(5)
        if isinstance(input[0], list):
            loss = 0
            for input_i in input:
                pred = input_i[-1]   # Tensor(bs,1,19,19)/Tensor(bs,1,11,11)
                target_tensor = self.get_target_tensor(pred, target_is_real)
                loss += self.loss(pred, target_tensor)
            return loss
        else:
            target_tensor = self.get_target_tensor(input[-1], target_is_real)
            return self.loss(input[-1], target_tensor)

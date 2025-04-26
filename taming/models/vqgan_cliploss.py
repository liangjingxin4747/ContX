import math
import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from PIL import Image
import torch.utils.tensorboard
from main import instantiate_from_config

from taming.modules.diffusionmodules.model import Encoder, Decoder
from taming.modules.vqvae.quantize import VectorQuantizer2 as VectorQuantizer
from taming.modules.vqvae.quantize import GumbelQuantize
from taming.modules.vqvae.quantize import EMAVectorQuantizer

from taming.modules.detectron2 import Conv2d, cat


cls_gray_dict = {}   # (87,)
with open("/home/liangxin01/code/VQGAN-CLIP/taming-transformers/concat_cat.txt", "r") as f:
    for cls, gray in enumerate(f.readlines()):
        gray_value = gray.strip()
        cls_gray_dict[cls] = int(gray_value)

class VQModel(pl.LightningModule):
    def __init__(self,
                 ddconfig,
                 lossconfig,
                 cliplossconfig,
                 n_embed,
                 embed_dim,
                 ckpt_path=None,
                 ignore_keys=[],
                 image_key="image",
                 colorize_nlabels=None,
                 monitor=None,
                 remap=None,
                 sane_index_shape=False,  # tell vector quantizer to return indices as bhw
                 clip_loss_weight=1
                 ):
        super().__init__()
        self.in_channels = ddconfig['in_channels']
        self.image_key = image_key
        self.encoder = Encoder(**ddconfig)
        self.decoder = Decoder(**ddconfig)
        self.loss = instantiate_from_config(lossconfig)
        self.cliploss = instantiate_from_config(cliplossconfig)
        self.quantize = VectorQuantizer(n_embed, embed_dim, beta=0.25,
                                        remap=remap, sane_index_shape=sane_index_shape)
        self.quant_conv = torch.nn.Conv2d(ddconfig["z_channels"], embed_dim, 1)
        self.noise_dim = 16
        # self.post_quant_conv = torch.nn.Conv2d(embed_dim + self.noise_dim, ddconfig["z_channels"], 1)
        self.post_quant_conv = torch.nn.Conv2d(embed_dim, ddconfig["z_channels"], 1)
        self.clip_loss_weight = clip_loss_weight

        if ckpt_path is not None:
            self.init_from_ckpt(ckpt_path, ignore_keys=ignore_keys)
        if colorize_nlabels is not None:
            assert type(colorize_nlabels)==int
            self.register_buffer("colorize", torch.randn(3, colorize_nlabels, 1, 1))
        if monitor is not None:
            self.monitor = monitor
        self.label_colours = np.random.randint(255, size=(self.in_channels, 3))

    def init_from_ckpt(self, path, ignore_keys=list()):
        sd = torch.load(path, map_location="cpu")["state_dict"]
        keys = list(sd.keys())
        for k in keys:
            for ik in ignore_keys:
                if k.startswith(ik):
                    print("Deleting key {} from state_dict.".format(k))
                    del sd[k]
        self.load_state_dict(sd, strict=False)
        print(f"Restored from {path}")

    def encode(self, x):   # Tensor(bs,87,256,256)
        h = self.encoder(x)   # Tensor(bs,256,16,16)
        h = self.quant_conv(h)   # Tensor(bs,256,16,16)
        quant, emb_loss, info = self.quantize(h)   # Tensor(bs,256,16,16)
        return quant, emb_loss, info

    def decode(self, quant):   # Tensor(bs,256,16,16)
        # noise = (torch.randn((quant.shape[0], self.noise_dim, 1))*8).to(self.device)   # Tensor(bs,16,1)
        # noise = noise.repeat(1, 1, quant.shape[2]*quant.shape[3]).view(-1, self.noise_dim, quant.shape[2], quant.shape[3])   # Tensor(bs,16,16,16)
        # quant = self.post_quant_conv(torch.cat([quant, noise], dim=1))   # Tensor(bs,256,16,16)
        quant = self.post_quant_conv(quant)  # Tensor(bs,256,16,16)
        dec = self.decoder(quant)   # Tensor(bs,87,256,256)
        return dec

    def decode_code(self, code_b):
        quant_b = self.quantize.embed_code(code_b)
        dec = self.decode(quant_b)
        return dec

    def forward(self, input):   # Tensor(bs,87,256,256)
        quant, diff, _ = self.encode(input)   # Tensor(bs,256,16,16)
        dec = self.decode(quant)   # Tensor(bs,87,256,256)
        return dec, diff

    def get_input(self, batch, k):    # {"image":Tensor(bs,256,256,87),"dis_image":,"file_path_":list(5),"dis_path_":}
        x = batch[k]   # Tensor(bs,256,256,87)
        if len(x.shape) == 3:
            x = x[..., None]
        x = x.permute(0, 3, 1, 2).to(memory_format=torch.contiguous_format)
        return x.float()   # Tensor(bs,87,256,256)

    def training_step(self, batch, batch_idx, optimizer_idx):
        x = self.get_input(batch, self.image_key)   # Tensor(bs,87,256,256)
        # dis_x is discriminator image.
        dis_x = self.get_input(batch, "dis_image")   # Tensor(bs,87,256,256)
        xrec, qloss = self(x)   # Tensor(bs,87,256,256)

        xrec_soft = self.softargmax(xrec)   # Tensor(bs,1,h,w)
        xrec_soft = self.replace(xrec_soft, x.detach().argmax(dim=1, keepdim=True))
        xrec_soft = self.cls2gray(xrec_soft)   # Tensor(bs,1,h,w)
        disx_hard = torch.argmax(dis_x, dim=1, keepdim=True)   # Tensor(bs,1,h,w)
        disx_hard = self.replace(disx_hard, x.detach().argmax(dim=1, keepdim=True))
        disx_hard = self.cls2gray(disx_hard)   # Tensor(bs,1,h,w)
        clip_loss = self.cliploss(xrec_soft, disx_hard, self.global_step, split='train')   # Tensor(1)
        
        if optimizer_idx == 0:
            # autoencode(generator)
            '''
            aeloss, log_dict_ae = self.loss(qloss, x, xrec, optimizer_idx, self.global_step,
                                            last_layer=self.get_last_layer(), split="train")
            '''
            aeloss, log_dict_ae = self.loss(qloss, dis_x, xrec, optimizer_idx, self.global_step,
                                            cond=x, last_layer=self.get_last_layer(), split="train")
            self.log("train/aeloss", aeloss, prog_bar=True, logger=True, on_step=True, on_epoch=True)
            log_dict_ae["train/clip_loss"] = clip_loss.clone().detach().mean()
            self.log_dict(log_dict_ae, prog_bar=False, logger=True, on_step=True, on_epoch=True)
            return aeloss + clip_loss * self.clip_loss_weight

        if optimizer_idx == 1:
            # discriminator
            '''
            discloss, log_dict_disc = self.loss(qloss, x, xrec, optimizer_idx, self.global_step,
                                            last_layer=self.get_last_layer(), split="train")
            '''
            discloss, log_dict_disc = self.loss(qloss, dis_x, xrec, optimizer_idx, self.global_step,
                                                cond=x, last_layer=self.get_last_layer(), split="train")
            self.log("train/discloss", discloss, prog_bar=True, logger=True, on_step=True, on_epoch=True)
            self.log_dict(log_dict_disc, prog_bar=False, logger=True, on_step=True, on_epoch=True)
            return discloss

    def validation_step(self, batch, batch_idx):
        x = self.get_input(batch, self.image_key)   # Tensor(bs,87,256,256)
        # dis_x is discriminator image.
        dis_x = self.get_input(batch, "dis_image")   # Tensor(bs,87,256,256)
        xrec, qloss = self(x)   # Tensor(bs,87,256,256)
        '''
        aeloss, log_dict_ae = self.loss(qloss, x, xrec, 0, self.global_step,
                                             last_layer=self.get_last_layer(), split="val")
        discloss, log_dict_disc = self.loss(qloss, x, xrec, 1, self.global_step,
                                            last_layer=self.get_last_layer(), split="val")
        '''
        xrec_soft = self.softargmax(xrec)   # Tensor(bs,1,h,w)
        xrec_soft = self.replace(xrec_soft, x.detach().argmax(dim=1, keepdim=True))
        xrec_soft = self.cls2gray(xrec_soft)   # Tensor(bs,1,h,w)
        disx_hard = torch.argmax(dis_x, dim=1, keepdim=True)   # Tensor(bs,1,h,w)
        disx_hard = self.replace(disx_hard, x.detach().argmax(dim=1, keepdim=True))
        disx_hard = self.cls2gray(disx_hard)   # Tensor(bs,1,h,w)
        clip_loss = self.cliploss(xrec_soft, disx_hard, self.global_step, split='val')   # Tensor(1)
        
        aeloss, log_dict_ae = self.loss(qloss, dis_x, xrec, 0, self.global_step,
                                        cond=x, last_layer=self.get_last_layer(), split="val")

        discloss, log_dict_disc = self.loss(qloss, dis_x, xrec, 1, self.global_step,
                                            cond=x, last_layer=self.get_last_layer(), split="val")
        # rec_loss = log_dict_ae["val/rec_loss"]
        # self.log("val/rec_loss", rec_loss,
        #            prog_bar=True, logger=True, on_step=True, on_epoch=True, sync_dist=True)
        self.log("val/aeloss", aeloss,
                   prog_bar=True, logger=True, on_step=True, on_epoch=True, sync_dist=True)
        self.log_dict({"val/clip_loss": clip_loss.clone().detach().mean()})
        self.log_dict(log_dict_ae)
        self.log_dict(log_dict_disc)
        return self.log_dict

    def softargmax(self, x):   # Tensor(bs,87,h,w)->Tensor(bs,1,h,w)
        bs, c, h, w = x.shape
        indices = torch.arange(c)[None,:,None,None].repeat(bs,1,h,w).to(self.device)   # Tensor(bs,87,h,w)
        x_soft = torch.sum(x.softmax(dim=1) * indices, dim=1, keepdim=True)   # Tensor(bs,1,h,w)
        x_hard = x.argmax(dim=1, keepdim=True)
        return x_soft + (x_hard - x_soft).detach()   # value=x_hard, grad=x_soft
    
    def replace(self, bg, fg):
        fg = fg.to(bg.dtype)
        return torch.where(fg == 86., bg, fg)
    
    def cls2gray(self, x):
        x_detach = x.detach().to(torch.long)
        for cls in sorted(cls_gray_dict, reverse=True):
            gray = cls_gray_dict[cls]
            x_detach = torch.where(x_detach == cls, gray, x_detach)
        return x + (x_detach - x).detach()
    
    def configure_optimizers(self):
        lr = self.learning_rate
        opt_ae = torch.optim.Adam(list(self.encoder.parameters())+
                                  list(self.decoder.parameters())+
                                  list(self.quantize.parameters())+
                                  list(self.quant_conv.parameters())+
                                  list(self.post_quant_conv.parameters()),
                                  lr=lr, betas=(0.5, 0.9))
        opt_disc = torch.optim.Adam(self.loss.discriminator.parameters(),
                                    lr=lr, betas=(0.5, 0.9))
        return [opt_ae, opt_disc], []

    def get_last_layer(self):
        return self.decoder.conv_out.weight

    def log_images(self, batch, **kwargs):
        log = dict()
        x = self.get_input(batch, self.image_key)   # input
        x = x.to(self.device)
        # dis_x is discriminator image.
        dis_x = self.get_input(batch, "dis_image")
        dis_x = dis_x.to(self.device)
        xrec, _ = self(x)

        ignore, target = torch.max(dis_x, 1)
        ignore2, target2 = torch.max(xrec, 1)
        ignore2, target3 = torch.max(x, 1)

        dis_x = target.cpu().numpy()
        xrec = target2.cpu().numpy()

        dis_x.astype(np.uint8)
        xrec.astype(np.uint8)

        inputs = target3.cpu().numpy()
        inputs.astype(np.uint8)
        n, w, h = xrec.shape
        
        for q in range(n):
            for i in range(w):
                for j in range(h):
                    if inputs[q][i][j] != self.in_channels-1:
                        xrec[q][i][j] = inputs[q][i][j]
                        dis_x[q][i][j] = inputs[q][i][j]

        # label_colours = np.random.randint(255, size=(self.in_channels, 3))

        dis_x = np.array([self.label_colours[c % self.in_channels] for c in dis_x])
        xrec = np.array([self.label_colours[c % self.in_channels] for c in xrec])
        inputs = np.array([self.label_colours[c % self.in_channels] for c in inputs])

        dis_x = torch.from_numpy(dis_x)
        xrec = torch.from_numpy(xrec)
        inputs = torch.from_numpy(inputs)

        xrec = xrec.permute(0, 3, 1, 2).to(memory_format=torch.contiguous_format)
        dis_x = dis_x.permute(0, 3, 1, 2).to(memory_format=torch.contiguous_format)
        inputs = inputs.permute(0, 3, 1, 2).to(memory_format=torch.contiguous_format)
        '''
        if x.shape[1] > 3:
            # colorize with random projection
            assert xrec.shape[1] > 3
            x = self.to_rgb(x)
            xrec = self.to_rgb(xrec)
            dis_x = self.to_rgb(dis_x)
        '''

        log["inputs"] = inputs
        log["ground_truth"] = dis_x
        log["reconstructions"] = xrec
        return log

    def to_rgb(self, x):
        # assert self.image_key == "segmentation"
        if not hasattr(self, "colorize"):
            self.register_buffer("colorize", torch.randn(3, x.shape[1], 1, 1).to(x))
        x = F.conv2d(x, weight=self.colorize)
        x = 2.*(x-x.min())/(x.max()-x.min()) - 1.
        return x


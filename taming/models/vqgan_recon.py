import math
import os
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from PIL import Image
from torchvision.transforms import Resize
# from torchvision.transforms.functional import InterpolationMode
import torch.utils.tensorboard
from main import instantiate_from_config

from taming.modules.diffusionmodules.model import Encoder, Decoder
from taming.modules.vqvae.quantize import VectorQuantizer2 as VectorQuantizer
from taming.modules.vqvae.quantize import GumbelQuantize
from taming.modules.vqvae.quantize import EMAVectorQuantizer

from taming.modules.detectron2 import Conv2d, cat
from taming.models.vqgan import VQModel


def nonlinearity(x):
    # swish
    return x*torch.sigmoid(x)

def Normalize(in_channels):
    return torch.nn.GroupNorm(num_groups=32, num_channels=in_channels, eps=1e-6, affine=True)


# class VQModel(pl.LightningModule):
    def __init__(self,
                 ddconfig,
                 lossconfig,
                 n_embed,
                 embed_dim,
                 ckpt_path=None,
                 ignore_keys=[],
                 image_key="image",
                 colorize_nlabels=None,
                 monitor=None,
                 remap=None,
                 sane_index_shape=False,  # tell vector quantizer to return indices as bhw
                 ):
        super().__init__()
        self.in_channels = ddconfig['in_channels']
        self.image_key = image_key
        self.encoder = Encoder(**ddconfig)
        self.decoder = Decoder(**ddconfig)
        self.loss = instantiate_from_config(lossconfig)
        self.quantize = VectorQuantizer(n_embed, embed_dim, beta=0.25,
                                        remap=remap, sane_index_shape=sane_index_shape)
        self.quant_conv = torch.nn.Conv2d(ddconfig["z_channels"], embed_dim, 1)
        self.noise_dim = 16
        # self.post_quant_conv = torch.nn.Conv2d(embed_dim + self.noise_dim, ddconfig["z_channels"], 1)
        self.post_quant_conv = torch.nn.Conv2d(embed_dim, ddconfig["z_channels"], 1)

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

        if optimizer_idx == 0:
            # autoencode(generator)
            '''
            aeloss, log_dict_ae = self.loss(qloss, x, xrec, optimizer_idx, self.global_step,
                                            last_layer=self.get_last_layer(), split="train")
            '''
            aeloss, log_dict_ae = self.loss(qloss, dis_x, xrec, optimizer_idx, self.global_step,
                                            cond=x, last_layer=self.get_last_layer(), split="train")
            self.log("train/aeloss", aeloss, prog_bar=True, logger=True, on_step=True, on_epoch=True)
            self.log_dict(log_dict_ae, prog_bar=False, logger=True, on_step=True, on_epoch=True)
            return aeloss

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
        aeloss, log_dict_ae = self.loss(qloss, dis_x, xrec, 0, self.global_step,
                                        cond=x, last_layer=self.get_last_layer(), split="val")

        discloss, log_dict_disc = self.loss(qloss, dis_x, xrec, 1, self.global_step,
                                            cond=x, last_layer=self.get_last_layer(), split="val")
        # rec_loss = log_dict_ae["val/rec_loss"]
        # self.log("val/rec_loss", rec_loss,
        #            prog_bar=True, logger=True, on_step=True, on_epoch=True, sync_dist=True)
        self.log("val/aeloss", aeloss,
                   prog_bar=True, logger=True, on_step=True, on_epoch=True, sync_dist=True)
        self.log_dict(log_dict_ae)
        self.log_dict(log_dict_disc)
        return self.log_dict

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


class VQModelwithSPRef(pl.LightningModule):
    def __init__(self,
                 ddconfig,
                 lossconfig,
                 cliplossconfig,
                 recon_net,
                 n_embed,
                 embed_dim,
                 ckpt_path=None,
                 ignore_keys=[],
                 image_key="image",
                 colorize_nlabels=None,
                 monitor=None,
                 remap=None,
                 sane_index_shape=False,  # tell vector quantizer to return indices as bhw
                 noise_dim=0,
                 use_memory=False,
                 clip_loss_weight=1.
                 ):
        super().__init__()
        self.vqgan = VQModel(ddconfig,
                 lossconfig,
                 n_embed,
                 embed_dim,
                 ckpt_path,
                 noise_dim=noise_dim)
        self.in_channels = ddconfig['in_channels']
        self.out_channels = ddconfig['out_ch']
        self.image_key = image_key
        self.decoder = Decoder(**ddconfig)
        # self.decoder.load_state_dict(self.vqgan.decoder.state_dict())
        # self.post_quant_conv = self.vqgan.post_quant_conv
        # self.loss = self.vqgan.loss   # instantiate_from_config(lossconfig)
        self.post_quant_conv = torch.nn.Conv2d(ddconfig["z_channels"], ddconfig["z_channels"], 1)
        self.loss = instantiate_from_config(lossconfig)
        self.cliploss = instantiate_from_config(cliplossconfig)
        # self.cliploss = None
        self.clip_loss_weight = clip_loss_weight
        if colorize_nlabels is not None:
            assert type(colorize_nlabels) == int
            self.register_buffer("colorize", torch.randn(3, colorize_nlabels, 1, 1))
        if monitor is not None:
            self.monitor = monitor
        self.label_colours = np.random.randint(255, size=(self.in_channels, 3))

        self.recon_net = instantiate_from_config(recon_net)   # General_Recon_Net
        self.recon_net.load_state_dict(torch.load(os.path.join(recon_net.params.ckpt_path, "memory.pth"), map_location=self.device))

        self.memorybank = np.load(os.path.join(recon_net.params.ckpt_path, 'cluster_memory_bank.npy'), allow_pickle=True)[()]
        
        self.freeze_model(self.recon_net)
        # self.freeze_model(self.vqgan.encoder)
        # self.freeze_model(self.vqgan.quantize)
        # self.freeze_model(self.vqgan.quant_conv)
        # self.freeze_model(self.vqgan.decoder)
        # self.freeze_model(self.vqgan.loss)
        # self.freeze_model(self.vqgan.post_quant_conv)
        self.SPRef = True   # recon_net.params.memory_refine
        self.SPk = 1   # recon_net.params.memory_refine_k
        self.noise_dim = noise_dim
        self.use_memory = use_memory
        
        self.sp_conv = nn.Sequential(
            Conv2d(
            87*8,   # ddconfig.z_channels + self.SPk * self.out_channels
            87*8,
            kernel_size=3,
            stride=2,
            padding=1
            ),
            nn.ReLU(),
            Conv2d(
            87*8,   # ddconfig.z_channels + self.SPk * self.out_channels
            512,
            kernel_size=3,
            stride=1,   # 2
            padding=1
            ),
            nn.ReLU(),
            Conv2d(
                512,
                256,
                kernel_size=3,
                stride=1,
                padding=1
            )
        )
        self.fuse_layer = nn.Sequential(
            # Normalize(self.out_channels),
            torch.nn.Conv2d(512,
                            512,
                            kernel_size=3,
                            stride=1,
                            padding=1),
            nn.ReLU(),
            torch.nn.Conv2d(512,
                            256,
                            kernel_size=3,
                            stride=1,
                            padding=1),
        )
        
        with open("/home/liangxin01/code/dyh/model/taming-transformer-origin/concat_cat.txt", "r") as f:
        # with open("/home/liangxin01/code/datasets/coco-stuff/my_category.txt", "r") as f:
            classes = f.read().splitlines()
        self.cls_gray_dict = dict()
        for i, cls in enumerate(classes):
            self.cls_gray_dict[i] = int(cls)
        self.cls2gray_dict = dict()
        self.gray2cls_dict = dict()
        with open("/home/liangxin01/code/dyh/model/taming-transformer-origin/concat_cat.txt", "r") as f:
        # with open("/home/liangxin01/code/datasets/coco-stuff/my_category.txt", "r") as f:
            cat_f = f.read().splitlines()
            for cls, gray in enumerate(cat_f):
                self.cls2gray_dict[cls] = int(gray)
                self.gray2cls_dict[int(gray)] = cls
                
    def cls2gray(self, cls_list):
        gray_list = []
        for cls in cls_list:
            gray = self.cls2gray_dict[cls]
            gray_list.append(gray)
        return gray_list

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

    def encode(self, x):  # Tensor(bs,87,256,256)
        h = self.encoder(x)  # Tensor(bs,256,16,16)
        h = self.quant_conv(h)  # Tensor(bs,256,16,16)
        quant, emb_loss, info = self.quantize(h)  # Tensor(bs,256,16,16)
        return quant, emb_loss, info

    def decode(self, quant):  # Tensor(bs,256,16,16)
        if self.noise_dim != 0:
            noise = quant.new_empty(quant.shape[0], self.noise_dim, 1).normal_().to(self.device)   # Tensor(bs,16,1)
            noise = noise.repeat(1, 1, quant.shape[2]*quant.shape[3]).view(-1, self.noise_dim, quant.shape[2], quant.shape[3])   # Tensor(bs,16,16,16)
            quant = torch.cat([quant, noise], dim=1)   # Tensor(bs,256,16,16)
        else:
            noise = None
        quant = self.post_quant_conv(quant)  # Tensor(bs,256,16,16)
        dec = self.decoder(quant)   # Tensor(bs,87,256,256)
        return dec   # revised
        return dec, noise

    def decode_code(self, code_b):
        quant_b = self.quantize.embed_code(code_b)
        dec = self.decode(quant_b)
        return dec

    def freeze_model(self, model):
        for param in model.parameters():
            param.requires_grad = False

    def get_shape_prior(self, pred_context_logits):   # Tensor(bs,87,h,w)
        B, _, H, W = pred_context_logits.shape
        # choose classes in input
        coarse_context = torch.argmax(pred_context_logits, dim=1)   # Tensor(bs,size,size)
        input_u_classes = [torch.unique(context).tolist() for context in coarse_context]   # list(bs)
        
        # get sp_masks
        spmask_per_context = torch.zeros((B, 87, 8, H//8, W//8)).to(self.device)   # Tensor(B,87,8,H/8,W/8)
        for bs, (cls_list, context) in enumerate(zip(input_u_classes, coarse_context)):   # list(num_cls) Tensor(size,size)
            spmask = torch.zeros((87, 8, H//8, W//8))
            num_cls = len(cls_list)
            masks_per_context = torch.zeros((num_cls, 1, H, W))   # Tensor(num_cls,1,size,size)
            context_cls_list = []
            for cls in cls_list:
                if self.cls2gray_dict[cls] not in self.memorybank.keys():
                    continue
                mask = torch.where(context == cls, 1, 0)   # Tensor(size,size)
                masks_per_context[len(context_cls_list)][0] = mask
                context_cls_list.append(cls)
            masks_per_context = masks_per_context[:len(context_cls_list)]
            z_e = self.recon_net.encode(masks_per_context.to(self.device))   # Tensor(num_cls,8,size/8,size/8)
            if self.use_memory:
                mb = [self.memorybank[self.cls2gray_dict[cls]] for cls in context_cls_list]
                z_q = self.recon_net.vq(z_e, mb)   # Tensor(num_cls,8,32,32)
            else:
                z_q = z_e
            for i, (cls, sp) in enumerate(zip(cls_list, z_q)):   # list(num_cls) Tensor(num_cls,8,size/4,size/4)
                spmask[cls] = sp   # Tensor(8,size/8,size/8)
            spmask_per_context[bs] = spmask.to(self.device)
        return spmask_per_context  # Tensor(B,87,8,H/8,W/8)

    def forward(self, input):  # Tensor(bs,87,256,256)
        # vqgan
        # quant, diff, _ = self.vqgan.encode(input)  # Tensor(bs,256,h/16,w/16)
        # coarse_context, _ = self.vqgan.decode(quant)  # Tensor(bs,87,h,w)
        start_time = time.time()
        quant = self.vqgan.encode(input)  # Tensor(bs,256,h/16,w/16) revised
        coarse_context = self.vqgan.decode(quant)  # Tensor(bs,87,h,w) revised
        coarse_duration = time.time() - start_time
        
        start_time = time.time()
        # shape_prior recon_net  # (8,h,w)
        sp_features = self.get_shape_prior(coarse_context)   # Tensor(B,87,8,32,32)
        B, _, _, h, w = sp_features.shape
        sp_features = self.sp_conv(sp_features.view(B, -1, h, w))   # Tensor(B,256,16,16)
        quant = self.fuse_layer(torch.cat([quant, sp_features], dim=1))
        dec, noise = self.decode(quant)
        fine_duration = time.time() - start_time
        if self.noise_dim != 0:
            dec1, noise1 = self.decode(quant)   # Tensor(bs,87,256,256)
            return [dec, dec1, noise, noise1], diff
        else:
            return dec, diff

    def get_input(self, batch, k):  # {"image":Tensor(bs,256,256,87),"dis_image":,"file_path_":list(5),"dis_path_":}
        x = batch[k]  # Tensor(bs,256,256,87)
        if len(x.shape) == 3:
            x = x[..., None]
        x = x.permute(0, 3, 1, 2).to(memory_format=torch.contiguous_format)
        return x.float()  # Tensor(bs,87,256,256)

    def training_step(self, batch, batch_idx, optimizer_idx):
        
        x = self.get_input(batch, self.image_key)  # Tensor(bs,87,256,256)
        # dis_x is discriminator image.
        dis_x = self.get_input(batch, "dis_image")  # Tensor(bs,87,256,256)
        xrec, qloss = self(x)  # Tensor(bs,87,256,256)

        if optimizer_idx == 0:
            # autoencode(generator)
            '''
            aeloss, log_dict_ae = self.loss(qloss, x, xrec, optimizer_idx, self.global_step,
                                            last_layer=self.get_last_layer(), split="train")
            '''
            aeloss, log_dict_ae = self.loss(qloss, dis_x, xrec, optimizer_idx, self.global_step,
                                            cond=x, last_layer=self.get_last_layer(), split="train",
                                            clip=self.cliploss, clip_weight=self.clip_loss_weight)
            
            self.log("train/aeloss", aeloss, prog_bar=True, logger=True, on_step=True, on_epoch=True)
            self.log_dict(log_dict_ae, prog_bar=False, logger=True, on_step=True, on_epoch=True)
            return aeloss

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
        x = self.get_input(batch, self.image_key)  # Tensor(bs,87,256,256)
        # dis_x is discriminator image.
        dis_x = self.get_input(batch, "dis_image")  # Tensor(bs,87,256,256)
        xrec, qloss = self(x)  # Tensor(bs,87,256,256)
        '''
        aeloss, log_dict_ae = self.loss(qloss, x, xrec, 0, self.global_step,
                                             last_layer=self.get_last_layer(), split="val")
        discloss, log_dict_disc = self.loss(qloss, x, xrec, 1, self.global_step,
                                            last_layer=self.get_last_layer(), split="val")
        '''
        aeloss, log_dict_ae = self.loss(qloss, dis_x, xrec, 0, self.global_step,
                                        cond=x, last_layer=self.get_last_layer(), split="val",
                                        clip=self.cliploss, clip_weight=self.clip_loss_weight)

        discloss, log_dict_disc = self.loss(qloss, dis_x, xrec, 1, self.global_step,
                                            cond=x, last_layer=self.get_last_layer(), split="val")
        
        # rec_loss = log_dict_ae["val/rec_loss"]
        # self.log("val/rec_loss", rec_loss,
        #          prog_bar=True, logger=True, on_step=True, on_epoch=True, sync_dist=True)
        self.log("val/aeloss", aeloss,
                 prog_bar=True, logger=True, on_step=True, on_epoch=True, sync_dist=True)
        self.log_dict(log_dict_ae)
        self.log_dict(log_dict_disc)
        return self.log_dict

    def configure_optimizers(self):
        lr = self.learning_rate
        # opt_ae = torch.optim.Adam(list(self.decoder.parameters()) +
        #                           list(self.post_quant_conv.parameters()) +
        #                           list(self.fuse_layer.parameters()) +
        #                           list(self.sp_conv.parameters()),
        #                           lr=lr, betas=(0.5, 0.9))
        opt_ae = torch.optim.Adam(
                                  list(self.vqgan.parameters()) +
                                  list(self.decoder.parameters()) +
                                  list(self.post_quant_conv.parameters()) +
                                  list(self.fuse_layer.parameters()) +
                                  list(self.sp_conv.parameters()),
                                  lr=lr, betas=(0.5, 0.9))
        opt_disc = torch.optim.Adam(self.loss.discriminator.parameters(),
                                    lr=lr, betas=(0.5, 0.9))
        return [opt_ae, opt_disc], []

    def get_last_layer(self):
        return self.decoder.conv_out.weight

    def log_images(self, batch, **kwargs):
        log = dict()
        x = self.get_input(batch, self.image_key)  # input
        x = x.to(self.device)
        # dis_x is discriminator image.
        dis_x = self.get_input(batch, "dis_image")
        dis_x = dis_x.to(self.device)
        reconstructions, _ = self(x)
        if isinstance(reconstructions, list):
            reconstructions = reconstructions[0]

        _, target = torch.max(dis_x, 1)
        _, target3 = torch.max(x, 1)
        dis_x = target.cpu().numpy()
        dis_x.astype(np.uint8)
        inputs = target3.cpu().numpy()
        inputs.astype(np.uint8)

        dis_x = np.where(inputs != self.in_channels -1, inputs, dis_x)

        dis_x_rgb = np.array([self.label_colours[c % self.in_channels] for c in dis_x])
        inputs_rgb = np.array([self.label_colours[c % self.in_channels] for c in inputs])

        dis_x_rgb = torch.from_numpy(dis_x_rgb).permute(0, 3, 1, 2).to(memory_format=torch.contiguous_format)
        inputs_rgb = torch.from_numpy(inputs_rgb).permute(0, 3, 1, 2).to(memory_format=torch.contiguous_format)

        log["inputs"] = inputs_rgb
        log["ground_truth"] = dis_x_rgb

        if not isinstance(reconstructions, list):
            reconstructions = [reconstructions]
        for i, xrec in enumerate(reconstructions):
            _, target2 = torch.max(xrec, 1)

            xrec = target2.cpu().numpy()
            xrec.astype(np.uint8)

            xrec = np.where(inputs != self.in_channels - 1, inputs, xrec)

            xrec_rgb = np.array([self.label_colours[c % self.in_channels] for c in xrec])
            xrec_rgb = torch.from_numpy(xrec_rgb).permute(0, 3, 1, 2).to(memory_format=torch.contiguous_format)

            log[f"reconstructions_{i}"] = xrec_rgb

        return log

    def to_rgb(self, x):
        # assert self.image_key == "segmentation"
        if not hasattr(self, "colorize"):
            self.register_buffer("colorize", torch.randn(3, x.shape[1], 1, 1).to(x))
        x = F.conv2d(x, weight=self.colorize)
        x = 2. * (x - x.min()) / (x.max() - x.min()) - 1.
        return x


class VQSegmentationModel(VQModel):
    def __init__(self, n_labels, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.register_buffer("colorize", torch.randn(3, n_labels, 1, 1))

    def configure_optimizers(self):
        lr = self.learning_rate
        opt_ae = torch.optim.Adam(list(self.encoder.parameters())+
                                  list(self.decoder.parameters())+
                                  list(self.quantize.parameters())+
                                  list(self.quant_conv.parameters())+
                                  list(self.post_quant_conv.parameters()),
                                  lr=lr, betas=(0.5, 0.9))
        return opt_ae

    def training_step(self, batch, batch_idx):
        x = self.get_input(batch, self.image_key)   # Tensor(bs,87,256,256)
        xrec, qloss = self(x)
        # dis_x is discriminator image.
        dis_x = self.get_input(batch, "dis_image")
        # aeloss, log_dict_ae = self.loss(qloss, x, xrec, split="train")
        aeloss, log_dict_ae = self.loss(qloss, dis_x, xrec, split="train")
        self.log_dict(log_dict_ae, prog_bar=False, logger=True, on_step=True, on_epoch=True)
        return aeloss

    def validation_step(self, batch, batch_idx):
        x = self.get_input(batch, self.image_key)   # Tensor(bs,87,256,256)
        xrec, qloss = self(x)   # Tensor(bs,87,256,256), Tensor()
        # dis_x is discriminator image.
        dis_x = self.get_input(batch, "dis_image")   # Tensor(bs,87,256,256)
        # aeloss, log_dict_ae = self.loss(qloss, x, xrec, split="val")
        aeloss, log_dict_ae = self.loss(qloss, dis_x, xrec, split="val")
        self.log_dict(log_dict_ae, prog_bar=False, logger=True, on_step=True, on_epoch=True)
        total_loss = log_dict_ae["val/total_loss"]
        self.log("val/total_loss", total_loss,
                 prog_bar=True, logger=True, on_step=True, on_epoch=True, sync_dist=True)
        return aeloss

    @torch.no_grad()
    def log_images(self, batch, **kwargs):
        log = dict()
        x = self.get_input(batch, self.image_key)
        x = x.to(self.device)
        xrec, _ = self(x)
        # dis_x is discriminator image.
        dis_x = self.get_input(batch, "dis_image").to(self.device)
        if x.shape[1] > 3:
            # colorize with random projection
            assert xrec.shape[1] > 3
            # convert logits to indices
            xrec = torch.argmax(xrec, dim=1, keepdim=True)
            xrec = F.one_hot(xrec, num_classes=x.shape[1])
            xrec = xrec.squeeze(1).permute(0, 3, 1, 2).float()
            x = self.to_rgb(x)
            xrec = self.to_rgb(xrec)
            dis_x = self.to_rgb(dis_x)
        log["inputs"] = x
        log["dis_image"] = dis_x
        log["reconstructions"] = xrec
        return log


class VQSegGANModel(VQModel):
    def __init__(self, n_labels, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.register_buffer("colorize", torch.randn(3, n_labels, 1, 1))

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

    def training_step(self, batch, batch_idx, optimizer_idx):
        x = self.get_input(batch, self.image_key)   # Tensor(bs,87,256,256)
        # dis_x is discriminator image.
        dis_x = self.get_input(batch, "dis_image")
        xrec, qloss = self(x)

        if optimizer_idx == 0:
            # autoencode(generator)
            aeloss, log_dict_ae = self.loss(qloss, dis_x, xrec, optimizer_idx, self.global_step, split="train")
            self.log("train/aeloss", aeloss, prog_bar=True, logger=True, on_step=True, on_epoch=True)
            self.log_dict(log_dict_ae, prog_bar=False, logger=True, on_step=True, on_epoch=True)
            return aeloss

        if optimizer_idx == 1:
            # autoencode(discriminator)
            discloss, log_dict_disc = self.loss(qloss, dis_x, xrec, optimizer_idx, self.global_step, split="train")
            self.log("train/discloss", discloss, prog_bar=True, logger=True, on_step=True, on_epoch=True)
            self.log_dict(log_dict_disc, prog_bar=False, logger=True, on_step=True, on_epoch=True)
            return discloss


    def validation_step(self, batch, batch_idx):
        x = self.get_input(batch, self.image_key)   # Tensor(bs,87,256,256)
        # dis_x is discriminator image.
        dis_x = self.get_input(batch, "dis_image")   # Tensor(bs,87,256,256)
        xrec, qloss = self(x)   # Tensor(bs,87,256,256), Tensor()

        # aeloss, log_dict_ae = self.loss(qloss, x, xrec, split="val")
        aeloss, log_dict_ae = self.loss(qloss, dis_x, xrec, 0, self.global_step, split="val")
        self.log("val/aeloss", aeloss, prog_bar=True, logger=True, on_step=True, on_epoch=True)
        self.log_dict(log_dict_ae, prog_bar=False, logger=True, on_step=True, on_epoch=True)

        discloss, log_dict_disc = self.loss(qloss, dis_x, xrec, 1, self.global_step, split="val")
        self.log("val/discloss", discloss, prog_bar=True, logger=True, on_step=True, on_epoch=True)
        self.log_dict(log_dict_disc, prog_bar=False, logger=True, on_step=True, on_epoch=True)

        total_loss = aeloss + discloss
        self.log("val/total_loss", total_loss,
                 prog_bar=True, logger=True, on_step=True, on_epoch=True, sync_dist=True)
        return self.log_dict

    @torch.no_grad()
    def log_images(self, batch, **kwargs):
        log = dict()
        x = self.get_input(batch, self.image_key)
        x = x.to(self.device)
        xrec, _ = self(x)
        # dis_x is discriminator image.
        dis_x = self.get_input(batch, "dis_image").to(self.device)
        if x.shape[1] > 3:
            # colorize with random projection
            assert xrec.shape[1] > 3
            # convert logits to indices
            xrec = torch.argmax(xrec, dim=1, keepdim=True)
            xrec = F.one_hot(xrec, num_classes=x.shape[1])
            xrec = xrec.squeeze(1).permute(0, 3, 1, 2).float()
            x = self.to_rgb(x)
            xrec = self.to_rgb(xrec)
            dis_x = self.to_rgb(dis_x)
        log["inputs"] = x
        log["dis_image"] = dis_x
        log["reconstructions"] = xrec
        return log


class VQNoDiscModel(VQModel):
    def __init__(self,
                 ddconfig,
                 lossconfig,
                 n_embed,
                 embed_dim,
                 ckpt_path=None,
                 ignore_keys=[],
                 image_key="image",
                 colorize_nlabels=None
                 ):
        super().__init__(ddconfig=ddconfig, lossconfig=lossconfig, n_embed=n_embed, embed_dim=embed_dim,
                         ckpt_path=ckpt_path, ignore_keys=ignore_keys, image_key=image_key,
                         colorize_nlabels=colorize_nlabels)

    def training_step(self, batch, batch_idx):
        x = self.get_input(batch, self.image_key)
        xrec, qloss = self(x)
        # dis_x is discriminator image.
        dis_x = self.get_input(batch, "dis_image")
        # autoencode
        # aeloss, log_dict_ae = self.loss(qloss, x, xrec, self.global_step, split="train")
        aeloss, log_dict_ae = self.loss(qloss, dis_x, xrec, self.global_step, split="train")
        output = pl.TrainResult(minimize=aeloss)
        output.log("train/aeloss", aeloss,
                   prog_bar=True, logger=True, on_step=True, on_epoch=True)
        output.log_dict(log_dict_ae, prog_bar=False, logger=True, on_step=True, on_epoch=True)
        return output

    def validation_step(self, batch, batch_idx):
        x = self.get_input(batch, self.image_key)
        xrec, qloss = self(x)
        # dis_x is discriminator image.
        dis_x = self.get_input(batch, "dis_image")
        # aeloss, log_dict_ae = self.loss(qloss, x, xrec, self.global_step, split="val")
        aeloss, log_dict_ae = self.loss(qloss, dis_x, xrec, self.global_step, split="val")
        rec_loss = log_dict_ae["val/rec_loss"]
        output = pl.EvalResult(checkpoint_on=rec_loss)
        output.log("val/rec_loss", rec_loss,
                   prog_bar=True, logger=True, on_step=True, on_epoch=True)
        output.log("val/aeloss", aeloss,
                   prog_bar=True, logger=True, on_step=True, on_epoch=True)
        output.log_dict(log_dict_ae)

        return output

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(list(self.encoder.parameters())+
                                  list(self.decoder.parameters())+
                                  list(self.quantize.parameters())+
                                  list(self.quant_conv.parameters())+
                                  list(self.post_quant_conv.parameters()),
                                  lr=self.learning_rate, betas=(0.5, 0.9))
        return optimizer


class GumbelVQ(VQModel):
    def __init__(self,
                 ddconfig,
                 lossconfig,
                 n_embed,
                 embed_dim,
                 temperature_scheduler_config,
                 ckpt_path=None,
                 ignore_keys=[],
                 image_key="image",
                 colorize_nlabels=None,
                 monitor=None,
                 kl_weight=1e-8,
                 remap=None,
                 ):

        z_channels = ddconfig["z_channels"]
        super().__init__(ddconfig,
                         lossconfig,
                         n_embed,
                         embed_dim,
                         ckpt_path=None,
                         ignore_keys=ignore_keys,
                         image_key=image_key,
                         colorize_nlabels=colorize_nlabels,
                         monitor=monitor,
                         )

        self.loss.n_classes = n_embed
        self.vocab_size = n_embed

        self.quantize = GumbelQuantize(z_channels, embed_dim,
                                       n_embed=n_embed,
                                       kl_weight=kl_weight, temp_init=1.0,
                                       remap=remap)

        self.temperature_scheduler = instantiate_from_config(temperature_scheduler_config)   # annealing of temp

        if ckpt_path is not None:
            self.init_from_ckpt(ckpt_path, ignore_keys=ignore_keys)

    def temperature_scheduling(self):
        self.quantize.temperature = self.temperature_scheduler(self.global_step)

    def encode_to_prequant(self, x):
        h = self.encoder(x)
        h = self.quant_conv(h)
        return h

    def decode_code(self, code_b):
        raise NotImplementedError

    def training_step(self, batch, batch_idx, optimizer_idx):
        self.temperature_scheduling()
        x = self.get_input(batch, self.image_key)
        xrec, qloss = self(x)
        # dis_x is discriminator image.
        dis_x = self.get_input(batch, "dis_image")
        if optimizer_idx == 0:
            # autoencode
            '''
            aeloss, log_dict_ae = self.loss(qloss, x, xrec, optimizer_idx, self.global_step,
                                            last_layer=self.get_last_layer(), split="train")
            '''
            aeloss, log_dict_ae = self.loss(qloss, dis_x, xrec, optimizer_idx, self.global_step,
                                            last_layer=self.get_last_layer(), split="train")

            self.log_dict(log_dict_ae, prog_bar=False, logger=True, on_step=True, on_epoch=True)
            self.log("temperature", self.quantize.temperature, prog_bar=False, logger=True, on_step=True, on_epoch=True)
            return aeloss

        if optimizer_idx == 1:
            # discriminator
            '''
            discloss, log_dict_disc = self.loss(qloss, x, xrec, optimizer_idx, self.global_step,
                                            last_layer=self.get_last_layer(), split="train")
            '''
            discloss, log_dict_disc = self.loss(qloss, dis_x, xrec, optimizer_idx, self.global_step,
                                            last_layer=self.get_last_layer(), split="train")
            self.log_dict(log_dict_disc, prog_bar=False, logger=True, on_step=True, on_epoch=True)
            return discloss

    def validation_step(self, batch, batch_idx):
        x = self.get_input(batch, self.image_key)
        xrec, qloss = self(x, return_pred_indices=True)
        # dis_x is discriminator image.
        dis_x = self.get_input(batch, "dis_image")
        '''
        aeloss, log_dict_ae = self.loss(qloss, x, xrec, 0, self.global_step,
                                        last_layer=self.get_last_layer(), split="val")

        discloss, log_dict_disc = self.loss(qloss, x, xrec, 1, self.global_step,
                                            last_layer=self.get_last_layer(), split="val")
        '''
        aeloss, log_dict_ae = self.loss(qloss, dis_x, xrec, 0, self.global_step,
                                        last_layer=self.get_last_layer(), split="val")

        discloss, log_dict_disc = self.loss(qloss, dis_x, xrec, 1, self.global_step,
                                            last_layer=self.get_last_layer(), split="val")
        rec_loss = log_dict_ae["val/rec_loss"]
        self.log("val/rec_loss", rec_loss,
                 prog_bar=True, logger=True, on_step=False, on_epoch=True, sync_dist=True)
        self.log("val/aeloss", aeloss,
                 prog_bar=True, logger=True, on_step=False, on_epoch=True, sync_dist=True)
        self.log_dict(log_dict_ae)
        self.log_dict(log_dict_disc)
        return self.log_dict

    def log_images(self, batch, **kwargs):
        log = dict()
        x = self.get_input(batch, self.image_key)
        x = x.to(self.device)
        # dis_x is discriminator image.
        dis_x = self.get_input(batch, "dis_image")
        dis_x = dis_x.to(self.device)
        # encode
        h = self.encoder(x)
        h = self.quant_conv(h)
        quant, _, _ = self.quantize(h)
        # decode
        x_rec = self.decode(quant)
        log["inputs"] = dis_x
        log["reconstructions"] = x_rec
        return log


class EMAVQ(VQModel):
    def __init__(self,
                 ddconfig,
                 lossconfig,
                 n_embed,
                 embed_dim,
                 ckpt_path=None,
                 ignore_keys=[],
                 image_key="image",
                 colorize_nlabels=None,
                 monitor=None,
                 remap=None,
                 sane_index_shape=False,  # tell vector quantizer to return indices as bhw
                 ):
        super().__init__(ddconfig,
                         lossconfig,
                         n_embed,
                         embed_dim,
                         ckpt_path=None,
                         ignore_keys=ignore_keys,
                         image_key=image_key,
                         colorize_nlabels=colorize_nlabels,
                         monitor=monitor,
                         )
        self.quantize = EMAVectorQuantizer(n_embed=n_embed,
                                           embedding_dim=embed_dim,
                                           beta=0.25,
                                           remap=remap)
    def configure_optimizers(self):
        lr = self.learning_rate
        #Remove self.quantize from parameter list since it is updated via EMA
        opt_ae = torch.optim.Adam(list(self.encoder.parameters())+
                                  list(self.decoder.parameters())+
                                  list(self.quant_conv.parameters())+
                                  list(self.post_quant_conv.parameters()),
                                  lr=lr, betas=(0.5, 0.9))
        opt_disc = torch.optim.Adam(self.loss.Discriminator.parameters(),
                                    lr=lr, betas=(0.5, 0.9))
        return [opt_ae, opt_disc], []
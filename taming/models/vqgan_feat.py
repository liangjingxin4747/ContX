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



def nonlinearity(x):
    # swish
    return x*torch.sigmoid(x)

def Normalize(in_channels):
    return torch.nn.GroupNorm(num_groups=32, num_channels=in_channels, eps=1e-6, affine=True)


class VQModel(pl.LightningModule):
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
                 ):
        super().__init__()
        self.in_channels = ddconfig['in_channels']
        self.out_channels = ddconfig['out_ch']
        self.image_key = image_key
        self.encoder = Encoder(**ddconfig)
        self.decoder = Decoder(**ddconfig)
        self.loss = instantiate_from_config(lossconfig)
        self.quantize = VectorQuantizer(n_embed, embed_dim, beta=0.25,
                                        remap=remap, sane_index_shape=sane_index_shape)
        self.quant_conv = torch.nn.Conv2d(ddconfig["z_channels"], embed_dim, 1)
        self.post_quant_conv = torch.nn.Conv2d(embed_dim, ddconfig["z_channels"], 1)
        if ckpt_path is not None:
            self.init_from_ckpt(ckpt_path, ignore_keys=ignore_keys)
        if colorize_nlabels is not None:
            assert type(colorize_nlabels) == int
            self.register_buffer("colorize", torch.randn(3, colorize_nlabels, 1, 1))
        if monitor is not None:
            self.monitor = monitor
        self.label_colours = np.random.randint(255, size=(self.in_channels, 3))

        self.recon_net = instantiate_from_config(recon_net)   # General_Recon_Net
        self.recon_net.load_state_dict(torch.load(os.path.join(recon_net.params.ckpt_dir, 'recon_net.pth')))
        self.recon_net.vector_dict = np.load(os.path.join(recon_net.params.ckpt_dir, 'codebook.npy'), allow_pickle=True)[()]
        self.freeze_model(self.recon_net)
        self.SPRef = recon_net.params.memory_refine
        self.SPk = recon_net.params.memory_refine_k

        self.fuse_layer = nn.Sequential(
            # Normalize(self.out_channels),
            torch.nn.Conv2d(ddconfig["z_channels"]+8,
                            ddconfig["z_channels"],
                            kernel_size=3,
                            stride=1,
                            padding=1)
        )
        self.refine_decoder = Decoder(**ddconfig)
        
        with open("/home/liangxin01/code/dyh/model/taming-transformer-origin/concat_cat.txt", "r") as f:
            classes = f.read().splitlines()
        self.cls_gray_dict = dict()
        for i, cls in enumerate(classes):
            self.cls_gray_dict[i] = int(cls)

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
        quant = self.post_quant_conv(quant)  # Tensor(bs,256,16,16)
        dec = self.decoder(quant)  # Tensor(bs,87,256,256)
        return dec

    def decode_code(self, code_b):
        quant_b = self.quantize.embed_code(code_b)
        dec = self.decode(quant_b)
        return dec

    def freeze_model(self, model):
        for param in model.parameters():
            param.requires_grad = False
    
    def classes_choose(self, logits, require_grad=False):   # Tensor(bs,87,h,w)
        bs, c, h, w = logits.shape
        pred_classes = torch.argmax(logits, dim=1)   # Tensor(bs,1,h,w)
        class_masks = F.one_hot(pred_classes, num_classes=c).permute(0,3,1,2)   # Tensor(bs,c,h,w)
        if require_grad:
            class_masks = logits + (class_masks - logits).detach()
        # class_masks = class_masks   # Tensor(bs,c,h,w)
        class_masks = class_masks.unsqueeze(2).repeat(1, 1, self.SPk, 1, 1)   # Tensot(bs,c,k=16,h,w)
        return class_masks

    def get_shape_prior(self, pred_context_logits):   # Tensor(bs,87,h,w)
        bs, c, h, w = pred_context_logits.shape
        
        reconstruction = pred_context_logits.view(bs, c, h*w)   # Tensor(bs,87,h*w)
        reconstruction = torch.sum(reconstruction, dim=-1)   # Tensor(bs,87)
        classes_ori = list(torch.argmax(reconstruction, dim=-1))   # Tensor(bs)
        max_pred = pred_context_logits[list(np.arange(bs)), classes_ori, :, :].view(bs,1,h,w)   # Tensor(bs,1,h,w)
        
        nn_latent_vectors = self.recon_net.encode(max_pred)   # Tensor(bs,8,h/8,w/8)
        classes = torch.tensor([self.cls_gray_dict[cls.item()] for cls in classes_ori]).to(self.device)   # Tensor(bs)
        # classes = torch.tensor([156 for i in range(len(classes))]).to(self.device)
        z_q, shape_prior = self.recon_net.nearest_decode(nn_latent_vectors, classes, k=self.SPk)   # Tensor(bs,SPk,h,w)

        return z_q  # Tensor(bs,8,h/8,w/8)

    def forward(self, input):  # Tensor(bs,87,256,256)
        # vqgan
        quant, diff, _ = self.encode(input)  # Tensor(bs,256,h/16,w/16)
        pred_context_logits = self.decode(quant)  # Tensor(bs,87,h,w)

        # shape_prior recon_net
        codebook_prior = self.get_shape_prior(pred_context_logits)   # Tensor(bs,8,h/8,w/8)
        codebook_prior = F.avg_pool2d(codebook_prior, kernel_size=2, stride=2)   # Tensor(bs,8,h/16,w/16)
       
        quant = self.fuse_layer(cat([quant, codebook_prior], dim=1))   # Tensor(bs,256+8,h/16,w/16)->Tensor(bs,256,h/16,w/16)
        pred_context_logits_ = self.refine_decoder(quant)  # Tensor(bs,87,h,w)

        return [pred_context_logits, pred_context_logits_], diff

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
                                        cond=x, last_layer=self.get_last_layer(), split="val")

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
        opt_ae = torch.optim.Adam(list(self.encoder.parameters()) +
                                  list(self.decoder.parameters()) +
                                  list(self.quantize.parameters()) +
                                  list(self.quant_conv.parameters()) +
                                  list(self.post_quant_conv.parameters()) +
                                  list(self.fuse_layer.parameters()) + 
                                  list(self.refine_decoder.parameters()),
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

    def mask_recon_inference(self, dis_x):
        vector_dict = {}

        target = torch.argmax(dis_x, dim=1)   # Tensor(bs,h,w)
        classes = []
        masks = []
        for image in target:   # Tensor(h,w)
            classes_per_image = image.unique()
            classes.append(classes_per_image)

            for label in classes_per_image:
                mask_per_image = torch.where(image == label, 1, 0)
                masks.append(mask_per_image.unsqueeze(0).float())

        masks = cat(masks, dim=0).unsqueeze(1)   # Tensor(B,1,h,w)
        classes = cat(classes, dim=0)   # Tensor(B,)

        assert masks.size(0) == classes.size(0)

        if self.recon_net.memory_aug:
            masks_aug = masks
            classes_aug = classes
            for degree in [0, 90, 180, 270]:
                for i in range(2):
                    if degree == 0 and i != 0:
                        continue
                    angle = - degree * math.pi / 180
                    theta = torch.tensor([
                        [math.cos(angle), math.sin(-angle), 0],
                        [math.sin(angle), math.cos(angle), 0]
                    ], dtype=torch.float)
                    theta = cat([theta.unsqueeze(0)] * masks.size(0), dim=0)
                    grid = F.affine_grid(theta, masks.size(), align_corners=True).to(masks.device)
                    output = F.grid_sample(masks, grid, align_corners=True)
                    if i == 0:
                        output = output.flip(2)

                    masks_aug = cat([masks_aug, output], dim=0)
                    classes_aug = cat([classes_aug, classes], dim=0)
            masks = masks_aug   # Tensor(N_aug,1,h,w)
            classes = classes_aug   # Tensor(N_aug,)

        latent_vectors = self.recon_net.encode(masks)   # Tensor(N_aug,8,h/8,w/8)
        for i in range(len(classes.unique())):
            index = (classes == classes.unique()[i].item()).nonzero()
            vector_dict[classes.unique()[i].item()] = latent_vectors[index].view(len(index), -1)

        self.recon_net.recording_vectors(vector_dict)

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
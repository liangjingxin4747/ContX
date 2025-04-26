import math

import torch
from sklearn.cluster import KMeans
from torch import nn
from torch.nn import functional as F

from taming.modules.detectron2 import Conv2d, get_norm, cat
from einops import rearrange

# def mask_recon_inference(pred_instances, targets, recon_net, iou_ths=0.9):
#     assert len(pred_instances) == 1
#     vector_dict = {}
#
#     pred_instance = pred_instances[0]
#     target = targets[0]
#     masks = []
#     classes = []
#     mask_side_len = pred_instances[0].pred_amodal_masks.size(2)
#
#     res = detector_postprocess(pred_instance, pred_instance.image_size[0], pred_instance.image_size[1])
#     gt_masks_orig_size = pred_instance.gt_masks_inference.tensor
#     pred_masks_orig_size = res.pred_amodal_masks.cuda()
#
#     iou = torch.sum((gt_masks_orig_size * pred_masks_orig_size) > 0, dim=(1, 2)).float() /\
#           torch.sum((gt_masks_orig_size + pred_masks_orig_size) > 0, dim=(1, 2)).float()
#     # print(iou)
#     filter_inds = (iou > iou_ths).nonzero()
#     if pred_instance.has("pred_amodal_masks"):
#         masks.append(pred_instance.pred_amodal2_masks[filter_inds[:, 0]])  # (1, Hmask, Wmask)
#     else:
#         masks.append(pred_instance.pred_amodal_masks[filter_inds[:, 0]])
#     classes.append(pred_instance.gt_classes_inference[filter_inds[:, 0]])
#
#     if filter_inds.size(0) != 0:
#         masks.append(pred_instance.gt_masks_inference[filter_inds[:, 0]].crop_and_resize(
#             pred_instance.pred_boxes.tensor[filter_inds[:, 0]], mask_side_len
#         ).to(device=pred_masks_orig_size.device).unsqueeze(1).float())
#         classes.append(pred_instance.gt_classes_inference[filter_inds[:, 0]])
#
#     masks.append(target.gt_masks.crop_and_resize(
#         target.gt_boxes.tensor, mask_side_len
#     ).to(device=pred_masks_orig_size.device).unsqueeze(1).float())
#     classes.append(target.gt_classes)
#
#     masks = cat(masks, dim=0)
#     classes = cat(classes, dim=0)
#
#     # else:
#     #     mask_side_len = pred_instances[0].pred_masks.size(2)
#     #
#     #     res = detector_postprocess(pred_instance, pred_instance.image_size[0], pred_instance.image_size[1])
#     #     gt_masks_orig_size = pred_instance.gt_masks_inference.tensor
#     #     pred_masks_orig_size = res.pred_masks.cuda()
#     #
#     #     iou = torch.sum((gt_masks_orig_size * pred_masks_orig_size) > 0, dim=(1, 2)).float() /\
#     #           torch.sum((gt_masks_orig_size + pred_masks_orig_size) > 0, dim=(1, 2)).float()
#     #     filter_inds = (iou > iou_ths).nonzero()
#     #
#     #     pred_amodal_masks_to_recon = pred_instance.pred_masks[filter_inds[:, 0]]  # (1, Hmask, Wmask)
#     #     classes_to_pred_recon = pred_instance.gt_classes_inference[filter_inds[:, 0]]
#     #
#     #     gt_masks_to_recon = target.gt_masks.crop_and_resize(
#     #         target.gt_boxes.tensor, mask_side_len
#     #     ).to(device=pred_masks_orig_size.device)
#     #     classes_to_gt_recon = target.gt_classes
#     #
#     #     masks = cat([pred_amodal_masks_to_recon, gt_masks_to_recon.unsqueeze(1).float()], dim=0)
#     #     classes = cat([classes_to_pred_recon, classes_to_gt_recon], dim=0)
#
#     if recon_net.memory_aug:
#         masks_aug = masks
#         classes_aug = classes
#         for degree in [0, 90, 180, 270]:
#             for i in range(2):
#                 if degree == 0 and i != 0:
#                     continue
#                 angle = - degree * math.pi / 180
#                 theta = torch.tensor([
#                     [math.cos(angle), math.sin(-angle), 0],
#                     [math.sin(angle), math.cos(angle), 0]
#                 ], dtype=torch.float)
#                 theta = cat([theta.unsqueeze(0)] * masks.size(0), dim=0)
#                 grid = F.affine_grid(theta, masks.size(), align_corners=True).to(masks.device)
#                 output = F.grid_sample(masks, grid, align_corners=True)
#                 if i == 0:
#                     output = output.flip(2)
#
#                 masks_aug = cat([masks_aug, output], dim=0)
#                 classes_aug = cat([classes_aug, classes], dim=0)
#         masks = masks_aug
#         classes = classes_aug
#
#     recon_masks_logits, latent_vectors = recon_net((masks > 0.5).float())
#     # recon_masks = (recon_masks_logits > 0.5).float()
#     for i in range(len(classes.unique())):
#         index = (classes == classes.unique()[i].item()).nonzero()
#         vector_dict[classes.unique()[i].item()] = latent_vectors[index].view(len(index), -1)
#
#
#         # if len(pred_instances) < 10:
#         #     if pred_masks_to_recon.size(0) != 0:
#         #         vis.images((pred_masks_to_recon > 0.5).float(), win_name="inference_pred_masks_in_recon_{}".format(len(pred_instances)))
#         #         vis.images(recon_masks_logits[:pred_masks_to_recon.size(0)], win_name="inference_pred_masks_out_recon_{}".format(len(pred_instances)))
#         #     vis.images((gt_masks_to_recon.unsqueeze(1) > 0.5).float(), win_name="inference_gt_masks_in_recon_{}".format(len(pred_instances)))
#         #     vis.images(recon_masks_logits[pred_masks_to_recon.size(0):], win_name="inference_gt_masks_out_recon_{}".format(len(pred_instances)))
#
#     recon_net.recording_vectors(vector_dict)


class General_Recon_Net(nn.Module):
    def __init__(self, num_conv, conv_dim, num_classes, kmeans, norm, rescoring, memo_aug, **args):
        super(General_Recon_Net, self).__init__()
        self.name = "AE"
        self.num_conv = num_conv
        self.conv_dims = conv_dim
        self.num_classes = num_classes
        self.num_cluster = kmeans
        self.norm = norm
        self.rescoring = rescoring
        self.memory_aug = memo_aug
        input_channels = 1
        self.vector_dict = {}
        self.encoder = []
        self.decoder = []

        for k in range(self.num_conv):
            conv = Conv2d(
                input_channels if k == 0 else self.conv_dims,
                self.conv_dims,
                kernel_size=3,
                stride=2,
                padding=1,
                bias=not self.norm,
                norm=nn.BatchNorm2d(self.conv_dims),
                activation=F.relu,
            )
            self.add_module("mask_fcn_enc{}".format(k + 1), conv)

            self.encoder.append(conv)

            deconv = nn.Sequential(nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True),
                                   Conv2d(self.conv_dims,
                                          self.conv_dims,
                                          kernel_size=3,
                                          stride=1,
                                        #   padding=0 if k == self.num_conv - 2 and self.num_conv > 2 else 1,
                                          padding=1,
                                          bias=not self.norm,
                                          norm=nn.BatchNorm2d(self.conv_dims)
                                          ),
                                   # get_norm(self.norm, self.conv_dims, num_groups=2),
                                   nn.ReLU())
            self.add_module("mask_fcn_dec{}".format(k + 1), deconv)
            self.decoder.append(deconv)
        self.outconv = nn.Sequential(nn.Conv2d(self.conv_dims,
                                               input_channels,
                                               kernel_size=3,
                                               stride=1,
                                               padding=1),
                                     nn.Sigmoid())
        # self.d = {1: 14, 2: 7, 3: 4, 4: 2}
        # self.fc = nn.Linear(self.conv_dims, self.conv_dims * self.d[self.num_conv] * self.d[self.num_conv], bias=False)

        # for layer in self.encoder + self.decoder:
        #     weight_init.c2_msra_fill(layer)
        # d = {3: 4, 4: 2}
        # self.codebook = nn.Embedding(K * num_classes, d[self.num_conv] ** 2)

    def forward(self, x):
        for layer in self.encoder:
            x = layer(x)
        latent_vector = x

        for layer in self.decoder:
            x = layer(x)
        x = self.outconv(x)
        return x, latent_vector

    def encode(self, x):
        for layer in self.encoder:
            x = layer(x)
        return x

    def decode(self, vectors):
        for layer in self.decoder:
            vectors = layer(vectors)
        x = self.outconv(vectors)
        return x

    def recording_vectors(self, vector_inference):
        for key, item in vector_inference.items():
            if self.vector_dict.get(key, None) is not None:
                self.vector_dict[key] = torch.cat([self.vector_dict[key], item], dim=0)
            else:
                self.vector_dict[key] = item

    def nearest_decode(self, vectors, pred_classes, k=1):
        z_e = vectors
        vectors = vectors.view(vectors.shape[0], -1)   # tensor(B,size*size/8)
        
        side_len = math.sqrt(vectors.size(1) / self.conv_dims)
        assert side_len % 1 == 0
        memo_latent_vectors = torch.zeros((vectors.size(0), k, vectors.size(1))).to(vectors.device)

        classes_lst = pred_classes.unique()
        for i in range(len(classes_lst)):
            with torch.no_grad():
                index = (pred_classes == pred_classes.unique()[i].item()).nonzero()   # 非0索引
                vectors_per_classes = vectors[index[:, 0]]
                if pred_classes.unique()[i].item() in self.vector_dict:
                    codebook = self.vector_dict[pred_classes.unique()[i].item()]

                    codebook_sqr = torch.sum(codebook ** 2, dim=1)
                    inputs_sqr = torch.sum(vectors_per_classes ** 2, dim=1, keepdim=True)

                    # Compute the distances to the codebook
                    distances = torch.addmm(codebook_sqr + inputs_sqr,
                                            vectors_per_classes, codebook.t(), alpha=-2.0, beta=1.0)

                    # _, indices_flatten = torch.min(distances, dim=1)
                    # indices = indices_flatten.view(*inputs_size[:-1])
                    indices = torch.topk(- distances, k)[1]
                    nn_vectors = codebook[indices]

                    # nn_vectors = torch.index_select(codebook, dim=0, index=indices)
                    memo_latent_vectors[index[:, 0]] = nn_vectors
                else:
                    memo_latent_vectors[index[:, 0]] = vectors_per_classes.unsqueeze(1)

        # memo_latent_vectors = (memo_latent_vectors + vectors) / 2
        vectors = memo_latent_vectors.view(pred_classes.size(0) * k, self.conv_dims, int(side_len), int(side_len))
        z_q = z_e + (vectors - z_e).detach()
        
        x = z_q
        for layer in self.decoder:
            x = layer(x)
        x = self.outconv(x)
        x = x.view(pred_classes.size(0), k, x.size(2), x.size(3))
        return z_q, x

    def cluster(self):
        for i in range(self.num_classes):
            # print("Start cluster No.{} class".format(i + 1))
            if i not in self.vector_dict:
                continue
            if self.vector_dict[i].size(0) > self.num_cluster:
                codes = self.vector_dict[i]
                kmeans = KMeans(n_clusters=self.num_cluster)
                kmeans.fit(codes.cpu())
                self.vector_dict[i] = torch.FloatTensor(kmeans.cluster_centers_).cuda()
    
    def vq(self, z_e, memorybank):   # (num_cls,16,8192)
        # reshape z -> (batch, height, width, channel) and flatten
        bs, c, h, w = z_e.shape   # Tensor(num_cls,8,32,32)
        z_q = torch.zeros_like(z_e).to(z_e.device)
        for i, (z, mb) in enumerate(zip(z_e, memorybank)):   # (16,8192)
            z_flattened = z.view(1, -1)   # (1,8192)
            mb = mb.to(z_flattened.device)
            # distances from z to embeddings e_j (z - e)^2 = z^2 + e^2 - 2 e * z   Tensor(1,16)
            d = torch.sum(z_flattened ** 2, dim=1, keepdim=True) + \
                torch.sum(mb**2, dim=1) - 2 * \
                torch.einsum('bd,dn->bn', z_flattened, rearrange(mb, 'n d -> d n'))

            min_encoding_indices = torch.argmin(d, dim=1)   # Tensor(1,)
            z_quant = mb[min_encoding_indices[0]].view(z.shape)   # Tensor(1,)->Tensor(1,8192)->Tensor(8,32,32)
            z_q[i] = z_quant

        return z_q   # Tensor(num_cls,8,32,32)
    

class AEModel(nn.Module):
    def __init__(self, in_channels, out_channels, num_hiddens, num_residual_layers, num_residual_hiddens, **args):
        super().__init__()
        self.encoder = Encoder(in_channels=in_channels, 
                               num_hiddens=num_hiddens,
                               num_residual_layers=num_residual_layers,
                               num_residual_hiddens=num_residual_hiddens)
        self.decoder = Decoder(in_channels=num_hiddens,
                               out_channels=out_channels,
                               num_hiddens=num_hiddens,
                               num_residual_layers=num_residual_layers,
                               num_residual_hiddens=num_residual_hiddens)
    
    def forward(self, x):
        z_e = self.encoder(x)   # Tensor(B,num_hiddens,h',w')
        x_recon = self.decoder(z_e)   # Tensor(B,1,H,W)
        x_recon = F.sigmoid(x_recon)
        return z_e, x_recon
        
    def encode(self, x):
        return self.encoder(x)
    
    def decode(self, z_e):
        return F.sigmoid(self.decoder(z_e))
    
    def vq(self, z_e, memorybank):   # (num_cls,16,8192)
        # reshape z -> (batch, height, width, channel) and flatten
        bs, c, h, w = z_e.shape   # Tensor(num_cls,8,32,32)
        z_q = torch.zeros_like(z_e).to(z_e.device)
        for i, (z, mb) in enumerate(zip(z_e, memorybank)):   # (16,8192)
            z_flattened = z.view(1, -1)   # (1,8192)
            mb = mb.to(z_flattened.device)
            # distances from z to embeddings e_j (z - e)^2 = z^2 + e^2 - 2 e * z   Tensor(1,16)
            d = torch.sum(z_flattened ** 2, dim=1, keepdim=True) + \
                torch.sum(mb**2, dim=1) - 2 * \
                torch.einsum('bd,dn->bn', z_flattened, rearrange(mb, 'n d -> d n'))

            min_encoding_indices = torch.argmin(d, dim=1)   # Tensor(1,)
            z_quant = mb[min_encoding_indices[0]].view(z.shape)   # Tensor(1,)->Tensor(1,8192)->Tensor(8,32,32)
            z_q[i] = z_quant

        return z_q   # Tensor(num_cls,8,32,32)
    


class Encoder(nn.Module):
    def __init__(self, in_channels, num_hiddens, num_residual_layers, num_residual_hiddens):
        super(Encoder, self).__init__()

        self._conv_1 = nn.Conv2d(in_channels=in_channels,
                                 out_channels=num_hiddens//2,
                                 kernel_size=4,
                                 stride=2, padding=1)
        self._conv_2 = nn.Conv2d(in_channels=num_hiddens//2,
                                 out_channels=num_hiddens,
                                 kernel_size=4,
                                 stride=2, padding=1)
        self._conv_3 = nn.Conv2d(in_channels=num_hiddens,
                                 out_channels=num_hiddens,
                                 kernel_size=3,
                                 stride=1, padding=1)
        self._residual_stack = ResidualStack(in_channels=num_hiddens,
                                             num_hiddens=num_hiddens,
                                             num_residual_layers=num_residual_layers,
                                             num_residual_hiddens=num_residual_hiddens)

    def forward(self, inputs):
        x = self._conv_1(inputs)
        x = F.relu(x)
        
        x = self._conv_2(x)
        x = F.relu(x)
        
        x = self._conv_3(x)
        return self._residual_stack(x)


class Decoder(nn.Module):
    def __init__(self, in_channels, out_channels, num_hiddens, num_residual_layers, num_residual_hiddens):
        super(Decoder, self).__init__()
        
        self._conv_1 = nn.Conv2d(in_channels=in_channels,
                                 out_channels=num_hiddens,
                                 kernel_size=3, 
                                 stride=1, padding=1)
        
        self._residual_stack = ResidualStack(in_channels=num_hiddens,
                                             num_hiddens=num_hiddens,
                                             num_residual_layers=num_residual_layers,
                                             num_residual_hiddens=num_residual_hiddens)
        
        self._conv_trans_1 = nn.ConvTranspose2d(in_channels=num_hiddens, 
                                                out_channels=num_hiddens//2,
                                                kernel_size=4, 
                                                stride=2, padding=1)
        
        self._conv_trans_2 = nn.ConvTranspose2d(in_channels=num_hiddens//2, 
                                                out_channels=out_channels,
                                                kernel_size=4, 
                                                stride=2, padding=1)

    def forward(self, inputs):
        x = self._conv_1(inputs)
        
        x = self._residual_stack(x)
        
        x = self._conv_trans_1(x)
        x = F.relu(x)
        
        return self._conv_trans_2(x)


class Residual(nn.Module):
    def __init__(self, in_channels, num_hiddens, num_residual_hiddens):
        super(Residual, self).__init__()
        self._block = nn.Sequential(
            nn.ReLU(True),
            nn.Conv2d(in_channels=in_channels,
                      out_channels=num_residual_hiddens,
                      kernel_size=3, stride=1, padding=1, bias=False),
            nn.ReLU(True),
            nn.Conv2d(in_channels=num_residual_hiddens,
                      out_channels=num_hiddens,
                      kernel_size=1, stride=1, bias=False)
        )
    
    def forward(self, x):
        return x + self._block(x)
    

class ResidualStack(nn.Module):
    def __init__(self, in_channels, num_hiddens, num_residual_layers, num_residual_hiddens):
        super(ResidualStack, self).__init__()
        self._num_residual_layers = num_residual_layers
        self._layers = nn.ModuleList([Residual(in_channels, num_hiddens, num_residual_hiddens)
                             for _ in range(self._num_residual_layers)])

    def forward(self, x):
        for i in range(self._num_residual_layers):
            x = self._layers[i](x)
        return F.relu(x)
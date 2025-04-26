import collections
import myclip
import torch
import torch.nn as nn
from torchvision import transforms
from torchvision.transforms.functional import InterpolationMode


class Loss(nn.Module):
    def __init__(self, layout_clip, shape_clip, device, train_with_clip, clip_weight=0, start_clip_iter=0,
                 clip_conv_loss=1, clip_text_guide=0):
        super().__init__()

        self.device = device
        self.train_with_clip = train_with_clip
        self.clip_weight = clip_weight
        self.start_clip_iter = start_clip_iter

        self.clip_conv_loss = clip_conv_loss
        self.clip_text_guide = clip_text_guide
        
        self.losses_to_apply = self.get_losses_to_apply()
        self.loss_mapper = {
            # "clip": CLIPLoss(**layout_clip),
                            "clip_conv_loss": CLIPConvLoss(**shape_clip)}
        
    def get_losses_to_apply(self):
        losses_to_apply = []
        if self.train_with_clip and self.start_clip_iter == 0:
            losses_to_apply.append("clip")
        if self.clip_conv_loss:
            losses_to_apply.append("clip_conv_loss")
        if self.clip_text_guide:
            losses_to_apply.append("clip_text")
        return losses_to_apply

    def update_losses_to_apply(self, iter):
        if "clip" not in self.losses_to_apply:
            if self.train_with_clip:
                if iter > self.start_clip_iter:
                    self.losses_to_apply.append("clip")
                    
    def forward(self, inputs, targets, split='train'):
        # self.update_losses_to_apply(iter)
        device = inputs.device
        losses_dict = dict.fromkeys(self.losses_to_apply, torch.tensor([0.0]).to(device))   # {"clip_conv_loss": Tensor([0.])}
        loss_weight = dict.fromkeys(self.losses_to_apply, 1.0)   # {"clip_conv_loss": 1.0}
        loss_weight["clip"] = self.clip_weight                   # {"clip": clip_weight}
        loss_weight["clip_text"] = self.clip_text_guide          # {"clip_text": clip_text_guide}
        
        for loss_name in self.losses_to_apply:
            if loss_name in ["clip_conv_loss"]:   # !
                conv_loss = self.loss_mapper[loss_name](   # self.loss_mapper:{"clip": CLIPLoss, "clip_conv_loss": CLIPConvLoss}
                    inputs, targets, split)
                for layer in conv_loss.keys():
                    losses_dict[layer] = conv_loss[layer]   # low-level
            else:
                losses_dict[loss_name] = self.loss_mapper[loss_name](
                    inputs, targets, split).mean()   # high-level
        
        for key in self.losses_to_apply:
            losses_dict[key] = losses_dict[key] * loss_weight[key]
        return sum(list(losses_dict.values()))
    

class CLIPLoss(nn.Module):
    def __init__(self, device, model_path=None, num_aug_clip=4):
        super().__init__()
        self.device = device
        self.model, preprocess = myclip.load('ViT-B/32', device="cpu", jit=False)
        if model_path is not None:
            self.model.load_state_dict(torch.load(model_path, map_location="cpu"))
        self.model.to(device)
        self.model.eval()
        self.preprocess = transforms.Compose([preprocess.transforms[-1]])   # Normalization(clip)
        
        self.NUM_AUGS = num_aug_clip
        augemntations = []
        augemntations.append(transforms.RandomResizedCrop(224, scale=(0.8, 0.8), ratio=(1.0, 1.0)))
        augemntations.append(transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)))
        self.augment_trans = transforms.Compose(augemntations)
        
        self.calc_target = True
        
    def forward(self, inputs, targets, split='train'):
        if self.calc_target:
            targets_ = self.preprocess(targets).to(self.device)
            self.targets_features = self.model.encode_image(targets_).detach()
        
        if split == 'eval':
            # for regular clip distance, no augmentations
            with torch.no_grad():
                inputs = self.preprocess(inputs).to(self.device)
                inputs_features = self.model.encode_image(inputs)
                return 1. - torch.cosine_similarity(inputs_features, self.targets_features)
        
        # split == 'train', augment inputs
        loss_clip = 0
        inputs_augs = []
        for i in range(self.NUM_AUGS):
            augmented_pair = self.augment_trans(torch.cat([inputs, targets]))
            inputs_augs.append(augmented_pair[0].unsqueeze(0))
        
        inputs_batch = torch.cat(inputs_augs)
        inputs_features = self.model.encode_image(inputs_batch)
        
        for i in range(self.NUM_AUGS):
            loss_clip += (1. - torch.cosine_similarity(inputs_features[i:i+1], self.targets_features, dim=1))
        return loss_clip
        

class CLIPConvLoss(nn.Module):
    def __init__(self, device, ckpt_path=None, clip_model_name='ViT-B/32', clip_conv_loss_type='L2', clip_fc_loss_type='Cos', clip_fc_loss_weight=0.1, 
                 clip_conv_layer_weights=[0,0,1,1,0], num_aug_clip=4):
        super().__init__()
        self.clip_model_name = clip_model_name
        assert self.clip_model_name in ["ViT-B/32", "ViT-B/16"]
        
        self.clip_conv_loss_type = clip_conv_loss_type   # L2
        self.clip_fc_loss_type = 'Cos'
        assert self.clip_conv_loss_type in ["L2", "Cos", "L1"]
        assert self.clip_fc_loss_type in ["L2", "Cos", "L1"]
        
        self.distance_metrics = {'L2': l2_layers,
                                 'L1': l1_layers,
                                 'Cos': cos_layers}
        
        self.model, preprocess = myclip.load(self.clip_model_name, device, jit=False)
        if ckpt_path is not None:
            self.model.load_state_dict(torch.load(ckpt_path, map_location=device))
        self.visual_encoder = CLIPVisualEncoder(self.model)
        
        self.img_size = preprocess.transforms[1].size   # CenterCrop
        self.target_transform = transforms.Compose([
            transforms.ToTensor(),
        ])  # clip normalisation
        # self.normalize_transform = transforms.Compose([
        #     preprocess.transforms[0],  # Resize
        #     preprocess.transforms[1],  # CenterCrop
        #     preprocess.transforms[-1],  # Normalize
        # ])
        self.normalize_transform = transforms.Compose([
            transforms.Resize([224], interpolation=InterpolationMode.NEAREST),
            transforms.CenterCrop(224),
            transforms.ConvertImageDtype(torch.float),
            transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
        ])
        
        self.model.eval()
        self.device = device
        self.num_augs = num_aug_clip
        
        augemntations = []
        augemntations.append(transforms.RandomResizedCrop(224, scale=(0.8, 0.8), ratio=(1.0, 1.0)))
        augemntations.append(transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)))
        self.augment_trans = transforms.Compose(augemntations)
        
        self.clip_fc_layer_dims = None
        self.clip_conv_layer_dims = None
        self.clip_fc_loss_weight = clip_fc_loss_weight
        self.clip_conv_layer_weights = clip_conv_layer_weights
    
    def forward(self, inputs, targets, split='train'):
        inputs = inputs.repeat(1,3,1,1).to(torch.uint8).to(self.device)   # Tensor(bs,3,h,w) L->RGB
        targets = targets.repeat(1,3,1,1).to(torch.uint8).to(self.device)   # Tensor(bs,3,h,w) L->RGB
        inputs_augs = [self.normalize_transform(inputs)]   # [Tensor(bs,3,224,224)]
        targets_augs = [self.normalize_transform(targets)]   # [Tensor(bs,3,224,224)]
        
        conv_loss_dict = {}
        if split == 'train':   # augment inputs & targets
            for i in range(self.num_augs):
                augmented_pair = self.augment_trans(torch.cat([inputs, targets]))
                inputs_augs.append(augmented_pair[0].unsqueeze(0))
                targets_augs.append(augmented_pair[1].unsqueeze(0))
        
        inputs_batch = torch.cat(inputs_augs, dim=0).to(self.device)   # Tensor(bs*(1+num_augs),3,224,224)
        targets_batch = torch.cat(targets_augs, dim=0).to(self.device)   # Tensor(bs*(1+num_augs),3,224,224)
        
        inputs_fc_features, inputs_conv_features = self.visual_encoder(inputs_batch)   # Tensor(bs,512)  list(12) of Tensor(bs,50,768)
        targets_fc_features, targets_conv_features = self.visual_encoder(targets_batch)
        
        conv_loss = self.distance_metrics[self.clip_conv_loss_type](   # L2 low-level_loss
            inputs_conv_features, targets_conv_features)   # list(12)
        
        for layer, w in enumerate(self.clip_conv_layer_weights):   # {0:0,1:0,2:1,3:1,4:0} low-level
            if w:
                conv_loss_dict[f"clip_conv_loss_layer{layer}"] = conv_loss[layer] * w
        
        if self.clip_fc_loss_weight:   # 0.1
            fc_loss = (1. - torch.cosine_similarity(inputs_fc_features, targets_fc_features, dim=1)).mean()
            conv_loss_dict['fc'] = fc_loss * self.clip_fc_loss_weight
            
        return conv_loss_dict
    

class CLIPVisualEncoder(nn.Module):
    def __init__(self, clip_model):
        super().__init__()
        self.model = clip_model
        self.featuremaps = None
        
        for i in range(12):   # 12 resblocks in ViT
            self.model.visual.transformer.resblocks[i].register_forward_hook(
                self.make_hook(i)
            )
            
    def make_hook(self, name):
        def hook(module, input, output):
            if len(output.shape) == 3:
                self.featuremaps[name] = output.permute(1,0,2)   # LND->NLD(bs,smth,768)
            else:
                self.featuremaps[name] = output
        return hook
    
    def forward(self, x):
        self.featuremaps = collections.OrderedDict()
        fc_features = self.model.encode_image(x).float()
        featuremaps = [self.featuremaps[k] for k in range(12)]
        
        return fc_features, featuremaps


def l2_layers(inputs_conv_features, targets_conv_features):
    return [torch.square(input_conv - target_conv).mean() 
            for input_conv, target_conv in zip(inputs_conv_features, targets_conv_features)]
    
def l1_layers(inputs_conv_features, targets_conv_features):
    return [torch.abs(input_conv - target_conv).mean() 
            for input_conv, target_conv in zip(inputs_conv_features, targets_conv_features)]
    
def cos_layers(inputs_conv_features, targets_conv_features):
    return [(1. - torch.cosine_similarity(input_conv, target_conv)).mean()
            for input_conv, target_conv in zip(inputs_conv_features, targets_conv_features)]
    


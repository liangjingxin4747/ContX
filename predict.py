import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "3"

import random
import numpy as np
import albumentations
from tqdm import tqdm
from PIL import Image, ImageDraw, ImageFont

import glob
import yaml
import torch
from omegaconf import OmegaConf
from taming.models.vqgan_recon import VQModelwithSPRef
from taming.models.vqgan import VQModel
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

from skimage import segmentation
from main import instantiate_from_config


# disable grad to save memory
torch.set_grad_enabled(False)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

font = ImageFont.truetype(
    fm.findfont(fm.FontProperties()),
    size=11)


SEED = 23
def setup_seed(seed):
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
setup_seed(SEED)


'''
    得到 类别标签 和 类别名 的对应关系
'''
id_cls = {}   # {cid: cname}
id_cls["other"] = 183
with open("./labels.txt", "r") as f:
    for line in f:
        cid, cname = line.strip().split(": ")
        id_cls[int(cid)] = cname

'''
    得到 灰度值 和 category 的对应关系 cat_dic
'''
cats_path = "./concat_cat.txt"
label_gray = []
with open(cats_path, "r") as f:
    cat_f = f.read().splitlines()
    for i, j in enumerate(cat_f):
        cat_dic[int(j)] = i
        label_gray.append(int(j))

label_colours = np.random.randint(255, size=(len(cat_dic), 3))
label_colours[-1] = [255, 255, 255]

in_channels = len(cat_dic)


def load_config(config_path, display=False):
    """
    load model config
    @param config_path: xxx-project.yaml
    @param display:
    @return:
    """
    config = OmegaConf.load(config_path)
    if display:
        print(yaml.dump(OmegaConf.to_container(config)))
    return config


def load_vqgan_recon(config, ckpt_path=None):
    """
    load model and checkpoint
    @param config:
    @param ckpt_path:
    @return:
    """
    model = instantiate_from_config(config.model)
    if ckpt_path is not None:
        sd = torch.load(ckpt_path, map_location="cpu")["state_dict"]
        missing, unexpected = model.load_state_dict(sd, strict=False)
    return model.eval()

def load_vqgan(config, ckpt_path=None):
    """
    load model and checkpoint
    @param config:
    @param ckpt_path:
    @return:
    """
    model = instantiate_from_config(config.model)
    if ckpt_path is not None:
        sd = torch.load(ckpt_path, map_location="cpu")["state_dict"]
        missing, unexpected = model.load_state_dict(sd, strict=False)
    return model.eval()


def cat_to_arr(x):
    """
    tensor(1,c,h,w) --> ndarray(h,w)
    @param dict:
    @return:
    """
    x = x[0].argmax(0)   # tensor(h,w)
    x = x.cpu().numpy().astype(np.uint8)   # ndarray(h,w)
    return x


@torch.no_grad()
def reconstruct_with_vqgan(vqmodel, x):
    """

    @param x: tensor(1,c,256,256)
    @return: tensor(1,c,256,256)
    """
    z, _, [_, _, indices] = vqmodel.encode(x)    # tensor(1,256,16,16)
    xrec = vqmodel.decode(z)   # tensor(1,c,256,256)
    return xrec[0]

@torch.no_grad()
def reconstruct_with_vqgan_recon(Gmodel, x):
    xrec_sp, _ = Gmodel(x)
    return xrec_sp[0], xrec_sp[1]

def arr_to_ten(x):
    """
    ndarray(h,w,c) --> tensor(1,c,h,w)
    @param x:
    @return:
    """
    x = torch.tensor(x, dtype=torch.float32)
    x = torch.unsqueeze(x, 0)
    if len(x.shape) == 3:
        x = x[..., None]
    x = x.permute(0, 3, 1, 2).to(memory_format= torch.contiguous_format)

    return x.float()


def con_preprocess_image(image_path, size=512, crop=True):
    """
    image_path --> ndarray(size,size,c)
    @param image_path:
    @param size:
    @param crop:
    @return:
    """
    rescaler = albumentations.SmallestMaxSize(max_size=size)
    cropper = albumentations.CenterCrop(height=size, width=size)
    image = Image.open(image_path)

    image = np.array(image).astype(np.uint8)
    b = np.unique(image)   # 数组去重，排序后输出（一维数组）
    image = rescaler(image=image)["image"]
    if crop == True:
        image = cropper(image=image)["image"]
    h, w = image.shape
    c = [[0, -1], [0, 1], [-1, 0], [0, 1], [-1, -1], [1, -1], [-1, 1], [1, 1]]  # 上,下,左,右,左上，右上，左下，右下
    for i in range(h):
        for j in range(w):
            if image[i][j] not in b:
                temp = 0
                for k in c:
                    e1 = k[0] + i
                    e2 = k[1] + j
                    if e1 < h and e2 < w and image[e1][e2] in b:
                        image[i][j] = image[e1][e2]
                        temp = 1
                        break
                if temp == 0:
                    image[i][j] = 255

    new_image = np.zeros((size, size, len(cat_dic)))
    for i in range(h):
        for j in range(w):
            try:
                dic_poz = cat_dic[image[i][j]]
            except:
                dic_poz = len(cat_dic) - 1
            new_image[i][j][dic_poz] = 1
    new_image.astype(np.uint8)
    return new_image


def reconstruction_pipeline(vqmodel, Gmodel, img_path=None, size=512):
    a = con_preprocess_image(img_path, size)
    a = arr_to_ten(a).to(DEVICE)
    x, _ = reconstruct_with_vqgan_recon(vqmodel, a)
    img = cat_to_arr(x)   # ndarray(h,w)
    
    x_sp1, x_sp2 = reconstruct_with_vqgan_recon(Gmodel, a)
    sp1_img = cat_to_arr(x_sp1)
    sp2_img = cat_to_arr(x_sp2)
    
    return img, sp1_img, sp2_img


def P2RGB(x):
    """
    channel 1 --> channel 3, ndarray(h,w) --> ndarray(h,w,3) --> PIL.Image
    @param x:
    @return:
    """
    x = np.array([label_colours[c % in_channels] for c in x])
    x = x.astype(np.uint8)
    x = Image.fromarray(x)
    return x


def total_pro(b):
    """
    ndarray(h,w,c) --> tensor(1,c,h,w) --> ndarray(h,w)
    @param b:
    @return:
    """
    b = arr_to_ten(b).to(DEVICE)
    b = cat_to_arr(b)
    return b


def merge_img(context, object):
    """

    @param context: ndarray(h,w)
    @param object: ndarray(h,w)
    @return:
    """
    scene = np.where(object == in_channels - 1, context, object)
    return scene


def slic_process(im):
    myp = P2RGB(im)
    labels = segmentation.felzenszwalb(myp, scale=16, sigma=0.5, min_size=2048)
    labels = labels.reshape(im.shape[0] * im.shape[1])
    u_labels = np.unique(labels)
    l_inds = []
    for i in range(len(u_labels)):
        l_inds.append(np.where(labels == u_labels[i])[0])
    im_target = im.reshape(1, -1)[0]
    for i in range(len(l_inds)):
        labels_per_sp = im_target[l_inds[i]]
        u_labels_per_sp = np.unique(labels_per_sp)
        hist = np.zeros(len(u_labels_per_sp))
        for j in range(len(hist)):
            hist[j] = len(np.where(labels_per_sp == u_labels_per_sp[j])[0])
        im_target[l_inds[i]] = u_labels_per_sp[np.argmax(hist)]
    im_target = im_target.reshape(im.shape).astype(np.uint8)
    return im_target


def label_mask(img):
    """
    获取分割图的标注蒙版
    @param img:
    @return:
    """
    img_arr = np.array(img)
    h, w, _ = img_arr.shape
    u_label = np.unique(img_arr.reshape(h*w, -1), axis=0)

    cls_rgb = {}
    for rgb in u_label:
        try:
            cat = np.where(np.all((label_colours == rgb), axis=1)==True)[0][0]
        except:
            continue
        gray = label_gray[cat]
        if gray == 255:
            cname = "unlabeled"
        else:
            cname = id_cls[gray+1]
        cls_rgb[cname] = rgb

    num_cats = len(cls_rgb)
    transp = Image.new('RGBA', (w,h))
    draw = ImageDraw.Draw(transp)
    draw.rectangle([(w-125,h-12*num_cats-10),(w,h)], fill=(255, 255, 255, 100)) 
    i = 0
    for cls, rgb in cls_rgb.items():
        draw.rectangle((w-120, h-12*(num_cats-i)-5, w-108, h-12*(num_cats-i)-5+10), fill=tuple(rgb), outline=(0,0,0))
        draw.text((w-103, h-12*(num_cats-i)-5), cls, size=15, font=font, fill=(0,0,0,225))
        i = i+1
    img.paste(transp, mask=transp)

    return img


def show_img(img, title=""):
    plt.imshow(img)
    plt.axis(False)
    plt.title(title)
    plt.show()

P = "<curpath>"
t = "<coarse_checkpoint_name>"
model_name = "custom_vqgan_clip_sp"
LOG_P = os.path.join(P, "logs", f"{t}_{model_name}")
vqconfigs = load_config(os.path.join(LOG_P, f"configs/{t}-project.yaml"))
vqconfigs.model.params.ddconfig.z_channels -= vqconfigs.model.params.noise_dim
vqconfigs.model.params.cliplossconfig.params.device = DEVICE
vqconfigs.model.params.cliplossconfig.params.shape_clip.device = DEVICE
vqconfigs.model.params.cliplossconfig.params.layout_clip.device = DEVICE
vqmodel = load_vqgan_recon(vqconfigs, ckpt_path=os.path.join(LOG_P, f"checkpoints/last.ckpt")).to(DEVICE)

t = "<refine_checkpoint_name>"
model_name = "custom_vqgan_clip_sp"
LOG_P = os.path.join(P, "logs", f"{t}_{model_name}")
Gconfigs = load_config(os.path.join(LOG_P, f"configs/{t}-project.yaml"))
Gconfigs.model.params.ddconfig.z_channels -= Gconfigs.model.params.noise_dim
Gconfigs.model.params.cliplossconfig.params.device = DEVICE
Gconfigs.model.params.cliplossconfig.params.shape_clip.device = DEVICE
Gconfigs.model.params.cliplossconfig.params.layout_clip.device = DEVICE
Gmodel = load_vqgan_recon(Gconfigs, ckpt_path=os.path.join(LOG_P, f"checkpoints/last.ckpt")).to(DEVICE)


if __name__ == '__main__':

    save_dir = f"{P}/clip_sp_result/{t}_{model_name}_ade"
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)

    test_images_obj = "<test_obj_file_path>"
    test_images_con = "<test_gt_file_path>"

    paths = dict()
    with open(test_images_obj, "r") as f:
        paths["test_object"] = f.read().splitlines()
    with open(test_images_con, "r") as f:
        paths["test_context"] = f.read().splitlines()
    
    size = 128
    num_images = len(paths["test_object"])
    for i in tqdm(range(num_images)):
        object_path = paths["test_object"][i]
        object = con_preprocess_image(object_path, size) 
        object = total_pro(object) 
        
        result = Image.new("RGB", (size*3, size*4))
        
        for ii in range(3):
            pred_context_list = reconstruction_pipeline(vqmodel, Gmodel, object_path, size)
            i = 0
            for pred_context in pred_context_list:   # coarse, refined
                pred_scene = merge_img(pred_context, object)
                pred_scene = P2RGB(pred_scene)   # PIL_img
                pred_scene_labeled = label_mask(pred_scene)
                result.paste(pred_scene_labeled, box=(size*i, (ii+1)*size))
                i += 1
        
        object = P2RGB(object)
        object_labeled = label_mask(object)
        
        result.paste(object_labeled, box=(0, 0))          

        img_name = object_path.split('/')[-1]
        new_img_path = os.path.join(save_dir, img_name)
        result.save(new_img_path)

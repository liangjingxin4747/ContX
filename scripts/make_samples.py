import argparse, os, sys, glob, math, time
import torch
import numpy as np
from omegaconf import OmegaConf
from PIL import Image
from main import instantiate_from_config, DataModuleFromConfig
from torch.utils.data import DataLoader
from torch.utils.data.dataloader import default_collate
from tqdm import trange


def save_image(x, path):
    c,h,w = x.shape
    assert c==3
    x = ((x.detach().cpu().numpy().transpose(1,2,0)+1.0)*127.5).clip(0,255).astype(np.uint8)
    Image.fromarray(x).save(path)


@torch.no_grad()
def run_conditional(model, dsets, outdir, top_k, temperature, batch_size=1):
    '''
        model: taming.models.cond_transformer.Net2NetTransformer
        dsets: main.DataModuleFromConfig
    '''
    # 准备dset
    if len(dsets.datasets) > 1:
        split = sorted(dsets.datasets.keys())[0]   # train
        dset = dsets.datasets[split]   # CustomTrain
    else:
        dset = next(iter(dsets.datasets.values()))
    print("Dataset: ", dset.__class__.__name__)

    # 分批采样
    for start_idx in trange(0, len(dset)-batch_size+1, batch_size):
        # 从dset中采样
        indices = list(range(start_idx, start_idx+batch_size))
        # dset[indice]的属性：'file_path', 'segmentation_path', 'image'(1,683,1024,3), 'segmentation'
        example = default_collate([dset[i] for i in indices])

        # 调整原始输入图像tensor(1,3,683,1024)，保存至originals/
        x = model.get_input("image", example).to(model.device)   # example["image"]
        for i in range(x.shape[0]):
            save_image(x[i], os.path.join(outdir, "originals",
                                          "{:06}.png".format(indices[i])))

        cond_key = model.cond_stage_key   # 'segmentation'
        c = model.get_input(cond_key, example).to(model.device)   # example['segmentation'] tensor(1,182,683,1024)

        # encode
        scale_factor = 1.0
        # first_stage_model
        # quan_z: tensor(1,256,42,64)  z_indices: tensor(1,2688)   downsampleing blocks=4, 42h=683H/2^m, 64w=1024W/2^m, 2688=42*64
        quant_z, z_indices = model.encode_to_z(x)   # Zq——a collection of codebook entries  z_indices——a sequence of h*w indices
        # cond_stage_model
        # quan_c: tensor(1,256,42,64) c_indices: tensor(2688,)
        quant_c, c_indices = model.encode_to_c(c)   # c——additional information

        cshape = quant_z.shape   # (1,256,42,64)

        # reconstruct 保存至reconstructions/
        xrec = model.Gmodel.decode(quant_z)   # xrec=G(Zq) tensor(1,3,683,1024)
        for i in range(xrec.shape[0]):
            save_image(xrec[i], os.path.join(outdir, "reconstructions",
                                             "{:06}.png".format(indices[i])))

        if cond_key == "segmentation":
            # get image from segmentation mask
            num_classes = c.shape[1]   # 182
            c = torch.argmax(c, dim=1, keepdim=True)
            c = torch.nn.functional.one_hot(c, num_classes=num_classes)
            c = c.squeeze(1).permute(0, 3, 1, 2).float()
            c = model.cond_stage_model.to_rgb(c)

        idx = z_indices   # tensor(1,2688)

        half_sample = False
        if half_sample:
            start = idx.shape[1] // 2
        else:
            start = 0

        idx[:,start:] = 0   # tensor(1,2688)=0
        idx = idx.reshape(cshape[0], cshape[2], cshape[3])  # tensor(1,42,64) = mask?
        start_i = start // cshape[3]   # 0
        start_j = start % cshape[3]    # 0

        cidx = c_indices   # tensor(2688,)
        cidx = cidx.reshape(quant_c.shape[0],quant_c.shape[2],quant_c.shape[3])   # tensor(1,42,64)

        sample = True

        for i in range(start_i,cshape[2]-0):   # [0,42)
            # set local_i
            if i <= 8:
                local_i = i
            elif cshape[2]-i < 8:   # i < 42-8
                local_i = 16-(cshape[2]-i)
            else:
                local_i = 8
            for j in range(start_j,cshape[3]-0):   # [0,64)
                # set local_j
                if j <= 8:
                    local_j = j
                elif cshape[3]-j < 8:   # j < 64-8
                    local_j = 16-(cshape[3]-j)
                else:
                    local_j = 8

                i_start = i-local_i
                i_end = i_start+16
                j_start = j-local_j
                j_end = j_start+16
                patch = idx[:,i_start:i_end,j_start:j_end]       # 取出大小为16*16的patch
                patch = patch.reshape(patch.shape[0],-1)         # 展开成sequence(1,256)
                cpatch = cidx[:, i_start:i_end, j_start:j_end]   # (16,16)
                cpatch = cpatch.reshape(cpatch.shape[0], -1)     # (1,256)
                patch = torch.cat((cpatch, patch), dim=1)        # (1,512)
                # model.transformer: taming.modules.transformer.mingpt.GPT
                logits,_ = model.transformer(patch[:,:-1])       # (1,511,1024)
                logits = logits[:, -256:, :]                     # (1,256,1024)
                logits = logits.reshape(cshape[0],16,16,-1)      # (1,16,16,1024)
                logits = logits[:,local_i,local_j,:]      # (1,1024)

                logits = logits/temperature

                if top_k is not None:
                    logits = model.top_k_logits(logits, top_k)   # (1,1024)   除了前top_k个其余置为-inf
                # apply softmax to convert to probabilities
                probs = torch.nn.functional.softmax(logits, dim=-1)   # （1,1024)
                # sample from the distribution or take the most likely
                if sample:
                    # 根据给定权重对数组进行多次采样，返回采样后的元素下标
                    ix = torch.multinomial(probs, num_samples=1)   # 取出probs最大的下标(1,1)
                else:
                    _, ix = torch.topk(probs, k=1, dim=-1)
                idx[:,i,j] = ix

        xsample = model.decode_to_img(idx[:,:cshape[2],:cshape[3]], cshape)   # idx(1,42,64)->(1,)
        for i in range(xsample.shape[0]):
            save_image(xsample[i], os.path.join(outdir, "samples",
                                                "{:06}.png".format(indices[i])))

# 参数设置
def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-r",
        "--resume",
        type=str,
        nargs="?",
        help="load from logdir or checkpoint in logdir",
    )
    parser.add_argument(
        "-b",
        "--base",
        nargs="*",
        metavar="base_config.yaml",
        help="paths to base configs. Loaded from left-to-right. "
        "Parameters can be overwritten or added with command-line options of the form `--key value`.",
        default=list(),
    )
    parser.add_argument(
        "-c",
        "--config",
        nargs="?",
        metavar="single_config.yaml",
        help="path to single config. If specified, base configs will be ignored "
        "(except for the last one if left unspecified).",
        const=True,
        default="",
    )
    parser.add_argument(
        "--ignore_base_data",
        action="store_true",
        help="Ignore data specification from base configs. Useful if you want "
        "to specify a custom datasets on the command line.",
    )
    parser.add_argument(
        "--outdir",
        required=True,
        type=str,
        help="Where to write outputs to.",
    )
    parser.add_argument(
        "--top_k",
        type=int,
        default=100,
        help="Sample from among top-k predictions.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=1.0,
        help="Sampling temperature.",
    )
    return parser

# 加载模型
def load_model_from_config(config, sd, gpu=True, eval_mode=True):
    '''
        config: config.model
        sd: state_dict (from checkpoing)
    '''
    # 修改config
    if "ckpt_path" in config.params:
        print("Deleting the restore-ckpt path from the config...")
        config.params.ckpt_path = None
    if "downsample_cond_size" in config.params:
        print("Deleting downsample-cond-size from the config and setting factor=0.5 instead...")
        config.params.downsample_cond_size = -1
        config.params["downsample_cond_factor"] = 0.5
    try:
        if "ckpt_path" in config.params.Gconfigs.params:
            config.params.Gconfigs.params.ckpt_path = None
            print("Deleting the first-stage restore-ckpt path from the config...")
        if "ckpt_path" in config.params.cond_stage_config.params:
            config.params.cond_stage_config.params.ckpt_path = None
            print("Deleting the cond-stage restore-ckpt path from the config...")
    except:
        pass

    # 加载模型Net2NetTransformer
    model = instantiate_from_config(config)
    # 加载模型参数
    if sd is not None:
        # missing_keys 列表包含了所有缺失的关键字, unexpected_keys 列表包含了意料之外的关键字.
        missing, unexpected = model.load_state_dict(sd, strict=False)   # 加载模型参数
        print(f"Missing Keys in State Dict: {missing}")
        print(f"Unexpected Keys in State Dict: {unexpected}")
    if gpu:
        model.cuda()
    if eval_mode:
        model.eval()
    return {"model": model}


def get_data(config):
    # get data
    # 1. 用config.data.params初始化config.data.target指定的类
    data = instantiate_from_config(config.data)    # main.DataModuleFromConfig
    # 2. 创建custom.CustomTrain和CustomTest类
    data.prepare_data()
    # 3. 准备DataModuleFromConfig.datasets(train/validation: 上述两个类)
    data.setup()
    return data


def load_model_and_dset(config, ckpt, gpu, eval_mode):
    # get data (DataModuleFromConfig类)
    dsets = get_data(config)   # calls data.config ...

    # now load the specified checkpoint
    if ckpt:
        pl_sd = torch.load(ckpt, map_location="cpu")
        global_step = pl_sd["global_step"]
    else:
        pl_sd = {"state_dict": None}
        global_step = None
    # get model
    model = load_model_from_config(config.model,
                                   pl_sd["state_dict"],
                                   gpu=gpu,
                                   eval_mode=eval_mode)["model"]
    return dsets, model, global_step


if __name__ == "__main__":
    # sys.path.append(os.getcwd())
    os.chdir('/home/liangxin01/code/dyh/model/taming-transformer/')

    # 获取参数
    parser = get_parser()

    # 运行命令中会传入所有需要设置的选项值，但是有时候仅获取到基本设置时可能要进行一些操作然后才继续导入剩下的参数设置
    opt, unknown = parser.parse_known_args()

    ckpt = None
    if opt.resume:
        # 判断logdir是否存在
        if not os.path.exists(opt.resume):
            raise ValueError("Cannot find {}".format(opt.resume))
        if os.path.isfile(opt.resume):
            # ../logs/logdir/xxx/xxx.ckpt
            paths = opt.resume.split("/")   # ['..', 'logs', 'logdir', 'xxx', 'xxx.ckpt']
            try:
                idx = len(paths)-paths[::-1].index("logs")+1   # 5 - 3 + 1 = 3
            except ValueError:
                idx = -2 # take a guess: path/to/logdir/checkpoints/model.ckpt
            logdir = "/".join(paths[:idx])   # ../logs/logdir
            ckpt = opt.resume   # model path
        else:
            # ../logs/logdir/
            assert os.path.isdir(opt.resume), opt.resume
            logdir = opt.resume.rstrip("/")   # 删除末尾 ../logs/logdir
            ckpt = os.path.join(logdir, "checkpoints", "last.ckpt")   # model path
        print(f"logdir:{logdir}")
        # glob.glob('通配符字符串') 找到相关文件的路径，并返回路径列表
        base_configs = sorted(glob.glob(os.path.join(logdir, "configs/*-project.yaml")))   # config file
        print("base_configs:", base_configs)
        # list of base configs
        opt.base = base_configs+opt.base

    if opt.config:   # path to single config
        if type(opt.config) == str:   # if single config exist, ignore base configs
            opt.base = [opt.config]
        else:
            opt.base = [opt.base[-1]]

    # 加载config文件中的设置到configs
    configs = [OmegaConf.load(cfg) for cfg in opt.base]
    cli = OmegaConf.from_dotlist(unknown)
    #  --ignore_base_data data="{target: main.DataModuleFromConfig, params: {batch_size: 1, validation: {target: taming.data.ade20k.Examples}}}"
    if opt.ignore_base_data:
        for config in configs:
            if hasattr(config, "data"): del config["data"]
    config = OmegaConf.merge(*configs, cli)

    print(ckpt)
    gpu = True
    eval_mode = True
    show_config = False
    if show_config:
        print(OmegaConf.to_container(config))

    # 载入模型和数据集
    dsets, model, global_step = load_model_and_dset(config, ckpt, gpu, eval_mode)
    print(f"Global step: {global_step}")

    # 创建输出文件夹
    outdir = os.path.join(opt.outdir, "{:06}_{}_{}".format(global_step,
                                                           opt.top_k,
                                                           opt.temperature))
    os.makedirs(outdir, exist_ok=True)
    print("Writing samples to ", outdir)
    for k in ["originals", "reconstructions", "samples"]:
        os.makedirs(os.path.join(outdir, k), exist_ok=True)

    run_conditional(model, dsets, outdir, opt.top_k, opt.temperature)

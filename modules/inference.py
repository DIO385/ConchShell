import os
import random
import numpy as np
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import DataLoader
from torch.nn.parallel import DistributedDataParallel as DDP
import torch
import time
import argparse
import torch.distributed as dist
import sys
from torchvision import transforms
from PIL import Image
import soundfile as sf
from tqdm import tqdm
import shutil

sys.path.append("../")
from jukebox.hparams import Hyperparams
from jukebox.make_models import make_vae_model

from dataloader import ConchShellDataset
from module import ConchShellDeep, Audio2Mel

from TCNN.TCNN_modules.tcnn_module import TCNN_ablation
from I3D.I3D_module import InceptionI3d
import warnings

warnings.filterwarnings("ignore")

transform_compose = transforms.Compose([
    # Pre-normalization and re-standardization.
    transforms.Resize((224, 224)),  # Specify the size of the picture to 224*224.
    transforms.ToTensor(),
])


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--local_rank', type=int, default=-1, metavar='N',
                        help='Local process rank.')  # you need this argument in your scripts for DDP to work

    parser.add_argument("--jukebox_type", default='5b', required=False)
    parser.add_argument("--save_path", default='../inference_sample')
    parser.add_argument("--img_path", required=True)

    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--jukebox_low_model", default='../jukebox_models/vqvae_low.pt', required=False)
    parser.add_argument("--tcnn_model", default='../TCNN/TCNN_models/tcnn__epoch=201.pth', required=False)
    parser.add_argument("--i3d_model", default='../I3D/I3D_models/rgb_imagenet.pt', required=False)

    return parser.parse_args()


def sample_by_picture(gen, jukebox, tcnn, i3d, args, device):
    os.makedirs(args.save_path, exist_ok=True)

    code_level = 0
    level_s = 0
    level_e = 1

    # tcnn
    itf, genre = tcnn_get_time_feature(tcnn, args.img_path, device)

    # i3d
    i3d_f = i3d.extract_features(itf).squeeze(0)  # [1024, 2, 1, 1]
    i3d_f = i3d_f.flatten(start_dim=0).unsqueeze(0).unsqueeze(0)  # [1, 2048]

    # conchshell and jukebox
    with torch.no_grad():
        pred_xs = gen.forward(i3d_f, genre)
        xs_code = []
        for _ in range(3):
            xs_code.append(pred_xs)
        zs_pred = jukebox.bottleneck.encode(xs_code)
        zs_pred_code = []
        zs_pred_code.append(zs_pred[code_level])

        _, pred_audio = jukebox._decode(zs_pred_code, start_level=level_s, end_level=level_e)

        pred_audio = pred_audio.squeeze().detach().cpu().numpy()
        sample_generated = f'{args.img_path.split("/")[-1].split(".")[0]}.wav'
        sample_generated_path = os.path.join(args.save_path, sample_generated)
        sf.write(sample_generated_path, pred_audio, 16000)
    print(f"{sample_generated}save successful!。")


def tcnn_get_time_feature(net, img_path, device):
    img = Image.open(img_path)
    img = transform_compose(img).unsqueeze(0)  # [3, 224, 224]
    img = img.to(device)
    with torch.no_grad():
        outputs = net.get_time_feature(img)
        img_time_feature = []
        for j, scale in enumerate(outputs):
            c_num = outputs[j].shape[1]
            interval = c_num // 3
            c_f = []
            for k in range(3):
                # [5,1,224,224]
                cur_f = scale.squeeze(0)[k * (interval + 1):k * (interval + 1) + interval, :, :, :]
                cf = cur_f[0, :, :, :]  # [1, 224, 224]
                for c in range(1, cur_f.shape[0]):
                    cf += cur_f[c, :, :, :]
                cf = cf / cur_f.shape[0]
                c_f.append(cf)

            # Get the feature for each convolutional layer
            per_scale_f = torch.cat(c_f, dim=0).unsqueeze(dim=1)  # [3, 1, 224, 224]
            img_time_feature.append(per_scale_f)
            img_time_feature.append(per_scale_f)
            img_time_feature.append(per_scale_f)

        itf = torch.cat(img_time_feature, dim=1)  # [3, 18, 224, 224]

        # get label
        genere = net.forward(img)  # [1, 6]
        one_hot_z = torch.zeros(genere.shape[1])
        one_hot_z[np.argmax(genere.cpu()).item()] = 1.0  # one-hot
        return itf.unsqueeze(0), torch.FloatTensor(one_hot_z.unsqueeze(0)).to(device)


def main(**kwargs):
    """
    python3 -m torch.distributed.launch --nproc_per_node=1 --nnodes=1 --node_rank 0 --master_port 8888 inference.py --img_path <img_path> --checkpoint <checkpoint>
    """
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    args = parse_args()
    dist.init_process_group(backend='nccl', rank=args.local_rank)
    torch.cuda.set_device(args.local_rank)
    device = args.local_rank if args.local_rank != -1 else (
        torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu'))
    print(f"Using device:{device}")

    hps = Hyperparams(**kwargs)

    """ init conchshell """
    gen_net = ConchShellDeep().to(device)
    gen_net.load_state_dict(torch.load(args.checkpoint))
    gen_net.eval()

    """ init jukebox """
    jukebox_net = make_vae_model(args.jukebox_type, device, hps).to(device)
    jukebox_net.load_state_dict(torch.load(args.jukebox_low_model))
    jukebox_net.eval()

    """ init tcnn """
    tcnn_net = TCNN_ablation().to(device)
    tcnn_net.load_state_dict(torch.load(args.tcnn_model), True)
    tcnn_net.eval()

    """ init i3d """
    i3d_net = InceptionI3d().to(device)
    i3d_net.load_state_dict(torch.load(args.i3d_model))
    i3d_net.eval()

    sample_by_picture(gen_net, jukebox_net, tcnn_net, i3d_net, args, device)


if __name__ == '__main__':
    main()

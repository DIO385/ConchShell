import os
import random

import numpy as np
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import StepLR
import torch
import time
import argparse
import torch.distributed as dist
import sys
import torch.nn.utils.prune as prune

sys.path.append("../")
from jukebox.hparams import Hyperparams
from jukebox.make_models import make_vae_model
import soundfile as sf
from dataloader import ConchShellDataset
from module import ConchShellDeep, Discriminator, Audio2Mel
import matplotlib.pyplot as plt
from tqdm import tqdm
import shutil
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
import warnings

warnings.filterwarnings("ignore")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--local_rank', type=int, default=1, metavar='N',
                        help='Local process rank.')  # you need this argument in your scripts for DDP to work

    parser.add_argument("--jukebox_type", default='5b')
    parser.add_argument("--save_sample_path", default='../sample/')
    parser.add_argument("--model_path", default='../checkpoints/')
    parser.add_argument("--save_loss_img_path", default='../loss/')
    parser.add_argument("--save_log_path", default='../log/')

    parser.add_argument("--is_log", type=bool, default=True)
    parser.add_argument("--is_sample", type=bool, default=True)

    parser.add_argument("--num_D", type=int, default=3)
    parser.add_argument("--n_layers_D", type=int, default=4)

    parser.add_argument("--batch_size", type=int, default=8)

    parser.add_argument("--epochs", type=int, default=3000)
    parser.add_argument("--save_interval", type=int, default=10, help="The epoch interval for saving the model.")
    parser.add_argument("--sample_interval", type=int, default=10)
    parser.add_argument("--plot_loss_interval", type=int, default=10)
    return parser.parse_args()


def model_prune(net, linear_amount=0.2, is_remove=False):
    for name, module in net.named_modules():
        # prune 20% of connections in all linear layers
        if isinstance(module, torch.nn.Linear):
            prune.l1_unstructured(module, name='weight', amount=linear_amount)
            if is_remove:
                prune.remove(module, name='weight')


def dumping_original_audio(data_loader, device):
    test_audio = []
    test_picture = []
    test_genre = []
    test_img_path = []
    for i, (img, a_t, genre, img_path) in enumerate(data_loader):
        test_picture.append(img.float().to(device))
        test_audio.append(a_t.float().to(device))
        test_genre.append(genre.float().to(device))
        test_img_path.append(img_path[0])
    print("Finish dumping samples", len(test_audio), len(test_picture), len(test_img_path))
    return test_audio, test_picture, test_genre, test_img_path


def logs(arg, g_loss, d_loss, epoch):
    os.makedirs(arg.save_log_path, exist_ok=True)

    log = f"Epoch:{epoch}\t" \
          f"[loss_G]:%.4f\t" \
          f"[loss_feat]:%.4f\t" \
          f"[xs_error]:%.4f\t" \
          f"[code_error]:%.4f\t" \
          f"[audio_error]:%.4f\t" \
          f"[mel_error]:%.4f\t" \
          f"[final_g_loss]:%.4f\t" \
          f"[loss_D]:%.4f\t" \
          f"Time:{time.asctime()}\n" % (
              g_loss[0], g_loss[1], g_loss[2], g_loss[3], g_loss[4], g_loss[5], g_loss[6], d_loss)

    log_path = os.path.join(arg.save_log_path, f"ConchShell_log.txt")
    with open(log_path, mode="a", encoding="utf8") as log_file:
        log_file.write(log)
        log_file.close()


def plot_g_loss(arg, g_loss, epoch):
    """ Plot the generator's loss variation curve. """
    plt.figure(figsize=(14, 14))

    plt.subplot(3, 3, 1)
    plt.plot(g_loss[:, 0], ms=0.1, label="loss_G")
    plt.legend(loc='best')
    plt.ylabel("loss")

    plt.subplot(3, 3, 2)
    plt.plot(g_loss[:, 1], ms=0.1, label="loss_feat")
    plt.legend(loc='best')

    plt.subplot(3, 3, 3)
    plt.plot(g_loss[:, 2], ms=0.1, label="xs_error")
    plt.legend(loc='best')

    plt.subplot(3, 3, 4)
    plt.plot(g_loss[:, 3], ms=0.1, label="code_error")
    plt.legend(loc='best')
    plt.ylabel("loss")

    plt.subplot(3, 3, 5)
    plt.plot(g_loss[:, 4], ms=0.1, label="audio_error")
    plt.legend(loc='best')
    plt.xlabel("epoch")

    plt.subplot(3, 3, 6)
    plt.plot(g_loss[:, 5], ms=0.1, label="mel_error")
    plt.legend(loc='best')
    plt.xlabel("epoch")

    plt.subplot(3, 3, 7)
    plt.plot(g_loss[:, 6], ms=0.1, label="final_loss")
    plt.legend(loc='best')
    plt.xlabel("epoch")
    plt.ylabel("loss")

    plt.suptitle("Generator's loss")
    plt.savefig(f'{arg.save_loss_img_path}/Generator-{epoch}.jpg')
    # plt.show()
    plt.clf()


def plot_d_loss(arg, d_loss, epoch):
    """ Plot the discriminator's loss variation curve. """

    plt.subplot(1, 1, 1)
    plt.plot(d_loss, ms=0.1, label="loss_D")
    plt.legend(loc='best')
    plt.ylabel("loss")
    plt.xlabel("epoch")

    plt.suptitle("Discriminator's loss")
    plt.savefig(f'{arg.save_loss_img_path}/Discriminator-{epoch}.jpg')
    # plt.show()
    plt.clf()


def plot_loss(arg, g_loss, d_loss, epoch):
    """ Plotting loss variation curves for generators and discriminators. """
    os.makedirs(arg.save_loss_img_path, exist_ok=True)

    g_loss = np.array(g_loss)
    d_loss = np.array(d_loss)
    if len(g_loss) <= 1:
        return

    plot_g_loss(arg, g_loss, epoch)
    plot_d_loss(arg, d_loss, epoch)


def sample(gen, jukebox, test_picture, test_audio, test_genre, test_img_path, args, epoch):
    os.makedirs(args.save_sample_path, exist_ok=True)

    code_level = 0
    level_s = 0
    level_e = 1

    gen.eval()
    with torch.no_grad():
        for i in range(1, 4):
            # Three samples were randomly selected for testing.
            index = random.randint(0, len(test_picture) - 1)

            p_t = test_picture[index]
            a_t = test_audio[index]
            genre = test_genre[index]
            img_path = test_img_path[index]

            # Save the original audio.
            audio = a_t.squeeze()  # [44100]
            sample_original = f'original_{epoch}_{i}.wav'
            sample_original = os.path.join(args.save_sample_path, sample_original)
            sf.write(sample_original, audio.detach().cpu().numpy(), 16000)

            # Save the audio generated using the image.
            pred_xs = gen(p_t, genre)
            xs_code = []
            for _ in range(3):
                xs_code.append(pred_xs)
            zs_pred = jukebox.bottleneck.encode(xs_code)
            zs_pred_code = []
            zs_pred_code.append(zs_pred[code_level])
            _, pred_audio = jukebox._decode(zs_pred_code, start_level=level_s, end_level=level_e)
            pred_audio = pred_audio.squeeze().detach().cpu().numpy()
            sample_generated = f'generated_{epoch}_{i}.wav'
            sample_generated_path = os.path.join(args.save_sample_path, sample_generated)
            sf.write(sample_generated_path, pred_audio, 16000)

            # Save the image.
            _img_path = os.path.join("../dataset/Img/val_set", img_path)
            img_name = img_path.split("/")[-1]
            img_dir = img_path.split("/")[0]
            sample_img_path = os.path.join(args.save_sample_path, "img", img_dir)
            if not os.path.exists(sample_img_path):
                os.makedirs(sample_img_path)

            sample_img = os.path.join(sample_img_path, img_name.replace(".jpg", f"_{epoch}_{i}.jpg"))
            shutil.copy(_img_path, sample_img)


def compute_g_loss(dis, fft, a_t, xs_pred, D_real, xs_t, xs_quantised_gt, audio_pred, code_level, args):
    """ Calculate the loss of the generator. """
    g_cost = []

    xs_error = F.l1_loss(xs_t[code_level].view(args.batch_size, 1, -1), xs_pred.view(args.batch_size, 1, -1))
    code_error = F.l1_loss(xs_quantised_gt[0].view(args.batch_size, 1, -1), xs_pred.view(args.batch_size, 1, -1))

    audio_error = F.l1_loss(a_t.transpose(1, 2), audio_pred)

    mel_t = fft(a_t)  # [B, 128, 172]
    mel_pred = fft(audio_pred.transpose(1, 2))
    mel_error = F.l1_loss(mel_t, mel_pred)

    D_fake = dis.forward(xs_pred)
    loss_G = 0
    for scale in D_fake:
        loss_G += -scale[-1].mean()

    loss_feat = 0
    feat_weights = 4.0 / (args.n_layers_D + 1)  # 4/5=0.8
    D_weights = 1.0 / args.num_D  # 1/3≈0.33
    wt = D_weights * feat_weights  # 4/15≈0.27

    for i in range(args.num_D):
        for j in range(len(D_fake[i]) - 1):
            loss_feat += wt * F.l1_loss(D_fake[i][j], D_real[i][j].detach())

    final_loss = loss_G + 3 * loss_feat + 5 * xs_error + 8 * code_error + 40 * audio_error + 15 * mel_error
    g_cost.append([loss_G.item(),
                   loss_feat.item(),
                   xs_error.item(),
                   code_error.item(),
                   audio_error.item(),
                   mel_error.item(),
                   final_loss.item()])
    return final_loss, g_cost


def train(gen, dis, jukebox, fft, optG, optD, train_loader, args, device):
    code_level = 0
    level_s = 0
    level_e = 1

    gen.train()
    dis.train()
    dist.barrier()
    g_costs, d_costs, n = [], [], 0
    ggloss, ddloss = 0.0, 0.0
    for num_step, (img, a_t, genre, _) in enumerate(tqdm(train_loader)):
        a_t = a_t.float().to(device)  # [B, 1, 128000]
        p_t = img.float().to(device)  # [B, 1, 2048]
        genre = genre.float().to(device)  # [B, 6]

        xs_pred = gen.forward(p_t, genre)  # [B, 64, 16000]

        with torch.no_grad():
            xs_t, zs_t = jukebox._encode(a_t.transpose(1, 2))

            # pred output
            xs_code = []
            for _ in range(3):
                xs_code.append(xs_pred)

            zs_pred = jukebox.bottleneck.encode(xs_code)  # len=3
            zs_pred_code = zs_pred[code_level]  # [B, 16000]

            xs_quantised_pred, audio_pred = jukebox._decode([zs_pred_code], start_level=level_s,
                                                            end_level=level_e)
            gt_code = [zs_t[code_level]]  # [B, 16000]
            xs_quantised_gt, gt_audio = jukebox._decode(gt_code, start_level=level_s,
                                                        end_level=level_e)

        dis.zero_grad()
        xs_t, zs_t = jukebox._encode(a_t.transpose(1, 2))

        xs_pred = xs_pred.view(args.batch_size, 1, -1)  # [B, 64, 16000]->[B, 1, 1024000]
        xs_tmp = xs_t[code_level].view(args.batch_size, 1, -1)  # [B, 64, 16000]->[B, 1, 1024000]
        D_fake_det = dis.forward(xs_pred.to(device).detach())
        D_real = dis.forward(xs_tmp.to(device))

        d_loss = 0
        for fake_scale, real_scale in zip(D_fake_det, D_real):
            d_loss += F.relu(1 + fake_scale[-1]).mean()
            d_loss += F.relu(1 - real_scale[-1]).mean()
        d_loss.backward()
        optD.step()

        gen.zero_grad()
        g_loss, g_cost = compute_g_loss(dis, fft, a_t, xs_pred, D_real, xs_t, xs_quantised_gt, audio_pred, code_level,
                                        args)
        g_loss.backward()
        optG.step()

        ggloss += g_loss.item()
        ddloss += d_loss.item()
        g_costs.append(g_cost[0])
        n += 1
    train_g_loss = ggloss / n
    train_d_loss = ddloss / n
    if args.is_master:
        print(f'device:{args.local_rank}\tGloss:%.4f\tDloss:%.4f' % (train_g_loss, train_d_loss))
    _g_cost = np.array(g_costs).sum(0)
    return _g_cost / _g_cost.shape[0], train_d_loss


def train_main(gen, dis, jukebox, fft, train_set, test_set, args, device):
    """
    :param gen: generator
    :param dis: discriminator
    :param jukebox: jukebox is a pre-trained model for coding and decoding audio.
    :param fft: Converting audio to Mel spectrograms.
    :param train_set: training set
    :param test_set: testing set
    """
    os.makedirs(args.model_path, exist_ok=True)
    torch.backends.cudnn.benchmark = True

    train_sampler = DistributedSampler(train_set)
    train_loader = DataLoader(train_set, sampler=train_sampler, batch_size=args.batch_size, num_workers=8,
                              drop_last=True)

    # val_sampler = DistributedSampler(test_set)
    val_sampler = None
    test_loader = DataLoader(test_set, sampler=val_sampler, batch_size=1, num_workers=8, drop_last=True)

    optG = torch.optim.AdamW(gen.parameters(), lr=1e-5, weight_decay=1e-5, betas=(0.5, 0.999), eps=0.00001)
    optD = torch.optim.AdamW(dis.parameters(), lr=1e-5, weight_decay=1e-5, betas=(0.5, 0.999), eps=0.00001)

    g_lr_scheduler = StepLR(optG, step_size=200, gamma=0.95)
    d_lr_scheduler = StepLR(optD, step_size=200, gamma=0.9)

    if args.is_master:
        test_audio, test_picture, test_genre, test_img_path = dumping_original_audio(data_loader=test_loader,
                                                                                     device=device)
    G_costs, D_costs = [], []
    for epoch in range(0, args.epochs + 1):
        train_sampler.set_epoch(epoch)
        # val_sampler.set_epoch(epoch)
        if args.is_master:
            print(f"epoch{epoch + 1}")

        # train_fun returns a list has all_loss
        g_cost, d_cost = train(gen=gen, dis=dis, jukebox=jukebox, fft=fft,
                               optG=optG, optD=optD, train_loader=train_loader,
                               args=args, device=device)
        g_lr_scheduler.step()
        d_lr_scheduler.step()

        # model_prune(dis)
        # model_prune(gen)

        if args.is_master:
            G_costs.append(g_cost)
            D_costs.append(d_cost)

        # sampling
        if args.is_master and epoch % args.sample_interval == 0 and args.is_sample:
            sample(gen, jukebox, test_picture, test_audio, test_genre, test_img_path, args, epoch + 1)

        # save BeachBoy model
        if args.is_master and epoch % args.save_interval == 0:
            print("Gen's lr: ", optG.state_dict()['param_groups'][0]['lr'])
            print("Dis's lr: ", optD.state_dict()['param_groups'][0]['lr'])

            # Generator
            torch.save(gen.module.state_dict(), os.path.join(args.model_path, f"G-epoch={epoch + 1}.pth"))
            # torch.save(optG.state_dict(), os.path.join(args.model_path, f"optG-epoch={epoch + 1}.pth"))

            # Discriminator
            torch.save(dis.module.state_dict(), os.path.join(args.model_path, f"D-epoch={epoch + 1}.pth"))
            # torch.save(optD.state_dict(), os.path.join(args.model_path, f"optD-epoch={epoch + 1}.pth"))

        # log
        if args.is_master and args.is_log:
            logs(args, g_cost, d_cost, epoch + 1)

        # save loss img
        if args.is_master and epoch % args.plot_loss_interval == 0:
            plot_loss(args, G_costs, D_costs, epoch + 1)


def main(**kwargs):
    """
    python3 -m torch.distributed.launch --nproc_per_node=1 --nnodes=1 --node_rank=0 --master_port=9999 train.py
    """
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    args = parse_args()
    args.is_master = args.local_rank == 0  # Only the master device will record logs etc.
    dist.init_process_group(backend='nccl', rank=args.local_rank)
    torch.cuda.set_device(args.local_rank)

    device = args.local_rank if args.local_rank != -1 else (
        torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu'))
    print(f"Using device:{device}, is master:{args.is_master}")
    hps = Hyperparams(**kwargs)

    jukebox_net = make_vae_model(args.jukebox_type, device, hps).to(device)
    jukebox_net.load_state_dict(torch.load("../jukebox_models/vqvae_low.pt"))
    jukebox_net.eval()

    gen_net = ConchShellDeep().to(device)
    # gen_net.load_state_dict(torch.load("../checkpoints/G-epoch=0.pth"))
    gen_net = DDP(gen_net, device_ids=[args.local_rank])

    dis_net = Discriminator(num_D=3, ndf=32, n_layers=4, downsampling_factor=4).to(device)
    # dis_net.load_state_dict(torch.load("../checkpoints/D-epoch=0.pth"))
    dis_net = DDP(dis_net, device_ids=[args.local_rank])

    fft_fun = Audio2Mel(n_mel_channels=128).to(device)

    # You can also use data that has not been enhanced with data.
    # Just replace 'Label.xlsx' with 'Label_small.xlsx'.
    train_set = ConchShellDataset(
        wav_root_path=r"../dataset/Wav/",
        img_files=r"../dataset/Npy/i3d_npy_train/",
        xlsx_path=r"../dataset/Label/Label.xlsx",  # ← replace 'Label.xlsx' with 'Label_small.xlsx' if you need.
        augment=True)

    val_set = ConchShellDataset(
        wav_root_path=r"../dataset/Wav/",
        img_files=r"../dataset/Npy/i3d_npy_val/",
        xlsx_path=r"../dataset/Label/Label.xlsx",  # ← replace it if you need.
        augment=False)

    train_main(gen=gen_net,
               dis=dis_net,
               jukebox=jukebox_net,
               fft=fft_fun,
               train_set=train_set,
               test_set=val_set,
               args=args, device=device)


if __name__ == '__main__':
    main()

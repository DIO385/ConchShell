import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os

from torch.utils.data import DataLoader

from I3D_module import InceptionI3d
from I3D_data_loader import I3DDataset

NPY_TRAIN_PATH = r"/home/fanwanpeng/home/fanwanpeng/yy_project_demo/yy_test/BeachBoy/BB_dataset/BeachBoyImg/i3d_npy_train/"
NPY_VAL_PATH = r"/home/fanwanpeng/home/fanwanpeng/yy_project_demo/yy_test/BeachBoy/BB_dataset/BeachBoyImg/i3d_npy_val/"


def extract_i3d_feature(net, data_loader):
    net.eval()
    with torch.no_grad():
        n = 1
        for x, npy_path in data_loader:
            x = x.to(device)
            print(f"{n}/{len(data_loader)}", end="\t")
            # 提取i3d特征，用于输入BeachBoy的生成器
            i3d_f = i3d.extract_features(x).squeeze(0)  # [1024, 2, 1, 1]
            i3d_f = i3d_f.flatten(start_dim=0).unsqueeze(0)  # [1, 2048],能够输入生成器

            # 保存到npy文件里面
            save_i3d_feature_2_npy(i3d_f, npy_path, NPY_VAL_PATH)
            n += 1


def save_i3d_feature_2_npy(i3d_f, path, save_npy_path):
    """ 保存i3d特征为npy文件
    :param i3d_f:特征矩阵 [1, 2048]
    :param path:由VGG提取出来的，包含图片-时间特征的npy文件路径
    :param save_npy_path:保存i3d特征的npy文件路径
    """
    # 开始保存为npy文件
    path_info = path[0].split("/")
    img_type = path_info[-2]  # 拿到类型，例如：coast-0
    npy_name = path_info[-1]  # 拿到npy文件名，例如：1-1.npy
    npy_dir_path = os.path.join(save_npy_path, img_type)
    if not os.path.exists(npy_dir_path):
        os.makedirs(npy_dir_path)
    npy_path = npy_dir_path + f"/{npy_name}"
    np.save(npy_path, i3d_f.cpu().numpy())
    print(f"{npy_path}  保存成功")
    # t = np.load(npy_path)
    # print(t.shape)


if __name__ == '__main__':
    device = torch.device('cuda:2' if torch.cuda.is_available() else 'cpu')
    IMG_NPY_TRAIN_PATH = r"/home/fanwanpeng/home/fanwanpeng/yy_project_demo/yy_test/BBP-VGG/TCNN/TCNN_feature_npy/tcnn_train_npy/"
    IMG_NPY_VAL_PATH = r"/home/fanwanpeng/home/fanwanpeng/yy_project_demo/yy_test/BBP-VGG/TCNN/TCNN_feature_npy/tcnn_test_npy/"
    i3d_data_set = I3DDataset(img_npy_path=IMG_NPY_VAL_PATH)
    i3d_loader = DataLoader(i3d_data_set, batch_size=1, num_workers=0, shuffle=False)

    i3d = InceptionI3d().to(device)
    i3d.load_state_dict(torch.load('I3D_models/rgb_imagenet.pt'))
    extract_i3d_feature(i3d, i3d_loader)

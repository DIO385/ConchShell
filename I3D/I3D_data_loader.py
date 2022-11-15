import torch
import torch.utils.data
import os

from torch.utils.data import DataLoader
import numpy as np


def files_to_list(npy_dir_path):
    """
    返回img的npy文件全路径
    """
    all_npy_path = []
    for root, _, files in os.walk(npy_dir_path):
        if root == npy_dir_path:
            continue
        all_npy_path.extend([os.path.join(root, file) for file in files])
    return all_npy_path


class I3DDataset(torch.utils.data.Dataset):
    def __init__(self, img_npy_path):
        self.img_npy_path = files_to_list(img_npy_path)  # 图片img的npy文件路径list

    def __len__(self):
        return len(self.img_npy_path)

    def __getitem__(self, index):
        npy_path = self.img_npy_path[index]
        x_img = self.load_npy_to_torch(npy_path)  # [3, 18, 224, 224]
        return x_img, npy_path

    def load_npy_to_torch(self, path):
        # print(path)
        x_img = torch.Tensor(np.load(path, encoding='bytes', allow_pickle=True))
        return x_img


if __name__ == '__main__':
    i3d_data_set = I3DDataset(img_npy_path=r"D:\DATA_SET\BeachBoyData\img_for_vgg\train_npy_set")
    i3d_loader = DataLoader(i3d_data_set, batch_size=16, num_workers=0, shuffle=False)
    print(len(i3d_loader))
    for i, (x, npy_path) in enumerate(i3d_loader):
        print(x.shape)
        print(npy_path)
        input("a")

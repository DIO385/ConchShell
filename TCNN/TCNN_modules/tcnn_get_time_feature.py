import torch
from torch.utils.data import DataLoader
import os
import numpy as np
from tcnn_module import TCNN, TCNN_ablation
from tcnn_data_loader import TCNNDataset
from I3D_module import InceptionI3d

NPY_TRAIN_PATH = r"../BB_dataset/BeachBoyNpy/i3d_npy_train/"
NPY_VAL_PATH = r"../BB_dataset/BeachBoyNpy/i3d_npy_val/"


def get_time_to_npy(net, i3d_net, data_loader):
    net.eval()
    with torch.no_grad():
        for i, (x, _, img_path) in enumerate(data_loader):
            x = x.to(device)
            # img_path = img_path.to(device)
            outputs = net.get_time_feature(x)
            # print(len(outputs))
            # print(outputs[0].shape)
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
                per_scale_f = torch.cat(c_f, dim=0).unsqueeze(dim=1)  # [3, 1, 224, 224]
                img_time_feature.append(per_scale_f)
                img_time_feature.append(per_scale_f)
                img_time_feature.append(per_scale_f)
            itf = torch.cat(img_time_feature, dim=1).unsqueeze(0)  # [3, 18, 224, 224]

            i3d_f = i3d_net.extract_features(itf).squeeze(0)  # [1024, 2, 1, 1]
            i3d_f = i3d_f.flatten(start_dim=0).unsqueeze(0)  # [1, 2048]

            save_i3d_feature_2_npy(i3d_f, img_path, NPY_TRAIN_PATH)


def save_i3d_feature_2_npy(i3d_f, path, save_npy_path):
    """
    :param i3d_f: [1, 2048]
    """
    path_info = path[0].split("/")
    img_type = path_info[-2]
    npy_name = path_info[-1].replace(".jpg", ".npy")
    npy_dir_path = os.path.join(save_npy_path, img_type)
    if not os.path.exists(npy_dir_path):
        os.makedirs(npy_dir_path)
    npy_path = npy_dir_path + f"/{npy_name}"
    np.save(npy_path, i3d_f.cpu().numpy())
    print(f"{npy_path}  save successfully")


if __name__ == '__main__':
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model = TCNN().to(device)
    model.load_state_dict(torch.load("../TCNN_models/tcnn__net3_epoch=201.pth", map_location=f"{device}"), True)
    vgg_data_set = TCNNDataset(x_img_path=r"../../dataset/Img/train_set/",
                               data_type="train")
    vgg_val_loader = DataLoader(vgg_data_set, batch_size=1, num_workers=0, shuffle=False)

    i3d = InceptionI3d().to(device)
    i3d.load_state_dict(torch.load('../../I3D/I3D_models/rgb_imagenet.pt'))
    get_time_to_npy(model, i3d, vgg_val_loader)

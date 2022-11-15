import torch
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
from torch.nn.utils import weight_norm


def WNConv2d(*arg, **kwargs):
    # return weight_norm(nn.Conv2d(*arg, **kwargs))
    return nn.Conv2d(*arg, **kwargs)


class ResnetBlock(nn.Module):
    def __init__(self, dim, dilation=1):
        super().__init__()
        self.block = nn.Sequential(
            # nn.LeakyReLU(0),
            nn.ReflectionPad2d(dilation),  # Mirror fill with {dilation} values left and right
            WNConv2d(dim, dim, kernel_size=3, dilation=dilation),
            # nn.LeakyReLU(0),
            WNConv2d(dim, dim, kernel_size=1),
        )
        self.shortcut = WNConv2d(dim, dim, kernel_size=1)

    def forward(self, x):
        # return self.shortcut(x) + self.block(x)
        return x + self.block(x)


class TCNN_ablation(nn.Module):
    """ A TCNN network without ResnetBlock.
    This is an ablation experiment to see how effective it is with or without ResnetBlock.
    """

    def __init__(self, class_num=6):
        """
        :param class_num: There are 6 kinds of pictures:
                beach, coral reef, deep sea, coast, island and waves.
        """
        super(TCNN_ablation, self).__init__()
        self.conv2_drop = nn.Dropout2d()
        self.future1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=(7, 7), dilation=(2, 2), padding=6)
        self.future2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=(7, 7), padding=3)
        self.future3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(5, 5), dilation=(2, 2), padding=4)
        self.future4 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(5, 5), padding=2)
        self.future5 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=(3, 3), dilation=(2, 2), padding=2)
        self.future6 = nn.Conv2d(in_channels=256, out_channels=128, kernel_size=(3, 3), padding=1)

        self.classifer = nn.Sequential(
            nn.Dropout(0.4),
            nn.Sigmoid(),
            nn.Linear(128 * 3 * 3, 1000),
            nn.Dropout(0.4),
            nn.Sigmoid(),
            nn.Linear(1000, 256),
            nn.Dropout(0.4),
            nn.Sigmoid(),
            nn.Linear(256, class_num),
            nn.Sigmoid()
        )

    def forward(self, x):
        x1 = F.relu(F.max_pool2d(self.future1(x), 2))  # [B, 20, 112, 112]

        x2 = F.relu(F.max_pool2d(self.future2(x1), 2))  # [B, 40, 56, 56]
        x3 = F.relu(F.max_pool2d(self.future3(x2), 2))  # [B, 80, 28, 28]
        x4 = F.relu(F.max_pool2d(self.future4(x3), 2))  # [B, 144, 14, 14]
        x5 = F.relu(F.max_pool2d(self.future5(x4), 2))  # [B, 224, 7, 7]
        x6 = F.relu(F.max_pool2d(self.conv2_drop(self.future6(x5)), 2))  # [B, 144, 3, 3]

        # flatten it out from the second dimension.
        x = x6.flatten(start_dim=1)  # [B, 1296]
        x = self.classifer(x)

        return x

    def get_time_feature(self, x):
        time_img_feature = []

        x1 = self.future1(x)  # [B, 20, 112, 112]

        x2 = self.future2(x1)  # [B, 40, 56, 56]
        x3 = self.future3(x2)  # [B, 80, 28, 28]
        x4 = self.future4(x3)  # [B, 144, 14, 14]
        x5 = self.future5(x4)  # [B, 224, 7, 7]
        x6 = self.future6(x5)  # [B, 144, 3, 3]

        # Collect the results of each layer in preparation
        # for later generation of the "time feature".
        time_img_feature.append([x1, x2, x3, x4, x5, x6])
        time_img_future = time_img_feature[0]
        for i, t in enumerate(time_img_future):
            time_img_future[i] = time_img_future[i].unsqueeze(2)

        return time_img_future

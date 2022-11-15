import sys,os
sys.path.append(os.path.dirname(__file__) + os.sep + '../')
import torch
import torch.utils.data
from TCNN_modules.tcnn_data_loader import TCNNDataset


def getStat(train_data):
    '''
    Compute mean and variance for training yujie_data
    '''
    print('Compute mean and variance for training data.')
    print(len(train_data))
    train_loader = torch.utils.data.DataLoader(
        train_data, batch_size=1, shuffle=False, num_workers=0,
        pin_memory=True)
    mean = torch.zeros(3)
    std = torch.zeros(3)
    for X, _ in train_loader:
        for d in range(3):
            mean[d] += X[:, d, :, :].mean()
            std[d] += X[:, d, :, :].std()
    mean.div_(len(train_data))
    std.div_(len(train_data))
    return list(mean.numpy()), list(std.numpy())



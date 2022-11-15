import torch
import torch.utils.data
import os

from torch.utils.data import DataLoader
from torchvision import transforms
from PIL import Image


def files_to_list(img_path):
    """ Returns the path to all images. """
    all_img_path = []
    for root, _, files in os.walk(img_path):
        if root == img_path:
            continue
        all_img_path.extend([os.path.join(root, file) for file in files])
    return all_img_path


class TCNNDataset(torch.utils.data.Dataset):
    def __init__(self, x_img_path, data_type):
        self.x_img_files = files_to_list(x_img_path)  # a list of image paths.
        self.data_type = data_type
        if self.data_type == "train":
            self.transform_standard = transforms.Normalize(
                mean=[0.3462801, 0.47744608, 0.5486459],  # depend on your train data set.
                std=[0.22619946, 0.20212314, 0.22058737]
            )
        else:  # val
            self.transform_standard = transforms.Normalize(
                mean=[0.34610438, 0.47745505, 0.54867613],  # depend on your val data set.
                std=[0.22629566, 0.20211853, 0.22066636]
            )
        self.transform_compose = transforms.Compose([
            # Pre-normalization and re-standardization.
            transforms.Resize((224, 224)),  # Specify the size of the picture to 224*224.
            transforms.ToTensor(),
            self.transform_standard
        ])

    def __len__(self):
        return len(self.x_img_files)

    def __getitem__(self, index):
        img_path = self.x_img_files[index]
        x_img, y_label = self.preprocess_img(img_path)
        return x_img, y_label, img_path

    def preprocess_img(self, path):
        img = Image.open(path)
        x_img = self.transform_compose(img)  # [3, 224, 224]
        img_label = path.split("\\")[-2].split("-")[-1]  # get label of img
        if img_label == "1":  # coast
            y_label = [0]
        elif img_label == "2":  # island
            y_label = [1]
        elif img_label == "3":  # sand_beach
            y_label = [2]
        elif img_label == "4":  # coral_reef
            y_label = [3]
        elif img_label == "5":  # deep
            y_label = [4]
        else:  # ocean_wave
            y_label = [5]

        return x_img, torch.tensor(y_label).squeeze()



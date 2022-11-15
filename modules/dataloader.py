import torch
import torch.utils.data
import torch.nn.functional as F
from librosa.core import load
from librosa.util import normalize
import pandas as pd

import numpy as np
import random
import os


def files_to_list(img_npy_path):
    all_img_path = []
    for root, _, files in os.walk(img_npy_path):
        if root == img_npy_path:
            continue
        for file in files:
            """
                You can also use data that has not been enhanced with data.
                Just open the following two lines of code.
            """
            # if "-1." in file or "-2." in file or "-3." in file or "-4." in file or "-5." in file:
            #     continue
            all_img_path.append(os.path.join(root, file))
    return all_img_path


class ConchShellDataset(torch.utils.data.Dataset):
    def __init__(self, wav_root_path, img_files, xlsx_path, sampling_rate=16000, augment=True):
        self.sampling_rate = sampling_rate
        self.segment_length = sampling_rate * 8  # Duration is 8 seconds.
        self.img_files = files_to_list(img_files)
        self.wav_root_path = wav_root_path
        self.df = pd.read_excel(xlsx_path, sheet_name="Sheet1")
        self.augment = augment

    def __len__(self):
        return len(self.img_files)

    def __getitem__(self, index):
        # Get the image(.npy file) path.
        img_filename = self.img_files[index]
        img_npy = self.load_img_to_torch(img_filename)  # [1, 2048]

        # Load audio
        img_path, audio_filename, label = self.get_label_and_path(img_filename)
        audio = self.load_wav_to_torch(audio_filename)  # audio:[128000]

        if audio.size(0) == self.segment_length:
            pass
        elif audio.size(0) > self.segment_length:
            max_audio_start = audio.size(0) - self.segment_length
            audio_start = random.randint(0, max_audio_start)
            audio = audio[audio_start: audio_start + self.segment_length]
        else:
            audio = F.pad(audio, (0, self.segment_length - audio.size(0)), "constant").data

        # audio = audio / 32768.0

        # 读取风格
        if label == "1":
            genre = [1.0, 0.0, 0.0, 0.0, 0.0, 0.0]  # coast
        elif label == "2":
            genre = [0.0, 1.0, 0.0, 0.0, 0.0, 0.0]  # island
        elif label == "3":
            genre = [0.0, 0.0, 1.0, 0.0, 0.0, 0.0]  # sand_beach
        elif label == "4":
            genre = [0.0, 0.0, 0.0, 1.0, 0.0, 0.0]  # coral_reef
        elif label == "5":
            genre = [0.0, 0.0, 0.0, 0.0, 1.0, 0.0]  # deep
        else:
            genre = [0.0, 0.0, 0.0, 0.0, 0.0, 1.0]  # ocean_wave

        return img_npy, audio.unsqueeze(0), torch.FloatTensor(genre), img_path

    def load_wav_to_torch(self, wav_path):
        data, _ = load(wav_path, sr=self.sampling_rate)
        data = 0.95 * normalize(data)
        data = normalize(data)
        if self.augment:
            amplitude = np.random.uniform(low=0.5, high=1.0)
            data = data * amplitude

        return torch.from_numpy(data).float()

    def load_img_to_torch(self, img_npy_path):
        data = np.load(img_npy_path)
        return torch.from_numpy(data).float()

    def get_label_and_path(self, x_path):
        """
        :param x_path: the path of image
        """
        path_info = x_path.split("/")
        img_npy_name = path_info[-1]
        img_type = path_info[-2]
        temp = self.df[self.df["img_type"] == img_type]
        item = temp[self.df["img_npy_name"] == img_npy_name]
        genre = item["genre"].item()
        muse_wav_name = item["muse_wav_name"].item()

        wav_path = os.path.join(self.wav_root_path, genre, muse_wav_name)
        img_path = os.path.join(img_type, img_npy_name.replace("npy", "jpg"))
        return img_path, wav_path, img_type[-1]

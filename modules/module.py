import torch.nn as nn
import torch.nn.functional as F
import torch
from librosa.filters import mel as librosa_mel_fn
from torch.nn.utils import weight_norm


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find("BatchNorm2d") != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


def WNConv1d(*args, **kwargs):
    return weight_norm(nn.Conv1d(*args, **kwargs))


class Audio2Mel(nn.Module):
    def __init__(
            self,
            n_fft=2048,
            hop_length=300,
            win_length=1200,
            sampling_rate=16000,
            n_mel_channels=128,  # 128
            mel_fmin=0.0,
            mel_fmax=None, ):
        super().__init__()

        window = torch.hann_window(win_length).float()  # [1024]
        mel_basis = librosa_mel_fn(sampling_rate, n_fft, n_mel_channels, mel_fmin, mel_fmax)  # [B, 513, 172]
        mel_basis = torch.from_numpy(mel_basis).float()
        self.register_buffer("mel_basis", mel_basis)
        self.register_buffer("window", window)
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length
        self.sampling_rate = sampling_rate
        self.n_mel_channels = n_mel_channels  # 128

    def forward(self, audio):  # [B, 1, 44100]
        p = (self.n_fft - self.hop_length) // 2  # 384
        audio = F.pad(audio, pad=(p, p), mode="reflect").squeeze(1)  # [B, 44868]
        fft = torch.stft(
            audio,  # [B, 44868]
            n_fft=self.n_fft,  # 1024
            hop_length=self.hop_length,  # 256
            win_length=self.win_length,  # 1024
            window=self.window,  # [1024]
            center=False,
        )  # [B, 513, 172, 2]
        real_part, imag_part = fft.unbind(-1)
        magnitude = torch.sqrt(real_part ** 2 + imag_part ** 2)
        mel_output = torch.matmul(self.mel_basis, magnitude)  # [B, 128, 172]
        log_mel_spec = torch.log10(torch.clamp(mel_output, min=1e-5))
        return log_mel_spec


class ResnetBlock(nn.Module):
    def __init__(self, dim, dilation=1):
        super().__init__()
        self.block = nn.Sequential(
            nn.LeakyReLU(0.2),
            nn.ReflectionPad1d(dilation),
            WNConv1d(dim, dim, kernel_size=3, dilation=dilation),
            nn.LeakyReLU(0.2),
            WNConv1d(dim, dim, kernel_size=1),
        )
        self.shortcut = WNConv1d(dim, dim, kernel_size=1)

    def forward(self, x):
        return self.shortcut(x) + self.block(x)


class NLayerDiscriminator(nn.Module):
    def __init__(self, ndf, n_layers, downsampling_factor):
        """
        :param ndf: 32
        :param n_layers: 4
        :param downsampling_factor: 4
        """
        super().__init__()
        model = nn.ModuleDict()

        model["layer_0"] = nn.Sequential(
            nn.ReflectionPad1d(7),
            WNConv1d(1, ndf, kernel_size=16),
            nn.LeakyReLU(0.2, True),
            # nn.LeakyReLU(0.2),
        )
        nf = ndf
        stride = downsampling_factor
        for n in range(1, n_layers + 1):
            nf_prev = nf
            nf = min(nf * stride, 1024)

            model["layer_%d" % n] = nn.Sequential(
                WNConv1d(
                    nf_prev,
                    nf,
                    kernel_size=stride * 10 + 1,
                    stride=stride,
                    padding=stride * 5,
                    groups=nf_prev // 4,
                ),
                nn.LeakyReLU(0.2, inplace=True),
                # nn.LeakyReLU(0.2),
            )

        nf = min(nf * 2, 512)
        model["layer_%d" % (n_layers + 1)] = nn.Sequential(
            WNConv1d(nf_prev, nf, kernel_size=10, stride=1, padding=1),
            nn.LeakyReLU(0.2, True),
        )

        model["layer_%d" % (n_layers + 2)] = WNConv1d(nf, 1, kernel_size=3, stride=1, padding=1)

        self.model = model

    def forward(self, x):
        results = []
        for key, layer in self.model.items():
            x = layer(x)
            results.append(x)
        return results  # len=7


class Discriminator(nn.Module):
    def __init__(self, num_D, ndf, n_layers, downsampling_factor):
        super().__init__()
        self.model = nn.ModuleDict()
        for i in range(num_D):
            self.model[f"disc_{i}"] = NLayerDiscriminator(
                ndf, n_layers, downsampling_factor
            )
        self.downsample = nn.AvgPool1d(kernel_size=4,
                                       stride=2,
                                       padding=1,
                                       count_include_pad=False)
        self.apply(weights_init)

    def forward(self, x):
        # x:[B, 1, 64000]
        results = []
        for key, disc in self.model.items():
            results.append(disc(x))
            x = self.downsample(x)
        return results


class ConchShellHigh(nn.Module):
    """ For ablation experiments """

    def __init__(self):
        super().__init__()
        self.lin = nn.Linear(2048 + 256, 4800)
        self.genre_embed = nn.Embedding(6, 256)

        model = [nn.Conv1d(1, 64, kernel_size=10, stride=2, padding=1)]
        model += [ResnetBlock(64, dilation=3 ** 0)]
        model += [ResnetBlock(64, dilation=3 ** 1)]
        model += [
            nn.LeakyReLU(0.2),
            nn.Conv1d(64, 128, kernel_size=20, stride=2, padding=1),
        ]
        model += [ResnetBlock(128, dilation=3 ** 0)]
        model += [ResnetBlock(128, dilation=3 ** 1)]
        model += [
            nn.LeakyReLU(0.2),
            nn.Conv1d(128, 256, kernel_size=40, stride=1, padding=1),
        ]
        model += [ResnetBlock(256, dilation=3 ** 0)]
        model += [ResnetBlock(256, dilation=3 ** 1)]
        model += [
            nn.LeakyReLU(0.2),
            nn.Conv1d(256, 512, kernel_size=60, stride=1, padding=1),
        ]
        model += [ResnetBlock(512, dilation=3 ** 0)]
        model += [ResnetBlock(512, dilation=3 ** 1)]
        model += [
            nn.LeakyReLU(0.2),
            nn.Conv1d(512, 256, kernel_size=60, stride=1, padding=1),
        ]
        model += [ResnetBlock(256, dilation=3 ** 0)]
        model += [ResnetBlock(256, dilation=3 ** 1)]
        model += [
            nn.LeakyReLU(0.2),
            nn.Conv1d(256, 64, kernel_size=42, stride=1, padding=1),
            nn.Tanh(),
        ]
        self.model = nn.Sequential(*model)
        self.apply(weights_init)

    def forward(self, x, genre):
        x = x.float()
        genre_idx = genre.nonzero(as_tuple=True)[1]
        genre_emb = self.genre_embed(genre_idx)
        genre_emb = genre_emb.unsqueeze(1)
        x = torch.cat((x, genre_emb), 2)
        x = self.lin(x)
        x = self.model(x)

        return x * 100


class ConchShellyLow(nn.Module):
    """ For ablation experiments """

    def __init__(self):
        super().__init__()
        self.lin = nn.Linear(2048 + 256, 4096)
        self.fc = nn.Linear(664, 4000)
        self.genre_embed = nn.Embedding(6, 256)
        model = [
            nn.ReflectionPad1d(16),
            nn.Conv1d(1, 32, kernel_size=40, stride=1, padding=1),
            nn.MaxPool1d(kernel_size=10, stride=2, padding=1, dilation=1),
        ]
        model += [ResnetBlock(32, dilation=3 ** 0)]
        model += [ResnetBlock(32, dilation=3 ** 1)]
        model += [
            nn.LeakyReLU(0.2),
            nn.ReflectionPad1d(32),
            nn.Conv1d(32, 64, kernel_size=40, stride=1, padding=1),
            nn.MaxPool1d(kernel_size=10, stride=2, padding=1, dilation=1),
        ]
        model += [ResnetBlock(64, dilation=3 ** 0)]
        model += [ResnetBlock(64, dilation=3 ** 1)]
        model += [
            nn.LeakyReLU(0.2),
            nn.ReflectionPad1d(64),
            nn.Conv1d(64, 128, kernel_size=40, stride=1, padding=1),
            nn.MaxPool1d(kernel_size=10, stride=2, padding=1, dilation=1),
        ]
        model += [ResnetBlock(128, dilation=3 ** 0)]
        model += [ResnetBlock(128, dilation=3 ** 1)]

        model += [
            nn.LeakyReLU(0.2),
            nn.ReflectionPad1d(64),
            nn.Conv1d(128, 256, kernel_size=40, stride=1, padding=1),
            nn.MaxPool1d(kernel_size=10, stride=2, padding=1, dilation=1),
        ]
        model += [ResnetBlock(256, dilation=3 ** 0)]
        model += [ResnetBlock(256, dilation=3 ** 1)]
        model += [
            nn.LeakyReLU(0.2),
            nn.ReflectionPad1d(64),
            nn.Conv1d(256, 512, kernel_size=40, stride=1, padding=1),
            nn.MaxPool1d(kernel_size=10, stride=1, padding=1, dilation=1),
        ]
        model += [ResnetBlock(512, dilation=3 ** 0)]
        model += [ResnetBlock(512, dilation=3 ** 1)]
        model += [
            nn.LeakyReLU(0.2),
            nn.ReflectionPad1d(64),
            nn.Conv1d(512, 1024, kernel_size=40, stride=1, padding=1),
            nn.MaxPool1d(kernel_size=10, stride=1, padding=1, dilation=1),
        ]
        model += [ResnetBlock(1024, dilation=3 ** 0)]
        model += [ResnetBlock(1024, dilation=3 ** 1)]
        model += [
            nn.LeakyReLU(0.2),
            nn.ReflectionPad1d(64),
            nn.Conv1d(1024, 256, kernel_size=40, stride=1, padding=1),
            nn.MaxPool1d(kernel_size=10, stride=1, padding=1, dilation=1),
        ]
        model += [
            nn.ReflectionPad1d(64),
            nn.Conv1d(256, 64, kernel_size=40, stride=1, padding=1),
            nn.Tanh(),
        ]
        self.model = nn.Sequential(*model)

    def forward(self, x, genre):
        genre_idx = genre.nonzero(as_tuple=True)[1]
        genre_emb = self.genre_embed(genre_idx)
        genre_emb = genre_emb.unsqueeze(1)
        x = torch.cat((x, genre_emb), 2)
        x = self.lin(x)
        x = self.model(x)
        x = self.fc(x)
        return x * 100


class ConchShellDeep(nn.Module):
    """ Official Model """

    def __init__(self):
        super().__init__()
        self.lin = nn.Linear(2048 + 256, 4096)
        self.fc = nn.Linear(985, 16000)
        self.genre_embed = nn.Embedding(6, 256)
        model = [
            nn.ReflectionPad1d(16),
            nn.Conv1d(1, 32, kernel_size=40, stride=1, padding=1),
            nn.MaxPool1d(kernel_size=10, stride=2, padding=1, dilation=1),
        ]
        model += [ResnetBlock(32, dilation=3 ** 0)]
        model += [ResnetBlock(32, dilation=3 ** 1)]
        model += [
            nn.LeakyReLU(0.2),
            nn.ReflectionPad1d(32),
            nn.Conv1d(32, 64, kernel_size=40, stride=1, padding=1),
            nn.MaxPool1d(kernel_size=10, stride=2, padding=1, dilation=1),
        ]
        model += [ResnetBlock(64, dilation=3 ** 0)]
        model += [ResnetBlock(64, dilation=3 ** 1)]
        model += [
            nn.LeakyReLU(0.2),
            nn.ReflectionPad1d(64),
            nn.Conv1d(64, 128, kernel_size=40, stride=1, padding=1),
            nn.MaxPool1d(kernel_size=10, stride=2, padding=1, dilation=1),
        ]
        model += [ResnetBlock(128, dilation=3 ** 0)]
        model += [ResnetBlock(128, dilation=3 ** 1)]
        model += [
            nn.LeakyReLU(0.2),
            nn.ReflectionPad1d(64),
            nn.Conv1d(128, 256, kernel_size=40, stride=1, padding=1),
            nn.MaxPool1d(kernel_size=10, stride=1, padding=1, dilation=1),
        ]
        model += [ResnetBlock(256, dilation=3 ** 0)]
        model += [ResnetBlock(256, dilation=3 ** 1)]
        model += [
            nn.LeakyReLU(0.2),
            nn.ReflectionPad1d(64),
            nn.Conv1d(256, 512, kernel_size=40, stride=1, padding=1),
            nn.MaxPool1d(kernel_size=10, stride=1, padding=1, dilation=1),
        ]
        model += [ResnetBlock(512, dilation=3 ** 0)]
        model += [ResnetBlock(512, dilation=3 ** 1)]
        model += [
            nn.LeakyReLU(0.2),
            nn.ReflectionPad1d(64),
            nn.Conv1d(512, 1024, kernel_size=40, stride=1, padding=1),
            nn.MaxPool1d(kernel_size=10, stride=1, padding=1, dilation=1),
        ]
        model += [ResnetBlock(1024, dilation=3 ** 0)]
        model += [ResnetBlock(1024, dilation=3 ** 1)]
        model += [
            nn.LeakyReLU(0.2),
            nn.ReflectionPad1d(64),
            nn.Conv1d(1024, 256, kernel_size=40, stride=1, padding=1),
            nn.MaxPool1d(kernel_size=10, stride=1, padding=1, dilation=1),
        ]
        model += [
            nn.ReflectionPad1d(64),
            nn.Conv1d(256, 64, kernel_size=40, stride=1, padding=1),
            nn.Tanh(),
        ]
        self.model = nn.Sequential(*model)
        self.apply(weights_init)

    def forward(self, x, genre):
        genre_idx = genre.nonzero(as_tuple=True)[1]
        genre_emb = self.genre_embed(genre_idx)
        genre_emb = genre_emb.unsqueeze(1)
        x = torch.cat((x, genre_emb), 2)
        x = self.lin(x)
        x = self.model(x)
        x = self.fc(x)
        return x * 100


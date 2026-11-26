# coding=utf-8
"""Discriminators for codec GAN training.

- MultiPeriodDiscriminator (HiFi-GAN): waveform reshaped by period p into 2D.
- MultiScaleSTFTDiscriminator (EnCodec): 2D convs over complex STFT at several resolutions.

Each discriminator returns a list of (logits, feature_maps) pairs, one per sub-discriminator.
"""

from typing import List, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.parametrizations import weight_norm

DiscOut = List[Tuple[torch.Tensor, List[torch.Tensor]]]


class PeriodDiscriminator(nn.Module):
    def __init__(self, period: int, channels: Sequence[int] = (32, 128, 512, 1024, 1024)):
        super().__init__()
        self.period = period
        convs, c_in = [], 1
        for i, c in enumerate(channels):
            stride = 3 if i < len(channels) - 1 else 1
            convs.append(weight_norm(nn.Conv2d(c_in, c, (5, 1), (stride, 1), padding=(2, 0))))
            c_in = c
        self.convs = nn.ModuleList(convs)
        self.post = weight_norm(nn.Conv2d(c_in, 1, (3, 1), 1, padding=(1, 0)))

    def forward(self, x: torch.Tensor):
        # x: (B, 1, T)
        b, c, t = x.shape
        if t % self.period:
            x = F.pad(x, (0, self.period - t % self.period), mode="reflect")
            t = x.shape[-1]
        x = x.view(b, c, t // self.period, self.period)
        feats = []
        for conv in self.convs:
            x = F.leaky_relu(conv(x), 0.1)
            feats.append(x)
        x = self.post(x)
        feats.append(x)
        return x.flatten(1), feats


class MultiPeriodDiscriminator(nn.Module):
    def __init__(self, periods: Sequence[int] = (2, 3, 5, 7, 11)):
        super().__init__()
        self.discs = nn.ModuleList([PeriodDiscriminator(p) for p in periods])

    def forward(self, x: torch.Tensor) -> DiscOut:
        return [d(x) for d in self.discs]


class STFTDiscriminator(nn.Module):
    def __init__(self, n_fft: int, hop: int, win: int, filters: int = 32, max_filters: int = 1024,
                 dilations: Sequence[int] = (1, 2, 4)):
        super().__init__()
        self.n_fft, self.hop, self.win = n_fft, hop, win
        self.register_buffer("window", torch.hann_window(win), persistent=False)
        convs = [weight_norm(nn.Conv2d(2, filters, (3, 9), padding=(1, 4)))]
        c = filters
        for d in dilations:
            c_out = min(c * 2, max_filters)
            convs.append(weight_norm(nn.Conv2d(c, c_out, (3, 9), stride=(1, 2), dilation=(d, 1), padding=(d, 4))))
            c = c_out
        convs.append(weight_norm(nn.Conv2d(c, c, (3, 3), padding=(1, 1))))
        self.convs = nn.ModuleList(convs)
        self.post = weight_norm(nn.Conv2d(c, 1, (3, 3), padding=(1, 1)))

    def forward(self, x: torch.Tensor):
        with torch.autocast(device_type=x.device.type, enabled=False):
            spec = torch.stft(x.squeeze(1).float(), self.n_fft, self.hop, self.win, self.window,
                              return_complex=True, normalized=True)
        z = torch.stack([spec.real, spec.imag], dim=1)  # (B, 2, F, T)
        z = z.transpose(2, 3)  # (B, 2, T, F) -> stride over frequency
        feats = []
        for conv in self.convs:
            z = F.leaky_relu(conv(z), 0.2)
            feats.append(z)
        z = self.post(z)
        feats.append(z)
        return z.flatten(1), feats


class MultiScaleSTFTDiscriminator(nn.Module):
    def __init__(self, n_ffts: Sequence[int] = (2048, 1024, 512, 256, 128)):
        super().__init__()
        self.discs = nn.ModuleList([STFTDiscriminator(n, n // 4, n) for n in n_ffts])

    def forward(self, x: torch.Tensor) -> DiscOut:
        return [d(x) for d in self.discs]


class CombinedDiscriminator(nn.Module):
    def __init__(self, periods=(2, 3, 5, 7, 11), n_ffts=(2048, 1024, 512, 256, 128)):
        super().__init__()
        self.mpd = MultiPeriodDiscriminator(periods)
        self.mstft = MultiScaleSTFTDiscriminator(n_ffts)

    def forward(self, x: torch.Tensor) -> DiscOut:
        return self.mpd(x) + self.mstft(x)

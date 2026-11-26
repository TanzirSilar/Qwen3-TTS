# coding=utf-8
"""Losses for codec training: multi-resolution mel, hinge GAN, feature matching, semantic distillation."""

from typing import List, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio

from discriminators import DiscOut


class MultiResolutionMelLoss(nn.Module):
    """L1 on log-mel at several resolutions (DAC-style)."""

    def __init__(
        self,
        sample_rate: int = 24000,
        win_lengths: Sequence[int] = (32, 64, 128, 256, 512, 1024, 2048),
        n_mels: Sequence[int] = (5, 10, 20, 40, 80, 160, 320),
    ):
        super().__init__()
        self.mels = nn.ModuleList([
            torchaudio.transforms.MelSpectrogram(
                sample_rate=sample_rate, n_fft=w, win_length=w, hop_length=w // 4,
                n_mels=m, power=1.0, center=True, f_min=0.0, f_max=sample_rate / 2,
            )
            for w, m in zip(win_lengths, n_mels)
        ])

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # pred/target: (B, 1, T)
        with torch.autocast(device_type=pred.device.type, enabled=False):
            p, t = pred.squeeze(1).float(), target.squeeze(1).float()
            loss = p.new_zeros(())
            for mel in self.mels:
                mp, mt = mel(p), mel(t)
                loss = loss + F.l1_loss(torch.log(mp.clamp(min=1e-5)), torch.log(mt.clamp(min=1e-5)))
            return loss / len(self.mels)


def discriminator_hinge_loss(real: DiscOut, fake: DiscOut) -> torch.Tensor:
    loss = 0.0
    for (lr, _), (lf, _) in zip(real, fake):
        loss = loss + F.relu(1 - lr).mean() + F.relu(1 + lf).mean()
    return loss / len(real)


def generator_hinge_loss(fake: DiscOut) -> torch.Tensor:
    loss = 0.0
    for lf, _ in fake:
        loss = loss + F.relu(1 - lf).mean()
    return loss / len(fake)


def feature_matching_loss(real: DiscOut, fake: DiscOut) -> torch.Tensor:
    """Relative L1 between discriminator feature maps (EnCodec-style)."""
    loss, n = 0.0, 0
    for (_, fr), (_, ff) in zip(real, fake):
        for a, b in zip(fr, ff):
            a = a.detach()
            loss = loss + (a - b).abs().mean() / (a.abs().mean() + 1e-5)
            n += 1
    return loss / max(n, 1)


def semantic_distill_loss(student: torch.Tensor, teacher: torch.Tensor, mask=None) -> torch.Tensor:
    """1 - cosine similarity per frame. student/teacher: (B, T, D)."""
    cos = F.cosine_similarity(student.float(), teacher.float(), dim=-1)  # (B, T)
    if mask is None:
        return (1 - cos).mean()
    m = mask.float()
    return ((1 - cos) * m).sum() / m.sum().clamp(min=1.0)

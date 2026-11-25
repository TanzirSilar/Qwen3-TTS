# coding=utf-8
"""Speaker Encoder Model - ECAPA-TDNN Architecture."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass


@dataclass
class SpeakerEncoderConfig:
    """Speaker Encoder Configuration."""
    mel_dim: int = 128
    enc_dim: int = 1024
    enc_channels: list = None
    enc_kernel_sizes: list = None
    enc_dilations: list = None
    enc_attention_channels: int = 128
    enc_res2net_scale: int = 8
    enc_se_channels: int = 128
    sample_rate: int = 24000

    def __post_init__(self):
        if self.enc_channels is None:
            self.enc_channels = [512, 512, 512, 512, 1536]
        if self.enc_kernel_sizes is None:
            self.enc_kernel_sizes = [5, 3, 3, 3, 1]
        if self.enc_dilations is None:
            self.enc_dilations = [1, 2, 3, 4, 1]


class Res2NetBlock(nn.Module):
    """Res2Net block for multi-scale feature extraction."""

    def __init__(self, in_channels, out_channels, scale=8, kernel_size=3, dilation=1):
        super().__init__()
        self.scale = scale

        in_channel = in_channels // scale
        hidden_channel = out_channels // scale

        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()

        for i in range(scale - 1):
            self.convs.append(
                nn.Conv1d(
                    in_channel,
                    hidden_channel,
                    kernel_size,
                    padding=dilation * (kernel_size - 1) // 2,
                    dilation=dilation,
                )
            )
            self.bns.append(nn.BatchNorm1d(hidden_channel))

        self.conv_res = nn.Conv1d(in_channels, out_channels, 1)
        self.bn_res = nn.BatchNorm1d(out_channels)

    def forward(self, x):
        """Forward pass."""
        residual = x

        # Split into scale groups
        xs = torch.chunk(x, self.scale, dim=1)
        ys = []

        # First chunk bypassed
        ys.append(xs[0])

        # Process remaining chunks
        y = xs[1]
        for i, (conv, bn) in enumerate(zip(self.convs, self.bns)):
            y = F.relu(bn(conv(y)))
            if i < len(xs) - 2:
                y = y + xs[i + 2]
            ys.append(y)

        # Concatenate
        y = torch.cat(ys, dim=1)

        # Residual connection
        y = y + self.conv_res(residual)
        y = F.relu(self.bn_res(y))

        return y


class SEBlock(nn.Module):
    """Squeeze-and-Excitation block."""

    def __init__(self, channels, squeeze_channels):
        super().__init__()
        self.fc1 = nn.Linear(channels, squeeze_channels)
        self.fc2 = nn.Linear(squeeze_channels, channels)

    def forward(self, x):
        """Forward pass."""
        # Global average pooling: (B, C, T) -> (B, C)
        y = torch.mean(x, dim=2)

        # Excitation
        y = F.relu(self.fc1(y))
        y = torch.sigmoid(self.fc2(y))

        # Scale channels
        y = y.unsqueeze(2)
        return x * y


class AttentiveStatisticsPooling(nn.Module):
    """Attentive Statistics Pooling."""

    def __init__(self, channels, attention_channels):
        super().__init__()
        self.fc1 = nn.Linear(channels, attention_channels)
        self.fc2 = nn.Linear(attention_channels, channels)

    def forward(self, x):
        """Forward pass.

        Args:
            x: (B, C, T) tensor

        Returns:
            (B, 2*C) pooled representation
        """
        # Compute attention weights: (B, T, C)
        weights = torch.softmax(self.fc2(F.relu(self.fc1(x.transpose(1, 2)))), dim=1)

        # Weighted mean
        mean = torch.sum(x.transpose(1, 2) * weights, dim=1)

        # Weighted variance
        var = torch.sum(((x.transpose(1, 2) - mean.unsqueeze(1)) ** 2) * weights, dim=1)

        # Concatenate mean and variance
        return torch.cat([mean, var], dim=1)


class SpeakerEncoder(nn.Module):
    """ECAPA-TDNN Speaker Encoder."""

    def __init__(self, config: SpeakerEncoderConfig):
        super().__init__()
        self.config = config

        # Initial TDNN layer
        self.tdnn = nn.Conv1d(
            config.mel_dim,
            config.enc_channels[0],
            kernel_size=config.enc_kernel_sizes[0],
            padding=config.enc_kernel_sizes[0] // 2,
        )
        self.tdnn_bn = nn.BatchNorm1d(config.enc_channels[0])

        # Res2Net blocks with SE
        self.res2net_blocks = nn.ModuleList()
        self.se_blocks = nn.ModuleList()

        for i in range(1, len(config.enc_channels) - 1):
            res2net = Res2NetBlock(
                config.enc_channels[i - 1],
                config.enc_channels[i],
                scale=config.enc_res2net_scale,
                kernel_size=config.enc_kernel_sizes[i],
                dilation=config.enc_dilations[i],
            )
            se = SEBlock(config.enc_channels[i], config.enc_se_channels)

            self.res2net_blocks.append(res2net)
            self.se_blocks.append(se)

        # Pre-pooling linear layer
        self.pre_pool_linear = nn.Conv1d(
            config.enc_channels[-2],
            config.enc_channels[-1],
            kernel_size=config.enc_kernel_sizes[-1],
            padding=0,
        )
        self.pre_pool_bn = nn.BatchNorm1d(config.enc_channels[-1])

        # Attentive statistics pooling
        self.pool = AttentiveStatisticsPooling(
            config.enc_channels[-1],
            config.enc_attention_channels,
        )

        # Post-pooling linear layer
        self.post_pool_linear = nn.Linear(config.enc_channels[-1] * 2, config.enc_dim)
        self.post_pool_bn = nn.BatchNorm1d(config.enc_dim)

    def forward(self, x):
        """Forward pass.

        Args:
            x: (B, mel_dim, T) mel-spectrogram

        Returns:
            (B, enc_dim) speaker embedding
        """
        # Initial TDNN
        x = F.relu(self.tdnn_bn(self.tdnn(x)))

        # Res2Net blocks with SE
        for res2net, se in zip(self.res2net_blocks, self.se_blocks):
            x = res2net(x)
            x = se(x)

        # Pre-pooling
        x = F.relu(self.pre_pool_bn(self.pre_pool_linear(x)))

        # Attentive statistics pooling
        x = self.pool(x)

        # Post-pooling
        x = F.relu(self.post_pool_bn(self.post_pool_linear(x)))
        x = F.normalize(x, p=2, dim=1)

        return x


class ArcFaceHead(nn.Module):
    """ArcFace classification head with margin and scale."""

    def __init__(self, embedding_dim, num_classes, margin=0.3, scale=30.0):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.num_classes = num_classes
        self.margin = margin
        self.scale = scale

        # Class centers (weight matrix)
        self.weight = nn.Parameter(torch.randn(num_classes, embedding_dim))
        nn.init.xavier_uniform_(self.weight)

        # Register buffer for computation
        self.register_buffer("cos_theta", None)

    def forward(self, embeddings, labels=None):
        """Forward pass.

        Args:
            embeddings: (B, embedding_dim) normalized embeddings
            labels: (B,) class labels (optional, for training)

        Returns:
            logits: (B, num_classes) or similarity scores
        """
        # Normalize embeddings and weight
        embeddings = F.normalize(embeddings, p=2, dim=1)
        weight = F.normalize(self.weight, p=2, dim=1)

        # Compute cosine similarity: (B, num_classes)
        cos_theta = torch.mm(embeddings, weight.t())

        if labels is None:
            # Inference: return raw similarity
            return cos_theta

        # Training: apply ArcFace margin
        # Create one-hot labels
        one_hot = torch.zeros_like(cos_theta)
        one_hot.scatter_(1, labels.unsqueeze(1), 1.0)

        # Compute angle
        theta = torch.acos(torch.clamp(cos_theta, -1.0 + 1e-7, 1.0 - 1e-7))

        # Apply margin
        theta_m = theta + self.margin * one_hot
        cos_theta_m = torch.cos(theta_m)

        # Apply scale
        logits = cos_theta_m * self.scale

        return logits


class SpeakerEncoderWithHead(nn.Module):
    """Speaker Encoder with ArcFace classification head."""

    def __init__(self, config: SpeakerEncoderConfig, num_speakers: int,
                 margin: float = 0.3, scale: float = 30.0):
        super().__init__()
        self.encoder = SpeakerEncoder(config)
        self.head = ArcFaceHead(config.enc_dim, num_speakers, margin, scale)

    def forward(self, x, labels=None):
        """Forward pass.

        Args:
            x: (B, mel_dim, T) mel-spectrogram
            labels: (B,) speaker labels (optional)

        Returns:
            embeddings: (B, enc_dim) speaker embeddings
            logits: (B, num_speakers) classification logits
        """
        embeddings = self.encoder(x)
        logits = self.head(embeddings, labels)
        return embeddings, logits

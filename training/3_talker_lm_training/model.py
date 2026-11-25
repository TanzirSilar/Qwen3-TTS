# coding=utf-8
"""Talker LLM Model - Transformer for speech code prediction."""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from typing import Optional, Tuple


@dataclass
class TalkerConfig:
    """Talker LLM Configuration."""
    vocab_size: int = 151936
    hidden_size: int = 4096
    intermediate_size: int = 22016
    num_hidden_layers: int = 32
    num_attention_heads: int = 32
    num_key_value_heads: int = 32
    head_dim: int = 128
    hidden_act: str = "silu"
    rms_norm_eps: float = 1e-6
    max_position_embeddings: int = 2048
    rope_theta: float = 10000.0

    # Speech specific
    num_codebooks: int = 16
    codebook_size: int = 2048
    speaker_embedding_dim: int = 1024


class RoPEEmbedding(nn.Module):
    """Rotary Position Embedding (RoPE)."""

    def __init__(self, dim: int, max_seq_len: int = 2048, base: float = 10000.0):
        super().__init__()
        self.dim = dim
        self.max_seq_len = max_seq_len
        self.base = base

        # Precompute frequencies
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)

    def forward(self, x: torch.Tensor, seq_len: Optional[int] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute RoPE embeddings.

        Args:
            x: (batch, seq_len, ...)
            seq_len: Sequence length (optional)

        Returns:
            Tuple of (cos, sin) embeddings
        """
        seq_len = seq_len or x.shape[1]
        t = torch.arange(seq_len, device=x.device, dtype=self.inv_freq.dtype)
        freqs = torch.outer(t, self.inv_freq)
        emb = torch.cat([freqs, freqs], dim=-1)

        return emb.cos(), emb.sin()


def apply_rotary_pos_emb(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """Apply rotary embeddings.

    Args:
        x: (batch, seq_len, num_heads, head_dim)
        cos: (seq_len, head_dim)
        sin: (seq_len, head_dim)

    Returns:
        Rotated tensor
    """
    # Reshape for broadcasting
    x1 = x[..., :x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2:]

    cos = cos.unsqueeze(0).unsqueeze(0)
    sin = sin.unsqueeze(0).unsqueeze(0)

    return torch.cat([
        x1 * cos - x2 * sin,
        x1 * sin + x2 * cos
    ], dim=-1)


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization."""

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        norm = torch.norm(x, p=2, dim=-1, keepdim=True)
        x = x / (norm + self.eps)
        return x * self.weight


class FlashAttention(nn.Module):
    """Multi-head attention with Flash Attention."""

    def __init__(self, config: TalkerConfig):
        super().__init__()
        self.num_heads = config.num_attention_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.head_dim = config.head_dim

        self.q_proj = nn.Linear(config.hidden_size, config.num_attention_heads * config.head_dim)
        self.k_proj = nn.Linear(config.hidden_size, config.num_key_value_heads * config.head_dim)
        self.v_proj = nn.Linear(config.hidden_size, config.num_key_value_heads * config.head_dim)
        self.o_proj = nn.Linear(config.num_attention_heads * config.head_dim, config.hidden_size)

        self.rope = RoPEEmbedding(config.head_dim, config.max_position_embeddings, config.rope_theta)

    def forward(
        self,
        x: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Forward pass.

        Args:
            x: (batch, seq_len, hidden_size)
            attention_mask: (batch, seq_len, seq_len) optional

        Returns:
            (batch, seq_len, hidden_size)
        """
        batch_size, seq_len, _ = x.shape

        # Project
        q = self.q_proj(x).reshape(batch_size, seq_len, self.num_heads, self.head_dim)
        k = self.k_proj(x).reshape(batch_size, seq_len, self.num_key_value_heads, self.head_dim)
        v = self.v_proj(x).reshape(batch_size, seq_len, self.num_key_value_heads, self.head_dim)

        # Apply RoPE
        cos, sin = self.rope(x, seq_len)
        q = apply_rotary_pos_emb(q, cos, sin)
        k = apply_rotary_pos_emb(k, cos, sin)

        # Grouped query attention (if num_key_value_heads < num_heads)
        if self.num_key_value_heads < self.num_heads:
            repeat_factor = self.num_heads // self.num_key_value_heads
            k = k.repeat_interleave(repeat_factor, dim=2)
            v = v.repeat_interleave(repeat_factor, dim=2)

        # Scaled dot-product attention
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)

        if attention_mask is not None:
            scores = scores + attention_mask.unsqueeze(1).unsqueeze(1) * -1e9

        attn_weights = F.softmax(scores, dim=-1)
        attn_output = torch.matmul(attn_weights, v)

        # Merge heads
        attn_output = attn_output.reshape(batch_size, seq_len, -1)
        output = self.o_proj(attn_output)

        return output


class MLP(nn.Module):
    """Feed-forward MLP."""

    def __init__(self, config: TalkerConfig):
        super().__init__()
        self.gate_proj = nn.Linear(config.hidden_size, config.intermediate_size)
        self.up_proj = nn.Linear(config.hidden_size, config.intermediate_size)
        self.down_proj = nn.Linear(config.intermediate_size, config.hidden_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class TransformerBlock(nn.Module):
    """Transformer decoder block."""

    def __init__(self, config: TalkerConfig):
        super().__init__()
        self.self_attn = FlashAttention(config)
        self.mlp = MLP(config)
        self.input_layernorm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, config.rms_norm_eps)

    def forward(
        self,
        x: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Forward pass."""
        # Self-attention with pre-norm
        h = x + self.self_attn(self.input_layernorm(x), attention_mask)

        # MLP with pre-norm
        h = h + self.mlp(self.post_attention_layernorm(h))

        return h


class TalkerLM(nn.Module):
    """Talker Language Model for speech code prediction."""

    def __init__(self, config: TalkerConfig):
        super().__init__()
        self.config = config

        # Embeddings
        self.text_embedding = nn.Embedding(config.vocab_size, config.hidden_size)
        self.codec_embeddings = nn.ModuleList([
            nn.Embedding(config.codebook_size, config.hidden_size)
            for _ in range(config.num_codebooks)
        ])
        self.speaker_linear = nn.Linear(config.speaker_embedding_dim, config.hidden_size)
        self.position_embedding = nn.Embedding(config.max_position_embeddings, config.hidden_size)

        # Transformer blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(config) for _ in range(config.num_hidden_layers)
        ])

        # Output layer norm
        self.norm = RMSNorm(config.hidden_size, config.rms_norm_eps)

        # Prediction heads (one per codebook)
        self.codec_heads = nn.ModuleList([
            nn.Linear(config.hidden_size, config.codebook_size)
            for _ in range(config.num_codebooks)
        ])

    def forward(
        self,
        input_ids: torch.Tensor,
        codec_ids: torch.Tensor,
        speaker_embeddings: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass.

        Args:
            input_ids: (batch, seq_len) text token IDs
            codec_ids: (batch, seq_len, num_codebooks) audio codec IDs
            speaker_embeddings: (batch, speaker_dim) speaker embeddings
            attention_mask: (batch, seq_len) optional

        Returns:
            Tuple of (codec_logits, codec_embeddings)
                codec_logits: (batch, seq_len, num_codebooks, codebook_size)
                codec_embeddings: (batch, seq_len, num_codebooks, hidden_size)
        """
        batch_size, seq_len = input_ids.shape

        # Embed inputs
        x = self.text_embedding(input_ids)

        # Add codec embeddings
        for i in range(self.config.num_codebooks):
            x = x + self.codec_embeddings[i](codec_ids[:, :, i])

        # Add speaker embedding
        speaker_emb = self.speaker_linear(speaker_embeddings).unsqueeze(1)
        x = x + speaker_emb

        # Add position embeddings
        positions = torch.arange(seq_len, device=x.device)
        x = x + self.position_embedding(positions).unsqueeze(0)

        # Create causal attention mask
        if attention_mask is None:
            attention_mask = torch.ones(batch_size, seq_len, device=x.device)

        # Create causal mask
        causal_mask = torch.triu(torch.ones(seq_len, seq_len, device=x.device) * -1e9, diagonal=1)

        # Apply transformer blocks
        for block in self.blocks:
            x = block(x, causal_mask.unsqueeze(0).unsqueeze(0))

        # Apply final norm
        x = self.norm(x)

        # Predict codecs
        codec_logits = []
        for head in self.codec_heads:
            codec_logits.append(head(x))  # (batch, seq_len, codebook_size)

        codec_logits = torch.stack(codec_logits, dim=2)  # (batch, seq_len, num_codebooks, codebook_size)

        return codec_logits, x


class TalkerLMWithLoss(nn.Module):
    """Talker LM with loss computation."""

    def __init__(self, config: TalkerConfig):
        super().__init__()
        self.model = TalkerLM(config)
        self.config = config

    def forward(
        self,
        input_ids: torch.Tensor,
        codec_ids: torch.Tensor,
        speaker_embeddings: torch.Tensor,
        codec_0_labels: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        return_loss: bool = True,
    ) -> dict:
        """Forward pass with loss.

        Args:
            input_ids: (batch, seq_len) text token IDs
            codec_ids: (batch, seq_len, num_codebooks) audio codec IDs
            speaker_embeddings: (batch, speaker_dim) speaker embeddings
            codec_0_labels: (batch, seq_len) labels for main codec prediction
            attention_mask: (batch, seq_len) optional
            return_loss: Whether to compute and return loss

        Returns:
            Dictionary with 'logits' and optionally 'loss'
        """
        codec_logits, hidden_states = self.model(
            input_ids,
            codec_ids,
            speaker_embeddings,
            attention_mask,
        )

        output = {'logits': codec_logits, 'hidden_states': hidden_states}

        if return_loss and codec_0_labels is not None:
            # Main codec loss (codec 0)
            codec_0_logits = codec_logits[:, :, 0, :]  # (batch, seq_len, codebook_size)
            loss = F.cross_entropy(
                codec_0_logits.reshape(-1, self.config.codebook_size),
                codec_0_labels.reshape(-1),
                ignore_index=-100,
            )
            output['loss'] = loss

        return output

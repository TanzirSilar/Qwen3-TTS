# coding=utf-8
"""Build / train / export the real Qwen3-TTS 12Hz tokenizer (Qwen3TTSTokenizerV2Model) from scratch.

Training graph (mirrors the inference path in modeling_qwen3_tts_tokenizer_v2.py exactly):

    wav (B,1,N) --Mimi encoder convs--> (B,512,2N/1920) --Mimi encoder_transformer-->
        --downsample(stride 2)--> latent (B,512,N/1920)                       [12.5 Hz]
    latent --TrainableSplitRVQ (decoder.quantizer tensors)--> quantized (B,512,T), codes (B,16,T)
    quantized --decoder.pre_conv / pre_transformer / upsample / decoder blocks--> wav_hat (B,1,N)

At export time the trained codebooks/projections are copied into the Mimi encoder's quantizer
so ``model.encode`` produces the same codes as training, and the model is saved in HF format.
"""

import json
import os
import shutil
import sys
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from qwen_tts.core.tokenizer_12hz.configuration_qwen3_tts_tokenizer_v2 import Qwen3TTSTokenizerV2Config  # noqa: E402
from qwen_tts.core.tokenizer_12hz.modeling_qwen3_tts_tokenizer_v2 import (  # noqa: E402
    Qwen3TTSTokenizerV2CausalConvNet,
    Qwen3TTSTokenizerV2CausalTransConvNet,
    Qwen3TTSTokenizerV2Model,
)

from quantizer import TrainableSplitRVQ  # noqa: E402

HERE = Path(__file__).resolve().parent
DEFAULT_ARCH = HERE / "reference" / "tokenizer_12hz_config.json"
DEFAULT_PREPROC = HERE / "reference" / "preprocessor_config.json"


def _deep_update(d: dict, u: dict) -> dict:
    for k, v in u.items():
        if isinstance(v, dict) and isinstance(d.get(k), dict):
            _deep_update(d[k], v)
        else:
            d[k] = v
    return d


def build_config(arch_json: Optional[str] = None, overrides: Optional[dict] = None) -> Qwen3TTSTokenizerV2Config:
    with open(arch_json or DEFAULT_ARCH) as f:
        cfg = json.load(f)
    for k in ("architectures", "transformers_version"):
        cfg.pop(k, None)
    if overrides:
        _deep_update(cfg, overrides)
    return Qwen3TTSTokenizerV2Config(**cfg)


@torch.no_grad()
def _init_decoder_weights(decoder: nn.Module, std: float = 0.02) -> None:
    """The released decoder classes define no _init_weights; give convs/linears a sane init."""
    for m in decoder.modules():
        if isinstance(m, (nn.Conv1d, nn.ConvTranspose1d)):
            nn.init.trunc_normal_(m.weight, std=std)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, std=std)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.LayerNorm):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)
    # Codebooks start empty (zeros); k-means initialises them on the first training step.


def build_tokenizer(arch_json: Optional[str] = None, overrides: Optional[dict] = None) -> Qwen3TTSTokenizerV2Model:
    config = build_config(arch_json, overrides)
    enc_cfg = config.encoder_config
    if config.encoder_valid_num_quantizers != config.decoder_config.num_quantizers:
        raise ValueError("encoder_valid_num_quantizers must equal decoder num_quantizers")
    if enc_cfg.vector_quantization_hidden_dimension != config.decoder_config.codebook_dim // 2:
        raise ValueError("encoder VQ dim must equal decoder codebook_dim // 2")
    model = Qwen3TTSTokenizerV2Model(config)
    _init_decoder_weights(model.decoder)
    return model


class TokenizerGenerator(nn.Module):
    """Everything that receives generator gradients, wrapped for DDP.

    Only modules actually used in the training graph are registered here (so DDP does not need
    find_unused_parameters): Mimi encoder/encoder_transformer/downsample, the decoder, and an
    optional linear head for semantic distillation.
    """

    def __init__(self, tokenizer: Qwen3TTSTokenizerV2Model, rvq_kwargs: dict, teacher_dim: int = 0):
        super().__init__()
        enc = tokenizer.encoder
        self.enc_convs = enc.encoder
        self.enc_transformer = enc.encoder_transformer
        self.enc_downsample = enc.downsample
        self.decoder = tokenizer.decoder
        self.rvq = TrainableSplitRVQ(tokenizer.decoder.quantizer, **rvq_kwargs)
        sem_dim = tokenizer.config.decoder_config.codebook_dim // 2
        self.semantic_head = nn.Linear(sem_dim, teacher_dim) if teacher_dim > 0 else None

    def encode_latent(self, wav: torch.Tensor) -> torch.Tensor:
        h = self.enc_convs(wav)  # (B, 512, 25 Hz)
        h = self.enc_transformer(h.transpose(1, 2), return_dict=True).last_hidden_state.transpose(1, 2)
        return self.enc_downsample(h)  # (B, 512, 12.5 Hz)

    def decode_latent(self, quantized: torch.Tensor) -> torch.Tensor:
        """Same as Qwen3TTSTokenizerV2Decoder.forward after quantizer.decode, minus the final clamp."""
        dec = self.decoder
        h = dec.pre_conv(quantized).transpose(1, 2)
        h = dec.pre_transformer(inputs_embeds=h).last_hidden_state.permute(0, 2, 1)
        for blocks in dec.upsample:
            for block in blocks:
                h = block(h)
        for block in dec.decoder:
            h = block(h)
        return h

    def forward(self, wav: torch.Tensor, frame_mask: Optional[torch.Tensor] = None):
        """wav: (B, 1, N) with N a multiple of 1920."""
        latent = self.encode_latent(wav)
        if frame_mask is not None:
            frame_mask = frame_mask[:, : latent.shape[-1]]
        q = self.rvq(latent, frame_mask)
        wav_hat = self.decode_latent(q["quantized"].to(latent.dtype))
        wav_hat = wav_hat[..., : wav.shape[-1]]
        out = {"wav_hat": wav_hat, **q}
        if self.semantic_head is not None:
            out["semantic_pred"] = self.semantic_head(q["semantic_q"].transpose(1, 2))  # (B, T, teacher_dim)
        return out


@torch.no_grad()
def sync_encoder_quantizer(tokenizer: Qwen3TTSTokenizerV2Model) -> None:
    """Copy trained codebooks/projections from decoder.quantizer into the Mimi encoder quantizer."""
    dq = tokenizer.decoder.quantizer
    eq = tokenizer.encoder.quantizer
    pairs = [
        (dq.rvq_first, eq.semantic_residual_vector_quantizer),
        (dq.rvq_rest, eq.acoustic_residual_vector_quantizer),
    ]
    for src, dst in pairs:
        dst.input_proj.weight.copy_(src.input_proj.weight)
        dst.output_proj.weight.copy_(src.output_proj.weight)
        for i, s_layer in enumerate(src.vq.layers):
            d_cb = dst.layers[i].codebook
            d_cb.embed_sum.copy_(s_layer._codebook.embedding_sum)
            d_cb.cluster_usage.copy_(s_layer._codebook.cluster_usage)
            d_cb.initialized.fill_(1.0)
            d_cb._embed = None  # drop cached codebook
        # Mimi has extra acoustic layers (32 total) beyond encoder_valid_num_quantizers; they are
        # never used (codes are truncated to 16) so they stay at their zero init.


def export_hf(tokenizer: Qwen3TTSTokenizerV2Model, out_dir: str, preproc_json: Optional[str] = None) -> None:
    sync_encoder_quantizer(tokenizer)
    os.makedirs(out_dir, exist_ok=True)
    tokenizer.save_pretrained(out_dir, safe_serialization=True)
    shutil.copy(preproc_json or DEFAULT_PREPROC, os.path.join(out_dir, "preprocessor_config.json"))

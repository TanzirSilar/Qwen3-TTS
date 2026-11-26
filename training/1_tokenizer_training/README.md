# Phase 1: Speech Tokenizer (Qwen3-TTS-Tokenizer-12Hz) from scratch

Trains the **real** `Qwen3TTSTokenizerV2Model` from `qwen_tts` with random initialisation, using the
exact architecture of the released `Qwen/Qwen3-TTS-Tokenizer-12Hz` (`reference/tokenizer_12hz_config.json`).
The result loads with `Qwen3TTSTokenizer.from_pretrained(...)` and plugs into the rest of `qwen_tts`.

## Architecture

```
24 kHz wav ──Mimi SEANet encoder (causal convs, ×960)──► 25 Hz ──Mimi transformer (8L)──► downsample ×2
          ──► latent (512-d, 12.5 Hz)
latent ──┬─ input_proj 512→256 ─ RVQ  1 codebook  (semantic, distilled from an SSL teacher)
         └─ input_proj 512→256 ─ RVQ 15 codebooks (acoustic)        each codebook: 2048 × 256
       (both branches see the same latent; output_proj 256→512 outputs are summed)
quantized ──pre_conv → 8L sliding-window transformer → 2× ConvNeXt upsample (×4)
          → SnakeBeta conv decoder (×8×5×4×3) ──► 24 kHz wav          (1920 samples per frame)
```

## How training works (`train_tokenizer.py`)

| Piece | File | Notes |
|---|---|---|
| Model build / init / export | `model.py` | random init of the released classes; decoder convs get trunc-normal init |
| Quantizer training | `quantizer.py` | EMA codebooks written straight into the released `embedding_sum` / `cluster_usage` tensors, straight-through estimator, commitment loss, k-means init, dead-code re-seeding, per-sample quantizer dropout (0–15 acoustic codebooks), EMA stats all-reduced across GPUs |
| Discriminators | `discriminators.py` | Multi-period (2,3,5,7,11) + multi-scale STFT (2048…128) |
| Losses | `losses.py` | multi-resolution log-mel L1, hinge GAN, relative feature matching, semantic distillation (cosine) |
| Data | `data_loader.py` | JSONL manifests from `0_data_preparation/prepare_tokenizer_data.py`; random 3.84 s crops, resample to 24 kHz, peak-norm + random gain |

Schedule: reconstruction-only for `disc_start_step` steps, then GAN. AdamW + warmup + cosine.
Codebooks only become "real" after the first step (k-means on the first batch).

**Export.** Training uses the decoder's quantizer as the single copy of the codebooks. Every
`save_steps` they are copied into the Mimi encoder quantizer and the whole model is saved to
`<output_dir>/hf_latest/` (HF format + `preprocessor_config.json`). `verify_export.py` checks
that the exported model's `encode`/`decode` match the training graph.

## Run

```bash
# smoke test on a compute node (synthetic audio, ~5 min)
srun --jobid=<JOBID> --overlap bash training/1_tokenizer_training/smoke_test.sh

# real training, 8 GPUs
cd training/1_tokenizer_training
torchrun --standalone --nproc_per_node=8 train_tokenizer.py \
  --config ../../configs/tokenizer.yaml \
  --train_manifest <data>/manifests/train_manifest.jsonl \
  --val_manifest   <data>/manifests/val_manifest.jsonl \
  --output_dir ../../checkpoints/tokenizer
```

Training auto-resumes from the newest `ckpt_step*.pt` in `--output_dir`.
The teacher (`losses.semantic_distill.teacher`, default `microsoft/wavlm-large`) is downloaded
from the HF Hub on first use; set the proxy (`http_proxy`/`https_proxy`) on compute nodes.

## What to watch

- `mel` should fall steadily; `val_mel` is the main quality number.
- `sem_ppl` / `ac_ppl_mean` (codebook perplexity, max 2048) and `val_cb_usage_*` (fraction of codes
  used): collapse to small values means dead codebooks.
- `adv_g`, `loss_d` should hover rather than one side winning (`loss_d → 0` means D dominates).

## Not reproduced exactly

The Qwen3-TTS report does not publish the full tokenizer recipe. Loss weights, the teacher model,
the discriminator set and schedule here follow Mimi / DAC / EnCodec practice and are configurable
in `configs/tokenizer.yaml`. For non-English data, consider a multilingual teacher.

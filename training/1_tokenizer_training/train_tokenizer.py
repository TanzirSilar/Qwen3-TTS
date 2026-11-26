# coding=utf-8
"""Phase 1: train the Qwen3-TTS 12Hz speech tokenizer (Qwen3TTSTokenizerV2Model) from scratch.

Launch (single node, 8 GPUs):
    torchrun --standalone --nproc_per_node=8 train_tokenizer.py \
        --config ../../configs/tokenizer.yaml \
        --train_manifest /path/train_manifest.jsonl --val_manifest /path/val_manifest.jsonl \
        --output_dir ../../checkpoints/tokenizer

Smoke test without data:  add --synthetic --max_steps 50
"""

import argparse
import glob
import json
import logging
import math
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
import torchaudio.functional as AF
import yaml
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler

sys.path.insert(0, str(Path(__file__).resolve().parent))
from data_loader import CodecAudioDataset, SyntheticAudioDataset, collate  # noqa: E402
from discriminators import CombinedDiscriminator  # noqa: E402
from losses import (  # noqa: E402
    MultiResolutionMelLoss,
    discriminator_hinge_loss,
    feature_matching_loss,
    generator_hinge_loss,
    semantic_distill_loss,
)
from model import TokenizerGenerator, build_tokenizer, export_hf  # noqa: E402

logger = logging.getLogger("train_tokenizer")


# ---------------------------------------------------------------------------- utils
def setup_dist():
    if "RANK" in os.environ:
        dist.init_process_group("nccl")
        rank, world = dist.get_rank(), dist.get_world_size()
        local = int(os.environ.get("LOCAL_RANK", 0))
    else:
        rank, world, local = 0, 1, 0
    torch.cuda.set_device(local)
    return rank, world, torch.device("cuda", local)


def is_main():
    return not dist.is_initialized() or dist.get_rank() == 0


def barrier():
    if dist.is_initialized():
        dist.barrier()


def lr_lambda(warmup: int, total: int, min_ratio: float):
    def f(step):
        if step < warmup:
            return (step + 1) / max(1, warmup)
        p = min(1.0, (step - warmup) / max(1, total - warmup))
        return min_ratio + (1 - min_ratio) * 0.5 * (1 + math.cos(math.pi * p))
    return f


class Teacher(torch.nn.Module):
    """Frozen SSL model whose features the first (semantic) codebook is distilled toward."""

    def __init__(self, name: str, layer: int, device):
        super().__init__()
        from transformers import AutoModel
        self.model = AutoModel.from_pretrained(name, torch_dtype=torch.float32).to(device).eval()
        self.model.requires_grad_(False)
        self.layer = layer
        self.dim = self.model.config.hidden_size

    @torch.no_grad()
    def forward(self, wav24: torch.Tensor, n_frames: int) -> torch.Tensor:
        x = AF.resample(wav24.squeeze(1).float(), 24000, 16000)
        x = (x - x.mean(-1, keepdim=True)) / (x.std(-1, keepdim=True) + 1e-5)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            hs = self.model(x, output_hidden_states=True).hidden_states[self.layer]  # (B, T50, D)
        hs = F.adaptive_avg_pool1d(hs.float().transpose(1, 2), n_frames).transpose(1, 2)
        return hs  # (B, n_frames, D)


def infinite(loader, sampler):
    epoch = 0
    while True:
        if sampler is not None:
            sampler.set_epoch(epoch)
        for b in loader:
            yield b
        epoch += 1


# ---------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(Path(__file__).resolve().parents[2] / "configs" / "tokenizer.yaml"))
    ap.add_argument("--train_manifest")
    ap.add_argument("--val_manifest")
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--synthetic", action="store_true", help="use synthetic audio (smoke test)")
    ap.add_argument("--max_steps", type=int)
    ap.add_argument("--batch_size", type=int)
    ap.add_argument("--no_teacher", action="store_true")
    args = ap.parse_args()

    with open(args.config) as f:
        C = yaml.safe_load(f)
    T, L, D_ = C["training"], C["losses"], C["data"]
    if args.max_steps:
        T["max_steps"] = args.max_steps
    if args.batch_size:
        T["batch_size"] = args.batch_size

    rank, world, device = setup_dist()
    logging.basicConfig(level=logging.INFO if is_main() else logging.WARNING,
                        format=f"%(asctime)s [rank{rank}] %(levelname)s %(message)s")
    seed = C.get("seed", 42) + rank
    random.seed(seed); np.random.seed(seed); torch.manual_seed(C.get("seed", 42))  # same init on all ranks
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    out_dir = Path(args.output_dir)
    if is_main():
        out_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------ model
    tokenizer = build_tokenizer(C["model"].get("arch_json"), C["model"].get("overrides"))
    teacher = None
    sem = L.get("semantic_distill", {})
    if sem.get("enabled", False) and not args.no_teacher:
        teacher = Teacher(sem["teacher"], sem.get("layer", -1), device)
    G = TokenizerGenerator(tokenizer, C["quantizer"], teacher.dim if teacher else 0).to(device)
    Dsc = CombinedDiscriminator(tuple(C["discriminator"]["periods"]), tuple(C["discriminator"]["stft_n_ffts"])).to(device)
    mel_loss = MultiResolutionMelLoss(C["audio"]["sample_rate"]).to(device)
    torch.manual_seed(seed)  # different data aug / dropout per rank from here on

    n_g = sum(p.numel() for p in G.parameters() if p.requires_grad)
    n_d = sum(p.numel() for p in Dsc.parameters())
    logger.info(f"generator trainable params: {n_g/1e6:.1f}M, discriminator: {n_d/1e6:.1f}M, world={world}")

    g_params = [p for p in G.parameters() if p.requires_grad]
    opt_g = torch.optim.AdamW(g_params, lr=T["lr_g"], betas=tuple(T["betas"]), weight_decay=T["weight_decay"])
    opt_d = torch.optim.AdamW(Dsc.parameters(), lr=T["lr_d"], betas=tuple(T["betas"]), weight_decay=T["weight_decay"])
    lam = lr_lambda(T["warmup_steps"], T["max_steps"], T.get("min_lr_ratio", 0.1))
    sch_g = torch.optim.lr_scheduler.LambdaLR(opt_g, lam)
    sch_d = torch.optim.lr_scheduler.LambdaLR(opt_d, lam)

    # ------------------------------------------------------------------ resume
    step = 0
    ckpts = sorted(glob.glob(str(out_dir / "ckpt_step*.pt")), key=lambda p: int(Path(p).stem[9:]))
    resume = T.get("resume_from") or (ckpts[-1] if ckpts and T.get("auto_resume", True) else None)
    if resume:
        sd = torch.load(resume, map_location="cpu", weights_only=False)
        G.load_state_dict(sd["G"]); Dsc.load_state_dict(sd["D"])
        opt_g.load_state_dict(sd["opt_g"]); opt_d.load_state_dict(sd["opt_d"])
        sch_g.load_state_dict(sd["sch_g"]); sch_d.load_state_dict(sd["sch_d"])
        step = sd["step"]
        logger.info(f"resumed from {resume} at step {step}")

    G_ddp = DDP(G, device_ids=[device.index], broadcast_buffers=False) if world > 1 else G
    D_ddp = DDP(Dsc, device_ids=[device.index], broadcast_buffers=False) if world > 1 else Dsc

    # ------------------------------------------------------------------ data
    seg = D_["segment_seconds"]
    if args.synthetic:
        train_ds = SyntheticAudioDataset(512, segment_seconds=seg)
        val_ds = SyntheticAudioDataset(16, segment_seconds=seg)
    else:
        kw = dict(sample_rate=C["audio"]["sample_rate"], segment_seconds=seg,
                  min_duration=D_["min_duration"], max_duration=D_["max_duration"])
        train_ds = CodecAudioDataset(args.train_manifest, **kw)
        val_ds = CodecAudioDataset(args.val_manifest, random_crop=False, segment_seconds=D_["val_segment_seconds"],
                                   sample_rate=C["audio"]["sample_rate"], min_duration=D_["min_duration"],
                                   max_duration=D_["max_duration"])
    sampler = DistributedSampler(train_ds, shuffle=True, drop_last=True) if world > 1 else None
    train_dl = DataLoader(train_ds, batch_size=T["batch_size"], sampler=sampler, shuffle=sampler is None,
                          num_workers=D_["num_workers"], collate_fn=collate, drop_last=True,
                          pin_memory=True, persistent_workers=D_["num_workers"] > 0)
    val_dl = DataLoader(val_ds, batch_size=4, shuffle=False, num_workers=2, collate_fn=collate)
    it = infinite(train_dl, sampler)

    tb = None
    if is_main() and C["logging"].get("tensorboard", True):
        try:
            from torch.utils.tensorboard import SummaryWriter
            tb = SummaryWriter(str(out_dir / "tb"))
        except Exception as ex:
            logger.warning(f"tensorboard unavailable ({ex}); logging to metrics.jsonl only")
    metrics_f = open(out_dir / "metrics.jsonl", "a") if is_main() else None

    # ------------------------------------------------------------------ loop
    amp = dict(device_type="cuda", dtype=torch.bfloat16, enabled=C["hardware"]["mixed_precision"] == "bf16")
    disc_start = T["disc_start_step"]
    w = L
    t0 = time.time()
    G_ddp.train(); D_ddp.train()
    while step < T["max_steps"]:
        batch = next(it)
        wav = batch["wav"].to(device, non_blocking=True)
        fmask = batch["frame_mask"].to(device, non_blocking=True)
        use_gan = step >= disc_start

        # ---------------- generator
        with torch.autocast(**amp):
            out = G_ddp(wav, fmask)
            wav_hat = out["wav_hat"].float()
            l_mel = mel_loss(wav_hat, wav)
            l_commit = out["commit_loss"]
            loss_g = w["mel_weight"] * l_mel + w["commit_weight"] * l_commit
            logs = {"mel": l_mel.item(), "commit": l_commit.item()}
            if teacher is not None:
                tgt = teacher(wav, out["semantic_pred"].shape[1])
                l_sem = semantic_distill_loss(out["semantic_pred"], tgt, fmask[:, : tgt.shape[1]])
                loss_g = loss_g + sem["weight"] * l_sem
                logs["sem_distill"] = l_sem.item()
            if use_gan:
                Dsc.requires_grad_(False)
                d_all = Dsc(torch.cat([wav, wav_hat], 0))  # unwrapped D: no DDP reduction for G pass
                B = wav.shape[0]
                d_real = [(lg[:B], [f[:B] for f in fs]) for lg, fs in d_all]
                d_fake = [(lg[B:], [f[B:] for f in fs]) for lg, fs in d_all]
                l_adv = generator_hinge_loss(d_fake)
                l_fm = feature_matching_loss(d_real, d_fake)
                loss_g = loss_g + w["adv_weight"] * l_adv + w["fm_weight"] * l_fm
                logs.update(adv_g=l_adv.item(), fm=l_fm.item())
                Dsc.requires_grad_(True)
        opt_g.zero_grad(set_to_none=True)
        loss_g.backward()
        gn_g = torch.nn.utils.clip_grad_norm_(g_params, T["max_grad_norm"])
        opt_g.step(); sch_g.step()
        logs["loss_g"] = loss_g.item(); logs["grad_norm_g"] = gn_g.item()

        # ---------------- discriminator
        if use_gan:
            with torch.autocast(**amp):
                d_all = D_ddp(torch.cat([wav, wav_hat.detach()], 0))
                B = wav.shape[0]
                l_d = discriminator_hinge_loss([(lg[:B], None) for lg, _ in d_all],
                                               [(lg[B:], None) for lg, _ in d_all])
            opt_d.zero_grad(set_to_none=True)
            l_d.backward()
            gn_d = torch.nn.utils.clip_grad_norm_(Dsc.parameters(), T["max_grad_norm"])
            opt_d.step()
            logs.update(loss_d=l_d.item(), grad_norm_d=gn_d.item())
        sch_d.step()
        step += 1

        # ---------------- logging
        if is_main() and step % C["logging"]["log_every"] == 0:
            st = out["stats"]
            sem_ppl = st.get("sem_perplexity_0", 0.0)
            ac_ppl = np.mean([v for k, v in st.items() if k.startswith("ac_")]) if st else 0.0
            logs.update(sem_ppl=sem_ppl, ac_ppl_mean=float(ac_ppl), lr=sch_g.get_last_lr()[0],
                        mem_gb=torch.cuda.max_memory_allocated() / 2**30)
            el = time.time() - t0
            logger.info(f"step {step} | " + " ".join(f"{k}={v:.4g}" for k, v in logs.items())
                        + f" | {el / C['logging']['log_every']:.2f}s/it")
            t0 = time.time()
            metrics_f.write(json.dumps({"step": step, **logs}) + "\n"); metrics_f.flush()
            if tb:
                for k, v in logs.items():
                    tb.add_scalar(f"train/{k}", v, step)

        # ---------------- validation
        if step % T["eval_steps"] == 0 or step == T["max_steps"]:
            if is_main():
                validate(G, val_dl, mel_loss, device, amp, step, tb, metrics_f, T.get("num_val_batches", 50))
            barrier()
            G_ddp.train()

        # ---------------- checkpoint / export
        if step % T["save_steps"] == 0 or step == T["max_steps"]:
            if is_main():
                path = out_dir / f"ckpt_step{step}.pt"
                torch.save({"G": G.state_dict(), "D": Dsc.state_dict(), "opt_g": opt_g.state_dict(),
                            "opt_d": opt_d.state_dict(), "sch_g": sch_g.state_dict(),
                            "sch_d": sch_d.state_dict(), "step": step, "config": C}, path)
                old = sorted(glob.glob(str(out_dir / "ckpt_step*.pt")), key=lambda p: int(Path(p).stem[9:]))
                for p in old[: -T["keep_checkpoints"]]:
                    os.remove(p)
                export_hf(tokenizer, str(out_dir / "hf_latest"))
                logger.info(f"saved {path} and exported HF model to {out_dir/'hf_latest'}")
            barrier()

    if is_main():
        export_hf(tokenizer, str(out_dir / "hf_final"))
        logger.info(f"done; final HF tokenizer at {out_dir/'hf_final'}")
    if dist.is_initialized():
        dist.destroy_process_group()


@torch.no_grad()
def validate(G, val_dl, mel_loss, device, amp, step, tb, metrics_f, num_batches):
    G.eval()
    mels, used = [], None
    first = None
    for i, batch in enumerate(val_dl):
        if i >= num_batches:
            break
        wav = batch["wav"].to(device)
        fmask = batch["frame_mask"].to(device)
        with torch.autocast(**amp):
            out = G(wav, fmask)
        wav_hat = out["wav_hat"].float().clamp(-1, 1)
        mels.append(mel_loss(wav_hat, wav).item())
        codes = out["codes"]  # (B, 16, T)
        m = fmask[:, : codes.shape[-1]]
        if used is None:
            used = [set() for _ in range(codes.shape[1])]
        for q in range(codes.shape[1]):
            used[q].update(codes[:, q][m].unique().tolist())
        if first is None:
            first = (wav[0, 0].cpu(), wav_hat[0, 0].cpu())
    usage = [len(u) / 2048 for u in used] if used else [0]
    res = {"val_mel": float(np.mean(mels)) if mels else float("nan"),
           "val_cb_usage_sem": usage[0], "val_cb_usage_ac_mean": float(np.mean(usage[1:])) if len(usage) > 1 else 0}
    logger.info(f"[val] step {step} | " + " ".join(f"{k}={v:.4f}" for k, v in res.items()))
    metrics_f.write(json.dumps({"step": step, **res}) + "\n"); metrics_f.flush()
    if tb:
        for k, v in res.items():
            tb.add_scalar(f"val/{k}", v, step)
        if first is not None:
            tb.add_audio("val/ref", first[0], step, 24000)
            tb.add_audio("val/recon", first[1], step, 24000)


if __name__ == "__main__":
    main()

# coding=utf-8
"""Training-time residual vector quantization for the Qwen3-TTS 12Hz tokenizer.

The released ``Qwen3TTSTokenizerV2Decoder.quantizer`` (a ``SplitResidualVectorQuantizer``)
only implements ``decode``.  Its codebooks are stored Mimi-style as two tensors per layer:

    embedding_sum : (codebook_size, dim)   EMA of the sum of assigned vectors
    cluster_usage : (codebook_size,)       EMA of the assignment counts
    codebook      = embedding_sum / cluster_usage

``TrainableSplitRVQ`` wraps that module and adds the training side (nearest-neighbour
assignment, straight-through estimator, commitment loss, EMA updates, k-means init,
dead-code replacement, per-sample quantizer dropout, multi-GPU synchronisation).
It updates the wrapped module's tensors *in place*, so after training the decoder
quantizer already holds the final codebooks in exactly the format inference expects.

Layout (identical in encoder and decoder, see the released config):
    latent (B, 512, T)
      ├─ rvq_first: input_proj 512->256, 1 codebook   (semantic)
      └─ rvq_rest : input_proj 512->256, 15 codebooks (acoustic)
    Both branches quantize the SAME latent (not a residual of each other);
    their output_proj(256->512) results are summed.
"""

from typing import Dict, List, Optional, Tuple

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F


def _dist_on() -> bool:
    return dist.is_available() and dist.is_initialized()


def _all_reduce_sum(t: torch.Tensor) -> torch.Tensor:
    if _dist_on():
        dist.all_reduce(t, op=dist.ReduceOp.SUM)
    return t


def _broadcast(t: torch.Tensor, src: int = 0) -> torch.Tensor:
    if _dist_on():
        dist.broadcast(t, src=src)
    return t


def _all_gather_cat(x: torch.Tensor) -> torch.Tensor:
    """All-gather variable-length (N_i, D) tensors and concatenate."""
    if not _dist_on():
        return x
    world = dist.get_world_size()
    n = torch.tensor([x.shape[0]], device=x.device, dtype=torch.long)
    sizes = [torch.zeros_like(n) for _ in range(world)]
    dist.all_gather(sizes, n)
    max_n = int(max(s.item() for s in sizes))
    padded = torch.zeros(max_n, x.shape[1], device=x.device, dtype=x.dtype)
    padded[: x.shape[0]] = x
    out = [torch.zeros_like(padded) for _ in range(world)]
    dist.all_gather(out, padded)
    return torch.cat([o[: int(s.item())] for o, s in zip(out, sizes)], dim=0)


@torch.no_grad()
def _kmeans(x: torch.Tensor, k: int, iters: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """Plain k-means on (N, D) fp32 data. Returns (centroids (k, D), counts (k,))."""
    n = x.shape[0]
    if n >= k:
        centroids = x[torch.randperm(n, device=x.device)[:k]].clone()
    else:
        idx = torch.randint(0, n, (k,), device=x.device)
        centroids = x[idx] + 1e-3 * torch.randn(k, x.shape[1], device=x.device) * x.std()
    counts = torch.ones(k, device=x.device)
    for _ in range(iters):
        assign = torch.cdist(x, centroids).argmin(dim=1)
        counts = torch.bincount(assign, minlength=k).float()
        sums = torch.zeros_like(centroids).index_add_(0, assign, x)
        nonzero = counts > 0
        centroids[nonzero] = sums[nonzero] / counts[nonzero, None]
    return centroids, counts.clamp(min=1.0)


class _EMACodebook:
    """Training logic for one released ``EuclideanCodebook`` (holds no state of its own)."""

    def __init__(self, codebook: nn.Module, decay: float, threshold: float, eps: float = 1e-5):
        self.cb = codebook  # has .embedding_sum (K, D) and .cluster_usage (K,) nn.Parameters
        self.decay = decay
        self.threshold = threshold
        self.eps = eps

    @property
    def embed(self) -> torch.Tensor:
        return self.cb.embedding_sum / self.cb.cluster_usage.clamp(min=self.eps)[:, None]

    @torch.no_grad()
    def init_kmeans(self, x: torch.Tensor, iters: int) -> None:
        x = _all_gather_cat(x.float())
        k = self.cb.cluster_usage.shape[0]
        centroids, counts = _kmeans(x, k, iters)
        centroids = _broadcast(centroids.contiguous())
        counts = _broadcast(counts.contiguous())
        # EMA state such that embed == centroids
        self.cb.cluster_usage.data.copy_(counts)
        self.cb.embedding_sum.data.copy_(centroids * counts[:, None])

    @torch.no_grad()
    def ema_update(self, x: torch.Tensor, codes: torch.Tensor) -> int:
        """x: (N, D) fp32 inputs, codes: (N,). Returns number of replaced dead codes."""
        k = self.cb.cluster_usage.shape[0]
        counts = torch.bincount(codes, minlength=k).float()
        sums = torch.zeros(k, x.shape[1], device=x.device).index_add_(0, codes, x)
        _all_reduce_sum(counts)
        _all_reduce_sum(sums)
        usage, esum = self.cb.cluster_usage.data, self.cb.embedding_sum.data
        usage.mul_(self.decay).add_(counts, alpha=1 - self.decay)
        esum.mul_(self.decay).add_(sums, alpha=1 - self.decay)

        # Dead-code replacement: re-seed rarely used codes with random batch vectors.
        # Decided on rank 0 and broadcast so every rank keeps identical codebooks.
        dead = (usage < self.threshold).float()
        dead = _broadcast(dead)
        n_dead = int(dead.sum().item())
        if n_dead > 0:
            pick = torch.randint(0, x.shape[0], (k,), device=x.device)
            new = x[pick]
            new = _broadcast(new.contiguous())
            m = dead.bool()
            esum[m] = new[m] * self.threshold
            usage[m] = self.threshold
        return n_dead


class TrainableSplitRVQ(nn.Module):
    """Training wrapper around the released ``SplitResidualVectorQuantizer``."""

    def __init__(
        self,
        split_rvq: nn.Module,
        decay: float = 0.99,
        dead_code_threshold: float = 2.0,
        kmeans_iters: int = 10,
        quantizer_dropout_prob: float = 0.5,
    ):
        super().__init__()
        self.rvq = split_rvq  # owns input_proj / output_proj / codebooks (registered as submodule)
        self.kmeans_iters = kmeans_iters
        self.quantizer_dropout_prob = quantizer_dropout_prob
        self.n_semantic = split_rvq.n_q_semantic
        self.n_acoustic = split_rvq.n_q_acoustic

        # Codebook tensors are updated by EMA, never by the optimizer.
        for layer in list(split_rvq.rvq_first.vq.layers) + list(split_rvq.rvq_rest.vq.layers):
            if not isinstance(layer.project_out, nn.Identity):
                raise ValueError("codebook_dim != quantizer dim is not supported (project_out must be Identity)")
            layer._codebook.embedding_sum.requires_grad_(False)
            layer._codebook.cluster_usage.requires_grad_(False)

        mk = lambda layers: [_EMACodebook(l._codebook, decay, dead_code_threshold) for l in layers]
        self._sem_cbs = mk(split_rvq.rvq_first.vq.layers)
        self._ac_cbs = mk(split_rvq.rvq_rest.vq.layers)
        # Persistent flag so resumed runs don't re-run k-means.
        self.register_buffer("initialized", torch.tensor(False))

    # ------------------------------------------------------------------ #
    def _rvq_forward(
        self,
        x: torch.Tensor,  # (B, D, T) fp32, already input-projected
        cbs: List[_EMACodebook],
        n_active: Optional[torch.Tensor],  # (B,) number of layers kept per sample, or None = all
        mask: Optional[torch.Tensor],  # (B, T) valid frames
        do_init: bool,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Dict[str, float]]:
        B, D, T = x.shape
        residual = x
        out = torch.zeros_like(x)
        codes_all, commit = [], x.new_zeros(())
        stats: Dict[str, float] = {}
        valid = mask.reshape(-1) if mask is not None else None

        for i, cb in enumerate(cbs):
            flat = residual.transpose(1, 2).reshape(-1, D)  # (B*T, D)
            if self.training and do_init:
                cb.init_kmeans(flat.detach() if valid is None else flat.detach()[valid], self.kmeans_iters)
            embed = cb.embed
            with torch.no_grad():
                codes = torch.cdist(flat.detach()[None], embed[None])[0].argmin(dim=-1)  # (B*T,)
            q = F.embedding(codes, embed).view(B, T, D).transpose(1, 2)  # (B, D, T)

            # commitment loss (encoder output pulled toward its codeword), masked
            se = (residual - q.detach()).pow(2).mean(dim=1)  # (B, T)
            keep = torch.ones(B, T, device=x.device)
            if n_active is not None:
                keep = keep * (i < n_active).float()[:, None]
            if mask is not None:
                keep = keep * mask.float()
            commit = commit + (se * keep).sum() / keep.sum().clamp(min=1.0)

            q_st = residual + (q - residual).detach()  # straight-through
            if n_active is not None:
                q_st = q_st * (i < n_active).to(q_st.dtype)[:, None, None]
            out = out + q_st

            if self.training:
                sel = valid if valid is not None else slice(None)
                cb.ema_update(flat.detach()[sel], codes[sel])
                with torch.no_grad():
                    c = codes[sel]
                    p = torch.bincount(c, minlength=embed.shape[0]).float()
                    p = p / p.sum().clamp(min=1.0)
                    stats[f"perplexity_{i}"] = float(torch.exp(-(p * torch.log(p + 1e-10)).sum()))

            residual = residual - q.detach()
            codes_all.append(codes.view(B, T))
        return out, torch.stack(codes_all, dim=1), commit, stats

    def forward(self, latent: torch.Tensor, mask: Optional[torch.Tensor] = None):
        """latent: (B, 512, T) encoder output.

        Returns dict with
            quantized      (B, 512, T)  -> decoder input (sum of both branches after output_proj)
            semantic_q     (B, 256, T)  first-codebook output before output_proj (for distillation)
            codes          (B, 16, T)
            commit_loss    scalar
            stats          dict of perplexities
        """
        with torch.autocast(device_type=latent.device.type, enabled=False):
            latent = latent.float()
            rf, rr = self.rvq.rvq_first, self.rvq.rvq_rest
            do_init = self.training and not bool(self.initialized)

            xs = rf.input_proj(latent)
            q_sem, c_sem, l_sem, st_sem = self._rvq_forward(xs, self._sem_cbs, None, mask, do_init)

            n_active = None
            if self.training and self.quantizer_dropout_prob > 0:
                B = latent.shape[0]
                n_active = torch.full((B,), self.n_acoustic, device=latent.device)
                drop = torch.rand(B, device=latent.device) < self.quantizer_dropout_prob
                rand_n = torch.randint(0, self.n_acoustic + 1, (B,), device=latent.device)
                n_active = torch.where(drop, rand_n, n_active)

            xa = rr.input_proj(latent)
            q_ac, c_ac, l_ac, st_ac = self._rvq_forward(xa, self._ac_cbs, n_active, mask, do_init)

            if do_init:
                self.initialized.fill_(True)

            quantized = rf.output_proj(q_sem) + rr.output_proj(q_ac)

        stats = {"sem_" + k: v for k, v in st_sem.items()}
        stats.update({"ac_" + k: v for k, v in st_ac.items()})
        return {
            "quantized": quantized,
            "semantic_q": q_sem,
            "codes": torch.cat([c_sem, c_ac], dim=1),
            "commit_loss": (l_sem + l_ac) / (self.n_semantic + self.n_acoustic),
            "stats": stats,
        }

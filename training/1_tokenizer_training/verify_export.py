# coding=utf-8
"""Check that an exported tokenizer loads via qwen_tts.Qwen3TTSTokenizer and matches the training graph.

    python verify_export.py --ckpt ../../checkpoints/tokenizer/ckpt_step5000.pt \
        --hf_dir ../../checkpoints/tokenizer/hf_latest [--wav some.wav]

Checks:
  1. codes from Qwen3TTSTokenizer.encode == codes from the training forward (eval mode)
  2. Qwen3TTSTokenizer.decode(codes) ~= training decoder output (clamped)
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))
from data_loader import SyntheticAudioDataset, read_audio  # noqa: E402
from model import REPO_ROOT, TokenizerGenerator, build_tokenizer  # noqa: E402,F401

from qwen_tts.inference.qwen3_tts_tokenizer import Qwen3TTSTokenizer  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--hf_dir", required=True)
    ap.add_argument("--wav")
    args = ap.parse_args()
    dev = torch.device("cuda")

    sd = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    C = sd["config"]
    tok = build_tokenizer(C["model"].get("arch_json"), C["model"].get("overrides"))
    G = TokenizerGenerator(tok, C["quantizer"], sd["G"].get("semantic_head.weight", torch.empty(0, 0)).shape[0])
    G.load_state_dict(sd["G"])
    G.to(dev).eval()

    if args.wav:
        wav = read_audio(args.wav, 24000)[: 24000 * 10]
    else:
        wav = SyntheticAudioDataset(1)[0]["wav"][0]
    wav = wav[: (wav.shape[0] // 1920) * 1920]
    x = wav[None, None].to(dev)

    with torch.no_grad():
        out = G(x)
    codes_train = out["codes"][0].T.cpu()  # (T, 16)
    wav_train = out["wav_hat"][0, 0].clamp(-1, 1).cpu()

    t = Qwen3TTSTokenizer.from_pretrained(args.hf_dir, device_map="cuda")
    enc = t.encode(wav.numpy(), sr=24000)
    codes_inf = enc.audio_codes[0].cpu()  # (T, 16)
    n = min(len(codes_inf), len(codes_train))
    match = (codes_inf[:n] == codes_train[:n]).float().mean(0)
    print(f"frames train={len(codes_train)} inference={len(codes_inf)}")
    print("per-codebook code agreement:", np.round(match.numpy(), 4).tolist())

    wavs, sr = t.decode(enc)
    w_inf = torch.as_tensor(wavs[0]).float()
    m = min(len(w_inf), len(wav_train))
    err = (w_inf[:m] - wav_train[:m]).abs().max().item()
    print(f"decode sr={sr} len={len(w_inf)} max|inference - train| = {err:.2e}")
    ok = match.min().item() > 0.99 and err < 1e-2
    print("EXPORT OK" if ok else "EXPORT MISMATCH")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()

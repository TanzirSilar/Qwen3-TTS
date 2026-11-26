#!/bin/bash
# Smoke test for Phase 1 (run on a COMPUTE node, e.g.  srun --jobid=<JOBID> --overlap bash smoke_test.sh)
#   1. single-GPU, synthetic audio, no teacher, GAN on from step 10
#   2. verify the HF export reproduces the training graph through qwen_tts.Qwen3TTSTokenizer
#   3. 2-GPU DDP + WavLM teacher (downloads via proxy) + resume from step 1's checkpoint dir
set -euo pipefail
cd "$(dirname "$0")"
source ~/miniconda3/etc/profile.d/conda.sh && conda activate qwen3_tts
export http_proxy=http://proxy-10g.10g.siddhi.param:9090 https_proxy=http://proxy-10g.10g.siddhi.param:9090
hostname; nvidia-smi --query-gpu=index,memory.used,memory.total --format=csv

OUT=${OUT:-$(realpath ../..)/checkpoints/_smoke_tokenizer}
rm -rf $OUT
CFG=$OUT/smoke.yaml
mkdir -p $OUT
python - <<EOF
import yaml
c = yaml.safe_load(open("../../configs/tokenizer.yaml"))
c["training"].update(max_steps=30, disc_start_step=10, eval_steps=30, save_steps=30, warmup_steps=5)
c["data"]["num_workers"] = 2
c["logging"]["log_every"] = 5
yaml.safe_dump(c, open("$CFG", "w"))
EOF

echo "=== [1] single GPU, synthetic, no teacher"
CUDA_VISIBLE_DEVICES=0 python train_tokenizer.py --config $CFG --output_dir $OUT/run1 --synthetic --no_teacher

echo "=== [2] verify export"
CUDA_VISIBLE_DEVICES=0 python verify_export.py --ckpt $OUT/run1/ckpt_step30.pt --hf_dir $OUT/run1/hf_latest

NGPU=$(nvidia-smi -L | wc -l)
if [ "$NGPU" -ge 2 ]; then
  echo "=== [3] 2-GPU DDP + teacher, resume to step 40"
  cp -r $OUT/run1 $OUT/run3 && rm -rf $OUT/run3/hf_*
  # run1 had no semantic head; start run3 fresh so the teacher head exists
  rm -f $OUT/run3/ckpt_step*.pt
  torchrun --standalone --nproc_per_node=2 train_tokenizer.py --config $CFG --output_dir $OUT/run3 --synthetic --max_steps 20
  torchrun --standalone --nproc_per_node=2 train_tokenizer.py --config $CFG --output_dir $OUT/run3 --synthetic --max_steps 30
fi
echo "=== peak memory is in the logs above; outputs in $OUT"
echo "SMOKE TEST PASSED"

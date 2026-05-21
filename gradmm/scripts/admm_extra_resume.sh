#!/bin/bash
# Resume the 4 paused ADMM generation runs (started 2026-05-21 03:05–03:10).
# Each job uses its ORIGINAL work_base_dir so generate.py picks up partial state
# from rng_states.pth + synthetic_data.jsonl and skips already-done outer iters.
#
# Usage: ./scripts/admm_extra_resume.sh  (run from gradmm/ directory)
# Parallel: launches all 4 in background on GPUs 0-3. Wait blocks until done.

set -u
LOG_DIR=logs/admm/
mkdir -p $LOG_DIR
PY=/data/yutong/envs/gradmm/bin/python
ts=$(date +"%Y-%m-%d_%H-%M-%S")  # only used for log filename; work_base_dir stays the same

# Shared args
COMMON="--rng_seed 42 --dataset sst2 --split validation --batch_size 50 --n_steps 30 \
--n_gen_samples 100 --subset_size 50 --n_gen 10 --gen_bs 10 \
--use_auto_gen_tokens true --print_full true --print_every 10 --save_every 1 \
--model_name phi --opt_alg admm --admm_inner_steps 50 --grad_clip 1.0 --topk 200"
DP="--use_dp true --dp_epsilon 0.05 --dp_delta 1e-4 --dp_c 1.0"

launch() {
    local gpu=$1 rho=$2 wbd=$3 tag=$4 dp_args=$5
    local log="${LOG_DIR}/${tag}_resume_${ts}.log"
    # Sanity: confirm work_base_dir exists, else skip
    if [ ! -d "$wbd" ]; then
        echo "SKIP gpu=$gpu $tag: $wbd does not exist"
        return
    fi
    # Check completion: count lines in synthetic_data.jsonl
    local sd="$wbd/SST2-validation-phi-nreal100-steps30-nsyn10-admm-"$(basename $wbd | sed -E 's/sst2-validation-phi-nreal50-steps30-nsyn10-admm-//; s/_2026-05-21.*//')"/synthetic_data.jsonl"
    local n=0
    [ -f "$sd" ] && n=$(wc -l < "$sd")
    if [ "$n" -ge 200 ]; then
        echo "SKIP gpu=$gpu $tag: already complete ($n samples)"
        return
    fi
    echo "RESUME gpu=$gpu $tag (current: $n samples) → log: $log"
    CUDA_VISIBLE_DEVICES=$gpu $PY generate.py $COMMON --admm_rho $rho $dp_args --work_base_dir "$wbd" > "$log" 2>&1 &
    echo "  pid=$!"
}

launch 0 0.001 \
    /data/yutong/synthetic_data/sst2-validation-phi-nreal50-steps30-nsyn10-admm-rho0.001-inner50-seed42_2026-05-21_03-05-58 \
    sst2-validation-phi-nreal50-steps30-nsyn10-admm-rho0.001-inner50-seed42 \
    ""

launch 1 0.05 \
    /data/yutong/synthetic_data/sst2-validation-phi-nreal50-steps30-nsyn10-admm-rho0.05-inner50-seed42_2026-05-21_03-10-27 \
    sst2-validation-phi-nreal50-steps30-nsyn10-admm-rho0.05-inner50-seed42 \
    ""

launch 2 0.001 \
    /data/yutong/synthetic_data/sst2-validation-phi-nreal50-steps30-nsyn10-admm-dp_eps0.05-rho0.001-inner50-seed42_2026-05-21_03-10-27 \
    sst2-validation-phi-nreal50-steps30-nsyn10-admm-dp_eps0.05-rho0.001-inner50-seed42 \
    "$DP"

launch 3 0.05 \
    /data/yutong/synthetic_data/sst2-validation-phi-nreal50-steps30-nsyn10-admm-dp_eps0.05-rho0.05-inner50-seed42_2026-05-21_03-10-27 \
    sst2-validation-phi-nreal50-steps30-nsyn10-admm-dp_eps0.05-rho0.05-inner50-seed42 \
    "$DP"

echo "All resume jobs dispatched."
wait
echo "=== All 4 generation runs complete ==="

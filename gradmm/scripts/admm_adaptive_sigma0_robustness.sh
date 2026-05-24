#!/bin/bash
# σ₀-robustness benchmark for adaptive ADMM rho in GRADMM.
# Sweep INITIAL rho across {0.001, 0.01, 0.1, 1, 10} — should all converge
# to similar synthetic-data quality if adaptive mode works.
#
# Each run: full paper config (n_steps=30, n_gen=10, subset_size=50).
# ~3-5h per run on a single GPU. Set LIST_GPU to parallelize.
#
# Usage: ./scripts/admm_adaptive_sigma0_robustness.sh

set -u
current_time=$(date +"%Y-%m-%d_%H-%M-%S")
LOG_DIR=logs/admm_adaptive/
mkdir -p $LOG_DIR
PY=/data/yutong/envs/gradmm/bin/python
BASE=/data/yutong/synthetic_data

# Paper hyperparams (fixed)
COMMON="--rng_seed 42 --dataset sst2 --split validation \
--batch_size 50 --n_steps 30 --n_gen_samples 100 \
--subset_size 50 --n_gen 10 --gen_bs 10 \
--use_auto_gen_tokens true --print_full true --print_every 10 --save_every 1 \
--model_name phi --opt_alg admm --admm_inner_steps 50 \
--grad_clip 1.0 --topk 200"

# Adaptive rho hyperparams (Lipschitz floor variant — the most theory-backed)
ADAPT="--rho_mode online_convex_bal_lipschitz \
--eta_u 0.05 --G_clip 10.0 --rho_ema_beta 0.0 --rho_update_freq 1 \
--lipschitz_floor_alpha 1.0 --lipschitz_ema_beta 0.9 \
--lipschitz_min_dz 1e-6 --lipschitz_max 1e4"

# σ₀ sweep — different initial rho values; adaptive should converge regardless
LIST_INIT_RHO=(0.001 0.01 0.1 1 10)
LIST_GPU=(0)   # parallelize: e.g. (0 1 2 3 4) for 5 GPUs

for i in "${!LIST_INIT_RHO[@]}"; do
    rho0="${LIST_INIT_RHO[i]}"
    gpu="${LIST_GPU[i % ${#LIST_GPU[@]}]}"
    tag="sst2-validation-phi-nreal50-steps30-nsyn10-admm-adaptive_lipschitz-rho0_${rho0}-inner50-seed42"
    work_base_dir="${BASE}/${tag}_${current_time}"
    log="${LOG_DIR}/${tag}_${current_time}.log"
    echo "=== rho0=${rho0} on GPU ${gpu} → ${work_base_dir} ==="
    CUDA_VISIBLE_DEVICES=$gpu $PY generate.py $COMMON --admm_rho $rho0 $ADAPT \
        --work_base_dir $work_base_dir 2>&1 | tee "$log" &

    # Stagger so wandb doesn't get confused; remove `&` and `sleep` for serial
    sleep 30
done

wait
echo "=== All sigma_0 runs complete. ==="

#!/bin/bash

current_time=$(date +"%Y-%m-%d_%H-%M-%S")
LOG_DIR=logs/admm_syn/
mkdir -p $LOG_DIR
task_name=SynSST2 # SynSST2, SynTwitterEmotion, SynRottenTomatoes, SynIMDB, SynRTPolarity
MODEL=microsoft/phi-1_5
# Insert synthetic data paths here
# Updated 2026-05-21: switched to top_n=20 + balance_score=False (cell 12/13 in Filtering.ipynb).
# All 10 configs now have >=8 neg samples per file. Compare to paper Table 1 SST-2 #data=20 (87.9/87.2).
list_syn_data_path=(
    "/data/yutong/synthetic_data/sst2-validation-phi-nreal50-steps30-nsyn10-admm-dp_eps0.05-rho0.01-inner50-seed42_2026-05-18_15-33-22/SST2-validation-phi-nreal100-steps30-nsyn10-admm-dp_eps0.05-rho0.01-inner50-seed42/synthetic_data_clean_remove_cls_phi_sst2_positive_negative_instrFalse_fsTrue_top20_score_alpha0.05_per_label.jsonl"
    "/data/yutong/synthetic_data/sst2-validation-phi-nreal50-steps30-nsyn10-admm-dp_eps0.05-rho0.01-inner50-seed42_2026-05-18_15-33-22/SST2-validation-phi-nreal100-steps30-nsyn10-admm-dp_eps0.05-rho0.01-inner50-seed42/synthetic_data_clean_remove_cls_phi_sst2_positive_negative_instrFalse_fsTrue_top20_score_alpha0_per_label.jsonl"
    "/data/yutong/synthetic_data/sst2-validation-phi-nreal50-steps30-nsyn10-admm-dp_eps0.05-rho0.1-inner50-seed42_2026-05-18_15-33-22/SST2-validation-phi-nreal100-steps30-nsyn10-admm-dp_eps0.05-rho0.1-inner50-seed42/synthetic_data_clean_remove_cls_phi_sst2_positive_negative_instrFalse_fsTrue_top20_score_alpha0.05_per_label.jsonl"
    "/data/yutong/synthetic_data/sst2-validation-phi-nreal50-steps30-nsyn10-admm-dp_eps0.05-rho0.1-inner50-seed42_2026-05-18_15-33-22/SST2-validation-phi-nreal100-steps30-nsyn10-admm-dp_eps0.05-rho0.1-inner50-seed42/synthetic_data_clean_remove_cls_phi_sst2_positive_negative_instrFalse_fsTrue_top20_score_alpha0_per_label.jsonl"
    "/data/yutong/synthetic_data/sst2-validation-phi-nreal50-steps30-nsyn10-admm-dp_eps0.05-rho0.5-inner50-seed42_2026-05-18_15-33-22/SST2-validation-phi-nreal100-steps30-nsyn10-admm-dp_eps0.05-rho0.5-inner50-seed42/synthetic_data_clean_remove_cls_phi_sst2_positive_negative_instrFalse_fsTrue_top20_score_alpha0.05_per_label.jsonl"
    "/data/yutong/synthetic_data/sst2-validation-phi-nreal50-steps30-nsyn10-admm-dp_eps0.05-rho0.5-inner50-seed42_2026-05-18_15-33-22/SST2-validation-phi-nreal100-steps30-nsyn10-admm-dp_eps0.05-rho0.5-inner50-seed42/synthetic_data_clean_remove_cls_phi_sst2_positive_negative_instrFalse_fsTrue_top20_score_alpha0_per_label.jsonl"
    "/data/yutong/synthetic_data/sst2-validation-phi-nreal50-steps30-nsyn10-admm-dp_eps0.05-rho1-inner50-seed42_2026-05-18_15-33-22/SST2-validation-phi-nreal100-steps30-nsyn10-admm-dp_eps0.05-rho1.0-inner50-seed42/synthetic_data_clean_remove_cls_phi_sst2_positive_negative_instrFalse_fsTrue_top20_score_alpha0.05_per_label.jsonl"
    "/data/yutong/synthetic_data/sst2-validation-phi-nreal50-steps30-nsyn10-admm-dp_eps0.05-rho1-inner50-seed42_2026-05-18_15-33-22/SST2-validation-phi-nreal100-steps30-nsyn10-admm-dp_eps0.05-rho1.0-inner50-seed42/synthetic_data_clean_remove_cls_phi_sst2_positive_negative_instrFalse_fsTrue_top20_score_alpha0_per_label.jsonl"
    "/data/yutong/synthetic_data/sst2-validation-phi-nreal50-steps30-nsyn10-admm-dp_eps0.05-rho5-inner50-seed42_2026-05-18_15-33-22/SST2-validation-phi-nreal100-steps30-nsyn10-admm-dp_eps0.05-rho5.0-inner50-seed42/synthetic_data_clean_remove_cls_phi_sst2_positive_negative_instrFalse_fsTrue_top20_score_alpha0.05_per_label.jsonl"
    "/data/yutong/synthetic_data/sst2-validation-phi-nreal50-steps30-nsyn10-admm-dp_eps0.05-rho5-inner50-seed42_2026-05-18_15-33-22/SST2-validation-phi-nreal100-steps30-nsyn10-admm-dp_eps0.05-rho5.0-inner50-seed42/synthetic_data_clean_remove_cls_phi_sst2_positive_negative_instrFalse_fsTrue_top20_score_alpha0_per_label.jsonl"
    "/data/yutong/synthetic_data/sst2-validation-phi-nreal50-steps30-nsyn10-admm-rho0.01-inner50-seed42_2026-05-18_15-31-31/SST2-validation-phi-nreal100-steps30-nsyn10-admm-rho0.01-inner50-seed42/synthetic_data_clean_remove_cls_phi_sst2_positive_negative_instrFalse_fsTrue_top20_score_alpha0.05_per_label.jsonl"
    "/data/yutong/synthetic_data/sst2-validation-phi-nreal50-steps30-nsyn10-admm-rho0.01-inner50-seed42_2026-05-18_15-31-31/SST2-validation-phi-nreal100-steps30-nsyn10-admm-rho0.01-inner50-seed42/synthetic_data_clean_remove_cls_phi_sst2_positive_negative_instrFalse_fsTrue_top20_score_alpha0_per_label.jsonl"
    "/data/yutong/synthetic_data/sst2-validation-phi-nreal50-steps30-nsyn10-admm-rho0.1-inner50-seed42_2026-05-18_15-31-31/SST2-validation-phi-nreal100-steps30-nsyn10-admm-rho0.1-inner50-seed42/synthetic_data_clean_remove_cls_phi_sst2_positive_negative_instrFalse_fsTrue_top20_score_alpha0.05_per_label.jsonl"
    "/data/yutong/synthetic_data/sst2-validation-phi-nreal50-steps30-nsyn10-admm-rho0.1-inner50-seed42_2026-05-18_15-31-31/SST2-validation-phi-nreal100-steps30-nsyn10-admm-rho0.1-inner50-seed42/synthetic_data_clean_remove_cls_phi_sst2_positive_negative_instrFalse_fsTrue_top20_score_alpha0_per_label.jsonl"
    "/data/yutong/synthetic_data/sst2-validation-phi-nreal50-steps30-nsyn10-admm-rho0.5-inner50-seed42_2026-05-18_15-31-31/SST2-validation-phi-nreal100-steps30-nsyn10-admm-rho0.5-inner50-seed42/synthetic_data_clean_remove_cls_phi_sst2_positive_negative_instrFalse_fsTrue_top20_score_alpha0.05_per_label.jsonl"
    "/data/yutong/synthetic_data/sst2-validation-phi-nreal50-steps30-nsyn10-admm-rho0.5-inner50-seed42_2026-05-18_15-31-31/SST2-validation-phi-nreal100-steps30-nsyn10-admm-rho0.5-inner50-seed42/synthetic_data_clean_remove_cls_phi_sst2_positive_negative_instrFalse_fsTrue_top20_score_alpha0_per_label.jsonl"
    "/data/yutong/synthetic_data/sst2-validation-phi-nreal50-steps30-nsyn10-admm-rho1-inner50-seed42_2026-05-18_15-31-31/SST2-validation-phi-nreal100-steps30-nsyn10-admm-rho1.0-inner50-seed42/synthetic_data_clean_remove_cls_phi_sst2_positive_negative_instrFalse_fsTrue_top20_score_alpha0.05_per_label.jsonl"
    "/data/yutong/synthetic_data/sst2-validation-phi-nreal50-steps30-nsyn10-admm-rho1-inner50-seed42_2026-05-18_15-31-31/SST2-validation-phi-nreal100-steps30-nsyn10-admm-rho1.0-inner50-seed42/synthetic_data_clean_remove_cls_phi_sst2_positive_negative_instrFalse_fsTrue_top20_score_alpha0_per_label.jsonl"
    "/data/yutong/synthetic_data/sst2-validation-phi-nreal50-steps30-nsyn10-admm-rho5-inner50-seed42_2026-05-18_15-31-31/SST2-validation-phi-nreal100-steps30-nsyn10-admm-rho5.0-inner50-seed42/synthetic_data_clean_remove_cls_phi_sst2_positive_negative_instrFalse_fsTrue_top20_score_alpha0.05_per_label.jsonl"
    "/data/yutong/synthetic_data/sst2-validation-phi-nreal50-steps30-nsyn10-admm-rho5-inner50-seed42_2026-05-18_15-31-31/SST2-validation-phi-nreal100-steps30-nsyn10-admm-rho5.0-inner50-seed42/synthetic_data_clean_remove_cls_phi_sst2_positive_negative_instrFalse_fsTrue_top20_score_alpha0_per_label.jsonl"
)
num_train=100
max_steps=200
per_device_train_batch_size=16
gradient_accumulation_steps=1
LIST_TRAIN_SET_SEED=(
    0
    1
    2
)
kept_eval_as_train=False
num_eval_to_keep=100

LIST_LR=(
    0.000007
    0.00001
    0.000015
)
LIST_GPU=(
    0
)

# Function to run training job
run_training_job() {
    local syn_data_path="$1"
    local lr="$2"
    local gpu="$3"
    local train_set_seed="$4"

    # Tag generation
    local synthetic_tag
    synthetic_tag=$(echo -n "$syn_data_path" | md5sum | awk '{print $1}')
    echo "Training Tag: $synthetic_tag"

    local training_tag="kept_eval_as_train=${kept_eval_as_train}_num_eval_to_keep=${num_eval_to_keep}_lr${lr}_seed${train_set_seed}"
    echo "Training Tag: $training_tag"

    local full_tag="${synthetic_tag}_${training_tag}"
    echo "Full Tag: $full_tag"
    local output_dir=./synthetic_data_FT/${current_time}/result/${full_tag}/output
    echo "Output Dir: $output_dir"

    CUDA_VISIBLE_DEVICES=$gpu PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True /data/yutong/envs/gradmm/bin/python run.py \
        --trainer regular \
        --num_dev 0 \
        --num_eval 1000 \
        --logging_steps 10 \
        --output_dir ${output_dir} \
        --tag $full_tag \
        --lr_scheduler_type linear \
        --load_best_model_at_end \
        --eval_strategy steps \
        --save_strategy steps \
        --eval_steps 50 \
        --save_steps 50 \
        --overwrite_output_dir \
        --no_save_weights \
        --save_only_model \
        --train_as_classification \
        --model_name $MODEL \
        --task_name $task_name \
        --train_set_seed $train_set_seed \
        --per_device_train_batch_size $per_device_train_batch_size \
        --gradient_accumulation_steps $gradient_accumulation_steps \
        --max_steps $max_steps \
        --learning_rate $lr \
        --kept_eval_as_train $kept_eval_as_train \
        --num_eval_to_keep $num_eval_to_keep \
        --num_train $num_train \
        --syn_data_path $syn_data_path 2>&1 | tee -a ${LOG_DIR}/${full_tag}_${current_time}.log
    
    sleep 10
}

# Parallel job execution — proper per-GPU tracking, no double-booking.
# Skips a (syn_data_path, lr, seed) combo if a matching main_results.json already exists in any launch dir.
run_parallel_jobs() {
    declare -A gpu_pid  # gpu_pid[$gpu] = pid of the job currently using that gpu
    local skipped=0 launched=0

    for syn_data_path in "${list_syn_data_path[@]}"; do
        for lr in "${LIST_LR[@]}"; do
            for train_set_seed in "${LIST_TRAIN_SET_SEED[@]}"; do
                # Build full_tag the same way run_training_job does, to check for prior completion.
                local synthetic_tag
                synthetic_tag=$(echo -n "$syn_data_path" | md5sum | awk '{print $1}')
                local training_tag="kept_eval_as_train=${kept_eval_as_train}_num_eval_to_keep=${num_eval_to_keep}_lr${lr}_seed${train_set_seed}"
                local full_tag="${synthetic_tag}_${training_tag}"

                # Skip if a previous launch already produced a final main_results.json for this combo.
                if find ./synthetic_data_FT -type f -path "*/result/${full_tag}/output/main_results.json" 2>/dev/null | grep -q .; then
                    echo "SKIP (already done): $full_tag"
                    ((skipped++))
                    continue
                fi

                # Wait for a GPU to become free (poll every 5s).
                local gpu=""
                while [ -z "$gpu" ]; do
                    for g in "${LIST_GPU[@]}"; do
                        local p="${gpu_pid[$g]:-}"
                        if [ -z "$p" ] || ! kill -0 "$p" 2>/dev/null; then
                            gpu="$g"
                            unset 'gpu_pid[$g]'
                            break
                        fi
                    done
                    [ -z "$gpu" ] && sleep 5
                done

                # Launch on the free GPU and remember which PID owns it.
                run_training_job "$syn_data_path" "$lr" "$gpu" "$train_set_seed" &
                gpu_pid[$gpu]=$!
                ((launched++))
                echo "LAUNCHED ${launched}: gpu=$gpu pid=${gpu_pid[$gpu]} $full_tag"
                sleep 2  # tiny stagger so CUDA init doesn't pile up
            done
        done
    done

    echo "All jobs dispatched. launched=$launched, skipped=$skipped. Waiting for remaining jobs..."
    wait
}

# Calculate total number of combinations
total_combinations=$((${#list_syn_data_path[@]} * ${#LIST_LR[@]} * ${#LIST_TRAIN_SET_SEED[@]}))

# Execute parallel jobs
run_parallel_jobs

echo "Total Combinations: $total_combinations"
echo "All training jobs completed."
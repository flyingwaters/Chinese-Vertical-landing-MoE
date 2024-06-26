#!/usr/bin/bash

#SBATCH -J mixtral
#SBATCH -o logs/%j.log
#SBATCH -e logs/%j.err
#SBATCH -p gpu
#SBATCH -N 4
#SBATCH --ntasks-per-node=1
#SBATCH -c 56
#SBATCH --mem=800G
#SBATCH --gres=gpu:8

TRAIN_DATASETS=(
    1:SkyPile-150B-2023
)

VALID_DATASETS=(
    SkyPile-150B-2023
)

TRAIN_PARAMS=""
TRAIN_PARAMS+=" --enable_lora"
TRAIN_PARAMS+=" --lora_alpha 128"
TRAIN_PARAMS+=" --lora_dropout 0.05"
TRAIN_PARAMS+=" --lora_rank 64"
TRAIN_PARAMS+=" --lora_target_modules q_proj v_proj k_proj o_proj w1 w2 w3"
TRAIN_PARAMS+=" --lora_modules_to_save embed_tokens lm_head"
TRAIN_PARAMS+=" --model_name_or_path /home/clouduser/fengly/deepseek-moe-16b-chat"
TRAIN_PARAMS+=" --tokenizer_name_or_path tokenizer/Mixtral-8x7B-v0.1-vocab"
TRAIN_PARAMS+=" --train_datasets ${TRAIN_DATASETS[*]}"
TRAIN_PARAMS+=" --valid_datasets ${VALID_DATASETS[*]}"
TRAIN_PARAMS+=" --dataloader_drop_last"
TRAIN_PARAMS+=" --cache_dir hf-cache"
TRAIN_PARAMS+=" --output_dir outputs"
TRAIN_PARAMS+=" --num_train_epochs 1"
TRAIN_PARAMS+=" --model_max_length 2048"
TRAIN_PARAMS+=" --per_device_train_batch_size 16"
TRAIN_PARAMS+=" --gradient_accumulation_steps 1"
TRAIN_PARAMS+=" --optim adamw_torch_fused"
TRAIN_PARAMS+=" --per_device_eval_batch_size 16"
TRAIN_PARAMS+=" --evaluation_strategy steps"
TRAIN_PARAMS+=" --eval_steps 500"
TRAIN_PARAMS+=" --save_strategy steps"
TRAIN_PARAMS+=" --save_steps 1000"
TRAIN_PARAMS+=" --learning_rate 1e-5"
TRAIN_PARAMS+=" --warmup_ratio 0.05"
TRAIN_PARAMS+=" --logging_dir logs"
TRAIN_PARAMS+=" --logging_strategy steps"
TRAIN_PARAMS+=" --logging_steps 1"
TRAIN_PARAMS+=" --lr_scheduler_type cosine"
TRAIN_PARAMS+=" --report_to tensorboard"
TRAIN_PARAMS+=" --gradient_checkpointing"
TRAIN_PARAMS+=" --bf16"
TRAIN_PARAMS+=" --deepspeed ds-config/config.json"

TORCHRUN_PARAMS='--nproc_per_node 8 --node_rank=$SLURM_NODEID --nnodes=$SLURM_JOB_NUM_NODES --rdzv_id=0 --rdzv_endpoint=$MASTER_ADDR:$MASTER_PORT'
srun --label --export=ALL bash -c "
    MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1);
    MASTER_PORT=55511;
    torchrun $TORCHRUN_PARAMS train.py $TRAIN_PARAMS
"

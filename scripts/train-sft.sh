#!/usr/bin/bash
export CUDA_VISIBLE_DEVICES=4,5,6,7

torchrun --nproc_per_node 4 train.py \
         --enable_lora \
         --lora_alpha 128 \
--lora_dropout 0.05 \
--lora_rank 32 \
 --lora_target_modules q_proj v_proj k_proj o_proj w1 w2 w3 \
 --lora_modules_to_save embed_tokens lm_head \
 --model_name_or_path /home/clouduser/fengly/models/HIT-SCIR/Chinese-Mixtral-8x7B \
 --tokenizer_name_or_path /home/clouduser/fengly/models/HIT-SCIR/Chinese-Mixtral-8x7B \
 --neftune_noise_alpha 5 \
--train_datasets "1:ky-finetune" \
 --valid_datasets "ky-finetune" \
--dataloader_drop_last \
 --cache_dir hf-cache \
 --output_dir outputs \
 --num_train_epochs 2 \
 --model_max_length 512 \
 --per_device_train_batch_size 8 \
 --gradient_accumulation_steps 10 \
 --optim adamw_torch_fused \
 --per_device_eval_batch_size 1 \
 --evaluation_strategy steps \
 --eval_steps 100 \
 --save_total_limit 5\
 --save_strategy steps \
 --save_steps 100 \
 --learning_rate 1e-5 \
 --warmup_ratio 0.03 \
 --logging_dir logs \
 --logging_strategy steps \
 --logging_steps 1 \
 --lr_scheduler_type cosine \
 --report_to tensorboard \
 --gradient_checkpointing True\
 --bf16 \
 --deepspeed /home/clouduser/fengly/Chinese-Mixtral-8x7B/ds-config/config.json

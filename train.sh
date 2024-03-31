DATA_PATH="/home/clouduser/fengly/DeepSeek-MoE/train_whole_answer.json"
OUTPUT_PATH="/home/clouduser/fengly/Chinese-Mixtral-8x7B/lora_rank_1_out_dropout_0.1"
MODEL_PATH="/home/clouduser/fengly/models/HIT-SCIR/Chinese-Mixtral-8x7B"
OMP_NUM_THREADS=8
MKL_NUM_THREADS=1


deepspeed --include localhost:4,5,6,7 finetune.py \
    --model_name_or_path $MODEL_PATH \
    --do_eval True \
    --data_path $DATA_PATH \
    --output_dir $OUTPUT_PATH \
    --num_train_epochs 3 \
    --model_max_length 800 \
    --per_device_train_batch_size 12 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 4 \
    --evaluation_strategy "steps" \
    --eval_steps 50 \
    --save_strategy "steps" \
    --save_steps 100 \
    --save_total_limit 15 \
    --learning_rate 5e-5 \
    --warmup_steps 30 \
    --logging_steps 1 \
    --lr_scheduler_type "cosine" \
    --gradient_checkpointing True \
    --report_to "tensorboard" \
    --deepspeed ds-config/ds_config_zero2_no_offload.json \
    --bf16 True \
    --use_lora True \
    --max_grad_norm 0.3 \
    --double_quant False \
    --lora_r 1 \
    --lora_dropout 0.1 \
    --lora_alpha 16 \
    --quant_type nf4

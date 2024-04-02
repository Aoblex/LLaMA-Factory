#!/bin/bash
accelerate launch src/train_bash.py \
    --ddp_timeout 18000000 \
    --stage sft \
    --do_train \
    --model_name_or_path /data/models/Baichuan2-13B-chat \
    --dataset statchat_identity,knowledge_dataset,statistics_dataset,deeplearning_dataset,mathematical_statistics_dataset,machine_learning_dataset \
    --template default \
    --finetuning_type lora \
    --lora_target W_pack \
    --output_dir saves/statchat/lora-v4 \
    --overwrite_output_dir \
    --overwrite_cache \
    --per_device_train_batch_size 8 \
    --gradient_accumulation_steps 4 \
    --lr_scheduler_type cosine \
    --logging_steps 10 \
    --save_steps 1000 \
    --learning_rate 6e-4 \
    --num_train_epochs 3.0 \
    --plot_loss \
    --lora_rank 8 \
    --lora_alpha 12 \
    --fp16
    
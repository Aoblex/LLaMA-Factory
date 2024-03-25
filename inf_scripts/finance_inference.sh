#!/bin/bash

# USE_MODELSCOPE_HUB=1 CUDA_VISIBLE_DEVICES=0 python src/train_bash.py \
accelerate launch src/train_bash.py \
    --stage sft \
    --do_predict \
    --model_name_or_path /data/models/Baichuan2-13B-chat-v2 \
    --adapter_name_or_path saves/finance/lora \
    --dataset finance_dataset \
    --dataset_dir data \
    --template default \
    --finetuning_type lora \
    --output_dir saves/finance/predict \
    --overwrite_cache \
    --overwrite_output_dir \
    --cutoff_len 4096 \
    --preprocessing_num_workers 16 \
    --per_device_eval_batch_size 16 \
    --predict_with_generate
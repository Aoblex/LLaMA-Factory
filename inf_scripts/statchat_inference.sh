#!/bin/bash
accelerate launch src/train_bash.py \
    --stage sft \
    --do_predict \
    --model_name_or_path /data/models/Baichuan2-13B-chat-v2 \
    --adapter_name_or_path saves/statchat/lora \
    --dataset statchat_identity,statistics_dataset,deeplearning_dataset,mathematical_statistics_dataset,machine_learning_dataset \
    --dataset_dir data \
    --template default \
    --finetuning_type lora \
    --output_dir saves/statchat/predict \
    --overwrite_cache \
    --overwrite_output_dir \
    --cutoff_len 4096 \
    --preprocessing_num_workers 16 \
    --per_device_eval_batch_size 8 \
    --predict_with_generate
    
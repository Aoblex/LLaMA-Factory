#!/bin/bash

CUDA_VISIBLE_DEVICES=0 python src/train_bash.py \
    --stage sft \
    --do_predict \
    --model_name_or_path models/ZhipuAI/chatglm3-6b \
    --adapter_name_or_path fine-tuned \
    --dataset self_cognition \
    --dataset_dir data \
    --template default \
    --finetuning_type lora \
    --output_dir saves/fine-tuned-chatglm3-6b/predict \
    --overwrite_cache \
    --overwrite_output_dir \
    --cutoff_len 1024 \
    --preprocessing_num_workers 16 \
    --per_device_eval_batch_size 1 \
    --max_samples 20 \
    --predict_with_generate
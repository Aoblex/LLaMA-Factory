#!/bin/bash
cd statchat-data
python ./src/rewrite_triplets.py --output_path datasets/know_tuning/statchat_triplets.json \
                                 --stable_temperature 0.01 \
                                 --unstable_temperature 0.3 \

python ./src/export_triplets.py
cp datasets/fine_tuning_data/statchat_KC.json ../data/statchat_KC.json
cp datasets/fine_tuning_data/statchat_KG.json ../data/statchat_KG.json
cd ..

accelerate launch src/train_bash.py \
    --ddp_timeout 18000000 \
    --stage sft \
    --do_train \
    --model_name_or_path /data/models/Baichuan2-13B-chat \
    --dataset statchat_identity,statchat_KG \
    --template default \
    --finetuning_type lora \
    --lora_target W_pack \
    --output_dir saves/statchat/knowtuning/sft \
    --overwrite_output_dir \
    --overwrite_cache \
    --per_device_train_batch_size 8 \
    --gradient_accumulation_steps 4 \
    --lr_scheduler_type cosine \
    --logging_steps 10 \
    --save_steps 1000 \
    --learning_rate 5e-5 \
    --num_train_epochs 3.0 \
    --plot_loss \
    --fp16 \

CUDA_VISIBLE_DEVICES=0 python src/train_bash.py \
    --stage dpo \
    --do_train \
    --model_name_or_path /data/models/Baichuan2-13B-chat \
    --adapter_name_or_path saves/statchat/knowtuning/sft \
    --create_new_adapter \
    --dataset statchat_KC \
    --template default \
    --finetuning_type lora \
    --lora_target W_pack \
    --output_dir saves/statchat/knowtuning/dpo \
    --overwrite_cache \
    --overwrite_output_dir \
    --preprocessing_num_workers 16 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 8 \
    --lr_scheduler_type cosine \
    --logging_steps 10 \
    --warmup_steps 20 \
    --save_steps 100 \
    --eval_steps 100 \
    --load_best_model_at_end \
    --learning_rate 1e-5 \
    --num_train_epochs 1.0 \
    --dpo_ftx 1.0 \
    --plot_loss \
    --fp16

python test_output.py
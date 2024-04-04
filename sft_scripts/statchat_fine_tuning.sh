with_book=$1

dpo_datasets="statchat_KC"
book_datasets="statistics_dataset,mathematical_statistics_dataset,deeplearning_dataset,machine_learning_dataset"

if [ $with_book = true ]; then
    sft_datasets="statchat_identity,statchat_KG,$book_datasets"
    sft_model="/data/models/StatChat-KnowTuning-sft-book"
    sft_lora_dir="saves/statchat/knowtuning/sft-book"
else
    sft_datasets="statchat_identity,statchat_KG"
    sft_model="/data/models/StatChat-KnowTuning-sft"
    sft_lora_dir="saves/statchat/knowtuning/sft"
fi

base_model="/data/models/Baichuan2-13B-chat"
dpo_lora_dir="saves/statchat/knowtuning/dpo"

CUDA_VISIBLE_DEVICES=0 python src/train_bash.py \
    --stage sft \
    --do_train \
    --model_name_or_path $base_model \
    --dataset $sft_datasets \
    --dataset_dir data \
    --template default \
    --finetuning_type lora \
    --lora_target W_pack \
    --output_dir $sft_lora_dir \
    --overwrite_cache \
    --overwrite_output_dir \
    --cutoff_len 2048 \
    --preprocessing_num_workers 16 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 4 \
    --lr_scheduler_type cosine \
    --logging_steps 5 \
    --learning_rate 5e-4 \
    --num_train_epochs 5.0 \
    --plot_loss \
    --fp16

CUDA_VISIBLE_DEVICES=0 python src/export_model.py \
    --model_name_or_path $base_model \
    --adapter_name_or_path $sft_lora_dir \
    --template default \
    --finetuning_type lora \
    --export_dir $sft_model \
    --export_size 2 \
    --export_legacy_format False

CUDA_VISIBLE_DEVICES=0 python src/train_bash.py \
    --stage dpo \
    --do_train \
    --model_name_or_path $sft_model \
    --create_new_adapter \
    --dataset $dpo_datasets \
    --template default \
    --finetuning_type lora \
    --lora_target W_pack \
    --output_dir $dpo_lora_dir \
    --overwrite_cache \
    --overwrite_output_dir \
    --preprocessing_num_workers 16 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 4 \
    --lr_scheduler_type cosine \
    --logging_steps 5 \
    --warmup_steps 20 \
    --save_steps 100 \
    --learning_rate 2e-5 \
    --num_train_epochs 1.0 \
    --dpo_ftx 1.0 \
    --plot_loss \
    --fp16

CUDA_VISIBLE_DEVICES=0 python src/export_model.py \
    --model_name_or_path $sft_model \
    --adapter_name_or_path $dpo_lora_dir \
    --template default \
    --finetuning_type lora \
    --export_dir $sft_dpo_model \
    --export_size 2 \
    --export_legacy_format False

python test_output.py
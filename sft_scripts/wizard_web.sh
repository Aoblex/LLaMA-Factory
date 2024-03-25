CUDA_VISIBLE_DEVICES=0 python src/web_demo.py \
    --model_name_or_path /data/models/WizardCoder-Python-34B-V1.0 \
    --adapter_name_or_path saves/wizard/lora \
    --template default \
    --finetuning_type lora
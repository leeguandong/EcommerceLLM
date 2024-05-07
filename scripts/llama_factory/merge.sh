#!/bin/bash
# DO NOT use quantized model or quantization_bit when merging lora weights

CUDA_VISIBLE_DEVICES=0 python ../../src/export_model.py \
    --model_name_or_path /root/lgd/e_commerce_llm/weights/Qwen1.5-0.5B/ \
    --adapter_name_or_path /root/lgd/e_commerce_llm/llama_factory/saves/Qwen1.5_0.5B_Base/lora/sft/ \
    --template qwen \
    --finetuning_type lora \
    --export_dir ../../models/qwen1.5_0.5B_base_sft \
    --export_size 2 \
    --export_legacy_format False

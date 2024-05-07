#!/bin/bash

CUDA_VISIBLE_DEVICES=0 python ../../src/cli_demo.py \
    --model_name_or_path /root/lgd/e_commerce_llm/weights/Qwen1.5-0.5B/ \
    --adapter_name_or_path /root/lgd/e_commerce_llm/llama_factory/saves/Qwen1.5_0.5B_Base/lora/sft/ \
    --template qwen \
    --finetuning_type lora

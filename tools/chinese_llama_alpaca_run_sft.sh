lr=1e-4
lora_rank=8
lora_alpha=32
lora_trainable="q_proj,v_proj,k_proj,o_proj,gate_proj,down_proj,up_proj"
modules_to_save="embed_tokens,lm_head"
lora_dropout=0.05

pretrained_model=/home/image_team/image_team_docker_home/lgd/common/Chinese-LLaMA-Alpaca-main/output_dir/
chinese_tokenizer_path=/home/image_team/image_team_docker_home/lgd/common/Chinese-LLaMA-Alpaca-main/chinese_alpaca_tokenizer/
dataset_dir=/home/image_team/image_team_docker_home/lgd/common/Chinese-LLaMA-Alpaca-main/data/
per_device_train_batch_size=4
per_device_eval_batch_size=2
gradient_accumulation_steps=8
output_dir=/home/image_team/image_team_docker_home/lgd/common/Chinese-LLaMA-Alpaca-main/output_dir_sft/
#peft_model=/home/image_team/image_team_docker_home/lgd/common/Chinese-LLaMA-Alpaca-main/output_dir_sft/
validation_file=/home/image_team/image_team_docker_home/lgd/common/Chinese-LLaMA-Alpaca-main/data/alpaca_valid.json

deepspeed_config_file=ds_zero2_no_offload.json

torchrun --nnodes 1 --nproc_per_node 4 run_clm_sft_with_peft.py \
    --deepspeed ${deepspeed_config_file} \
    --model_name_or_path ${pretrained_model} \
    --tokenizer_name_or_path ${chinese_tokenizer_path} \
    --dataset_dir ${dataset_dir} \
    --validation_split_percentage 0.001 \
    --per_device_train_batch_size ${per_device_train_batch_size} \
    --per_device_eval_batch_size ${per_device_eval_batch_size} \
    --do_train \
    --do_eval \
    --seed $RANDOM \
    --fp16 \
    --num_train_epochs 1 \
    --lr_scheduler_type cosine \
    --learning_rate ${lr} \
    --warmup_ratio 0.03 \
    --weight_decay 0 \
    --logging_strategy steps \
    --logging_steps 10 \
    --save_strategy steps \
    --save_total_limit 3 \
    --evaluation_strategy steps \
    --eval_steps 500 \
    --save_steps 2000 \
    --gradient_accumulation_steps ${gradient_accumulation_steps} \
    --preprocessing_num_workers 8 \
    --max_seq_length 512 \
    --output_dir ${output_dir} \
    --overwrite_output_dir \
    --ddp_timeout 30000 \
    --logging_first_step True \
    --lora_rank ${lora_rank} \
    --lora_alpha ${lora_alpha} \
    --trainable ${lora_trainable} \
    --modules_to_save ${modules_to_save} \
    --lora_dropout ${lora_dropout} \
    --torch_dtype float16 \
    --validation_file ${validation_file} \
    --gradient_checkpointing \
    --ddp_find_unused_parameters False
#    --peft_path ${peft_model}
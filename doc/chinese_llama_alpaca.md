[📘使用文档]() |
[🛠安装教程]() |
[👀模型库]() |
[🆕更新日志]() |
[🚀进行中的项目]() |
[🤔报告问题]()

https://zhuanlan.zhihu.com/p/631360711
peft版本：https://github.com/huggingface/peft/tree/13e53fc
https://liguandong.blog.csdn.net/article/details/134119566?spm=1001.2014.3001.5502

**转换格式**

```
python convert_llama_weights_to_hf.py --input_dir /home/image_team/image_team_docker_home/lgd/e_commerce_llm/weights/LLaMA-7B-Base/ --model_size 7B --output_dir /home/image_team/image_team_docker_home/lgd/e_commerce_llm/weights/LLaMA-7B-Base-hf/
```

**词表合并**

```
python merge_tokenizers.py   --llama_tokenizer_dir /home/image_team/image_team_docker_home/lgd/e_commerce_llm/weights/LLaMA-7B-Base-hf/   --chinese_sp_model_file /home/image_team/image_team_docker_home/lgd/common/Chinese-LLaMA-Alpaca-main/scripts/merge_tokenizer/chinese_sp.model
```

merged_tokenizer_sp：为训练好的词表模型
merged_tokenizer_hf：HF格式训练好的词表模型

**预训练第二阶段**

```
bash run.sh
```

**将lora权重与基础模型合并**

```
python merge_llama_with_chinese_lora.py     --base_model /home/image_team/image_team_docker_home/lgd/e_commerce_llm/weights/LLaMA-7B-Base-hf/   --lora_model /home/image_team/image_team_docker_home/lgd/common/Chinese-LLaMA-Alpaca-main/output_dir/pt_lora_model/     --output_type huggingface     --output_dir /home/image_team/image_team_docker_home/lgd/common/Chinese-LLaMA-Alpaca-main/output_dir/
```

```
python merge_llama_with_chinese_lora_low_mem.py     --base_model /home/image_team/image_team_docker_home/lgd/e_commerce_llm/weights/LLaMA-7B-Base-hf/      --lora_model /home/image_team/image_team_docker_home/lgd/common/Chinese-LLaMA-Alpaca-main/output_dir/pt_lora_model/     --output_type huggingface     --output_dir /home/image_team/image_team_docker_home/lgd/common/Chinese-LLaMA-Alpaca-main/output_dir/
```

**指令精调**
一定要把tokenizer换成chinese_alpaca_tokenizers，一共是49954,多一个pad token

```
bash run_sft.sh
```

**合并预训练权重和指令精调的lora权重**

```
python merge_llama_with_chinese_lora.py     --base_model /home/image_team/image_team_docker_home/lgd/e_commerce_llm/weights/LLaMA-7B-Base-hf/   --lora_model /home/image_team/image_team_docker_home/lgd/common/Chinese-LLaMA-Alpaca-main/output_dir/pt_lora_model/,"/home/image_team/image_team_docker_home/lgd/common/Chinese-LLaMA-Alpaca-main/output_dir_sft/sft_lora_model/"     --output_type huggingface     --output_dir /home/image_team/image_team_docker_home/lgd/common/Chinese-LLaMA-Alpaca-main/output_dir_all/
```

**前向推理**

```
python inference_hf.py      --base_model /home/image_team/image_team_docker_home/lgd/common/Chinese-LLaMA-Alpaca-main/output_dir_all/     --with_prompt    --interactive
```






    
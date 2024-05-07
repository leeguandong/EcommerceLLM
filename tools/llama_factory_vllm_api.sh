CUDA_VISIBLE_DEVICES=0,1 API_PORT=8000 python ../../src/api_demo.py \
  --model_name_or_path /home/image_team/image_team_docker_home/lgd/e_commerce_llm/weights/Qwen1.5-7B/ \
  --adapter_name_or_path /home/image_team/image_team_docker_home/lgd/e_commerce_llm/llama_factory/saves/Qwen1.5-7B/lora/sft/ \
  --template qwen \
  --finetuning_type lora
 # --infer_backend vllm \
 # --vllm_enforce_eager
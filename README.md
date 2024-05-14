## EcommerceLMM

基于电商数据微调的Qwen1.5系列的电商大模型，包括0.5b-base，0.5b-chat，1.8b-base，7b-base，以及基于llama3-chinese-sft版本的基础模型的sft后电商大模型。

 <p align="center">
      <a href='https://github.com/leeguandong/EcommerceLLM'>
            <img src='https://img.shields.io/badge/Project-Page-Green'>
      </a>
      <a href='https://github.com/leeguandong/EcommerceLLM'>
            <img src='https://img.shields.io/badge/Paper-Arxiv-red'>
      </a>
      </br>
      <a href="https://github.com/leeguandong/EcommerceLLM/graphs/contributors">
        <img alt="GitHub Contributors" src="https://img.shields.io/github/contributors/leeguandong/EcommerceLLM" />
      </a>
      <a href="https://github.com/leeguandong/EcommerceLLM/issues">
        <img alt="Issues" src="https://img.shields.io/github/issues/leeguandong/EcommerceLLM?color=0088ff" />
      </a>
      <a href="https://github.com/leeguandong/EcommerceLLM/pulls">
        <img alt="GitHub pull requests" src="https://img.shields.io/github/issues-pr/leeguandong/EcommerceLLM?color=0088ff" />
      </a>
      <a href=href="https://github.com/leeguandong/EcommerceLLM/stargazers">
        <img src="https://img.shields.io/github/stars/leeguandong/EcommerceLLM?color=ccf">
      </a>
      <a href=href="https://github.com/leeguandong/EcommerceLLM">
        <img src="https://img.shields.io/github/repo-size/leeguandong/EcommerceLLM.svg?style=flat-square">
      </a>
      </br>
      <a href=href="https://github.com/leeguandong/EcommerceLLM">
        <img src="https://visitor-badge.laobi.icu/badge?page_id=https://github.com/leeguandong/EcommerceLLM">
      </a>
      <a href=href="https://github.com/leeguandong/EcommerceLLM">
        <img src="https://img.shields.io/github/last-commit/leeguandong/EcommerceLLM">
      </a>
      <a href="https://github.com/leeguandong/EcommerceLLM/blob/main/LICENSE">
        <img alt="GitHub Contributors" src="https://img.shields.io/badge/License-CC%20BY--NC--SA%204.0-lightgrey.svg" />
      </a>
  </p>

## 本文贡献

- 借助电商领域的开源数据集，包括alpaca的通用数据，直播带货文案，商品主题词抽取，场景问答，选择题，客服问答，标题扩展，商品介绍seo，写文章，短标题抽取，小红书文案，根据参数扩写，文章总结，人设，tiktok数据。基于Qwen1.5和llama3-chinese-sft训练微调，权重coming soon。
## 数据集

- 数据集放开了很小一部分的开源tiktok收集的数据，我们的电商数据非常多样化，包括常见的电商领域的任务。
- 训练借助llama-factory，数据格式类似
```bash
  "alpaca_gpt4_data_zh": {
    "file_name": "alpaca_gpt4_data_zh.json",
    "file_sha1": ""
  },
  "tiktok_v1": {
    "file_name": "tiktok_v1.json",
    "file_sha1": ""
  },
```


## 快速上手

### 1.安装环境
```bash
pip install -r requirement
```
### 2.模型推理

|模型权重|下载链接|微调方法|
|:-|:-|:-|
|ecommercellm-qwen-0.5b-base|comming soon....|LoRA|
|ecommercellm-qwen-0.5b-chat|comming soon....|LoRA|
|ecommercellm-qwen-1.8b-base|comming soon....|LoRA|
|ecommercellm-qwen-7b-base|comming soon....|LoRA|
|ecommercellm-llama-chinese|comming soon....|LoRA|

#### CLI推理

```python
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

CUDA_VISIBLE_DEVICES=0 python ../../src/cli_demo.py \
    --model_name_or_path /root/lgd/e_commerce_llm/weights/Qwen1.5-0.5B/ \
    --adapter_name_or_path /root/lgd/e_commerce_llm/llama_factory/saves/Qwen1.5_0.5B_Base/lora/sft/ \
    --template qwen \
    --finetuning_type lora
```
### 3.模型训练（复现CommerceLLM）

<details>
  <summary>硬件资源</summary>
  <p>* 实验在V100 (4X, 32GB)上进行</p>
</details>

- （1）准备数据，并且写入到dataset_info.json中；
- （2）开始训练：
- 
```bash
#!/bin/bash
 
deepspeed --num_gpus 4 ../../src/train_bash.py \
    --deepspeed ../deepspeed/ds_z3_config.json \
    --stage sft \
    --do_train \
    --model_name_or_path /root/e_commerce_llm/weights/Qwen1.5-0.5B/ \
    --dataset alpaca_gpt4_data_zh,sft_gpt4_data \
    --dataset_dir ../../data \
    --template qwen \
    --finetuning_type lora \
    --lora_target q_proj,v_proj  \
    --output_dir ../../saves/Qwen1.5_0.5B_Base/lora/sft \
    --overwrite_cache \
    --overwrite_output_dir \
    --cutoff_len 1024 \
    --preprocessing_num_workers 16 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 2 \
    --lr_scheduler_type cosine \
    --logging_steps 10 \
    --warmup_steps 20 \
    --save_steps 100 \
    --eval_steps 100 \
    --evaluation_strategy steps \
    --learning_rate 5e-5 \
    --num_train_epochs 3.0 \
    --max_samples 3000 \
    --val_size 0.1 \
    --ddp_timeout 180000000 \
    --plot_loss
```
这里的复现过程非常简单，主要是很多过程我们都为大家准备好了，大家可以随时复现一个自己的`EcommerceLLM`。

## 效果展示

*以下效果来自于**低质量**的数据训练和权重

影像诊断：
|instruction| qwen1.5-1.8b                                                 | qwen1.5-7b                                                   |llama3-chinese-sft|
|:-|:-|:-|--|
| 你用了开源的数据吗？                                         | 是的，我使用了开源的数据。                                   | 是的，我使用了开源的数据。我使用了由xx开发的xx零售垂域大模型，它基于深度学习技术，使用了大规模的电商数据进行训练。这些数据包括商品信息、用户行为、评论等，用于生成回答。 | 我没有使用开源的数据，因为我是一个人工智能语言模型，我没有能力获取和使用开源数据。我的回答都是基于我所接受的训练数据和用户输入的信息。我的回答是根据用户的需求和问题生成的。 |
| 请帮我写一篇小红书风格的电商营销文案                         | 小红书风格的电商营销文案：\n\n亲爱的朋友们，你们是否经常在小红书上看到各种各样的美妆、时尚、美食等推荐？这些推荐的背后，是无数个美妆博主和美食博主们的心血和努力。他们用自己的专业知识和经验，为消费者提供最真实、最实用的购物建议。\n\n今天，我要给大家推荐一款超级好用的电商产品，它将帮助你们在小红书上获得更多的曝光和流量。这款产品就是我们的XXX品牌，它是一款专为小红书用户设计的美妆工具。\n\n首先，这款美妆工具具有强大的数据分析功能，可以实时监测你的美妆和护肤效果，为你提供最精准的建议和指导。无论你是想提升你的妆容效果，还是想了解最新的护肤趋势，这款工具都能满足你的需求。\n\n其次，这款美妆工具还具有强大的社交分享功能。你可以将你的美妆和护肤心得分享到小红书上，让更多的人看到你的推荐和分享。同时，你也可以通过这款工具，与其他美妆博主和美食博主进行交流和互动，获取更多的灵感和建议。\n\n最后，我要强调的是，这款美妆工具的使用非常简单。只需要下载并安装，就可以开始使用。无论你是新手还是老手，都能轻松上手，享受美妆和护肤的乐趣。\n\n希望我的推荐能帮助到你们，让你们在小红书上获得更多的曝光和流量。谢谢大家的关注和支持！ | 标题：小红书风格的电商营销文案，让你的产品瞬间爆红！\n\n正文：大家好，有没有遇到过这样的情况，你想要购买一款产品，但是却不知道该如何选择？或者你想要推广一款产品，但是却不知道如何吸引更多的用户？今天，我就来给大家分享一下如何写出小红书风格的电商营销文案，让你的产品瞬间爆红！\n\n首先，我们要明确一点，小红书风格的电商营销文案是以用户为中心的。我们要从用户的角度出发，了解他们的需求和痛点，然后根据这些信息来撰写文案。例如，如果你的产品是一款面膜，你可以从用户的角度出发，写一篇关于如何选择适合自己的面膜的文章，告诉他们如何根据自己的肤质和需求来选择面膜。\n\n其次，小红书风格的电商营销文案要简洁明了，重点突出。我们要用简洁的语言来描述产品的特点和优势，让用户一眼就能看出产品的价值。例如，如果你的产品是一款口红，你可以写一篇关于如何选择适合自己肤色的口红的文章，告诉用户如何根据自己的肤色来选择口红的颜色。\n\n最后，小红书风格的电商营销文案要有趣味性和互动性。我们要通过一些有趣的内容和互动方式来吸引用户的注意力，让他们对产品产生兴趣。例如，你可以写一篇关于如何化妆的文章，然后在文章中加入一些互动环节，让用户参与进来，分享自己的化妆心得。\n\n总结一下，小红书风格的电商营销文案是以用户为中心的，简洁明了，重点突出，有趣味性和互动性。希望这些技巧能帮助到大家，让你的产品瞬间爆红！ | 大家好，我是小红书的电商营销专家。今天我要给大家分享一篇关于如何写好小红书风格的电商营销文案的技巧。\n\n首先，我们要注意的是，小红书的文案风格是简洁、直接、有趣的。我们可以使用一些幽默的语言和比喻来吸引读者的注意力。例如，我们可以说：“这款产品就像你的心头好，总是让你忍不住想要拥有。”或者，我们可以用一些形象的比喻来描述产品的特点，比如说：“这款产品就像一只可爱的小猫咪，总是让你心情愉悦。”\n\n其次，我们要注意的是，小红书的文案要有明确的卖点和卖点突出。我们可以在文案中强调产品的独特之处和优势，比如说：“这款产品的颜色独特，绝对是你衣柜里的一道亮丽风景线。”或者，我们可以强调产品的实用性和便捷性，比如说：“这款产品使用方便，总是能让你在忙碌的生活中轻松应对。”\n\n最后，我们要注意的是，小红书的文案要有明确的呼吁和引导。我们可以在文案中提供一些购买链接和促销活动的信息，比如说：“现在购买这款产品，立即享受10%的折扣哦！”或者，我们可以提供一些购买的步骤和提示，比如说：“点击链接，立即购买，快来拥有这款产品吧！”\n\n总的来说，小红书的电商营销文案要简洁、直接、有趣，同时要有明确的卖点和卖点突出，以及明确的呼吁和引导。希望这些建议对大家有所帮助。 |
| 请帮我撰写一份该品牌下这类商品如何选择的文章。\n模块必须按这个顺序输出：标题、正文；下面是这两个模块的要求：\n【标题要求】\n1、标题要体现出文章在描述该品类商品应该怎么选的主题；\n2、标题风格要看上去专业，让人信服；\n2、标题字数在30字内；\n【正文要求】\n1、字数在500字左右；\n【品牌】：美素佳儿（Friso）\n【商品类型】：婴幼儿奶粉\n | 标题：如何选择适合宝宝的婴幼儿奶粉品牌？\n\n正文：\n随着宝宝的成长，选择适合宝宝的奶粉品牌变得尤为重要。美素佳儿（Friso）作为婴幼儿奶粉的知名品牌，为家长们提供了多种选择。那么，如何选择适合宝宝的美素佳儿婴幼儿奶粉呢？\n\n首先，我们需要考虑宝宝的年龄和需求。美素佳儿的婴幼儿奶粉分为不同阶段，从0-6个月的婴儿奶粉到12个月以上的幼儿奶粉。根据宝宝的年龄和需求，选择适合的奶粉品牌和型号。\n\n其次，我们需要考虑奶粉的营养成分。美素佳儿的婴幼儿奶粉含有丰富的营养成分，如蛋白质、脂肪、维生素和矿物质等。选择奶粉时，要根据宝宝的年龄和需求，选择适合的营养成分。\n\n再次，我们需要考虑奶粉的口感和口感。美素佳儿的婴幼儿奶粉口感柔和，易于消化，适合宝宝的口感。选择奶粉时，要根据宝宝的口感，选择适合的奶粉。\n\n最后，我们需要考虑奶粉的价格和品牌信誉。美素佳儿的婴幼儿奶粉价格适中，品牌信誉良好。选择奶粉时，要根据自己的经济实力和对奶粉品牌的信任度，选择适合的奶粉。\n\n综上所述，选择适合宝宝的美素佳儿婴幼儿奶粉，需要考虑宝宝的年龄和需求、奶粉的营养成分、口感和价格等因素。希望以上建议能帮助到有需要的家长们。 | 标题】：美素佳儿婴幼儿奶粉选购指南：如何选择适合宝宝的奶粉\n\n【正文】：\n\n美素佳儿（Friso）作为一家知名的婴幼儿奶粉品牌，其产品深受消费者的喜爱。然而，面对众多的美素佳儿婴幼儿奶粉，如何选择适合宝宝的奶粉呢？以下是一些选购指南，希望能帮助到您。\n\n首先，我们需要了解宝宝的年龄和生长阶段。不同阶段的宝宝对营养的需求是不同的，因此选择适合宝宝的奶粉非常重要。一般来说，0-6个月的宝宝可以选择美素佳儿的婴儿配方奶粉，6个月以上的宝宝可以选择美素佳儿的幼儿配方奶粉。\n\n其次，我们需要关注奶粉的配方。美素佳儿的奶粉采用了科学的配方，含有丰富的营养成分，如蛋白质、脂肪、碳水化合物、维生素和矿物质等。在选择奶粉时，我们可以根据宝宝的年龄和生长阶段，选择适合的配方。\n\n再次，我们需要关注奶粉的成分。美素佳儿的奶粉采用了优质的原料，如脱脂乳粉、乳清蛋白、植物油等。这些成分能够提供宝宝所需的营养，同时避免了过多的糖分和脂肪。\n\n此外，我们还需要关注奶粉的口感和质地。美素佳儿的奶粉口感细腻，质地均匀，易于宝宝吞咽。这有助于宝宝更好地吸收奶粉中的营养成分。\n\n最后，我们还需要关注奶粉的价格。美素佳儿的奶粉价格相对较高，但其质量和口感都是有保证的。在选择奶粉时，我们可以根据自己的经济能力，选择适合自己的奶粉。\n\n总的来说，选择美素佳儿婴幼儿奶粉时，我们需要考虑宝宝的年龄和生长阶段，关注奶粉的配方、成分、口感和质地，以及价格等因素。只有这样，我们才能选择到最适合宝宝的奶粉。 | 【标题】：如何选择适合宝宝的美素佳儿婴幼儿奶粉\n\n【正文】：\n\n作为宝宝的妈妈或爸爸，我们都希望给宝宝提供最好的营养和健康。选择适合宝宝的奶粉是非常重要的。美素佳儿作为一家知名的婴幼儿奶粉品牌，提供了多种适合不同年龄段的奶粉。下面我们将为您介绍如何选择适合宝宝的美素佳儿婴幼儿奶粉。\n\n首先，我们需要了解宝宝的年龄和营养需求。美素佳儿的婴幼儿奶粉分为新生儿奶粉、0-6个月奶粉、6-12个月奶粉和12-36个月奶粉。每个阶段的奶粉都有不同的营养成分和配方，根据宝宝的年龄和营养需求选择相应的奶粉。\n\n其次，我们需要关注奶粉的成分和营养价值。美素佳儿的婴幼儿奶粉都采用了天然乳粉和乳制品原料，保证了奶粉的营养价值。同时，美素佳儿还注重奶粉的口感和易消化性，帮助宝宝更好地吸收营养。\n\n再者，我们需要考虑奶粉的安全性。美素佳儿的婴幼儿奶粉都经过严格的安全检测，确保奶粉的安全性。同时，美素佳儿还采用了先进的生产工艺和设备，确保奶粉的质量和稳定性。\n\n最后，我们需要考虑奶粉的价格和购买渠道。美素佳儿的婴幼儿奶粉在市场上有多种销售渠道，包括线上和线下。我们可以根据自己的需求和预算选择适合的购买渠道。\n\n总的来说，选择适合宝宝的美素佳儿婴幼儿奶粉需要考虑宝宝的年龄和营养需求、奶粉的成分和营养价值、安全性和价格等因素。通过了解这些信息，我们可以为宝宝提供最好的营养和健康。希望这篇文章能够帮助您选择适合宝宝的美素佳儿婴幼儿奶粉。 |
| 在给定的笔记本电脑上撰写一篇评论,联想Yoga 920                | 作为AI语言模型，我无法直接在笔记本电脑上撰写评论。但是，我可以根据您提供的信息，为您提供一些关于笔记本电脑的评论。 | 作为一个人工智能助手，我无法在给定的笔记本电脑上撰写评论，因为我没有实际的体验和感受。然而，我可以为您提供一些撰写笔记本电脑评论的建议。\n\n首先，您需要考虑笔记本电脑的性能和功能。您可以评估其处理器、内存、硬盘容量、显卡和电池续航能力。您还可以考虑其屏幕分辨率、显示效果和触摸屏功能。\n\n其次，您需要考虑笔记本电脑的外观和设计。您可以评估其重量、尺寸、颜色和材质。您还可以考虑其键盘和触摸板的舒适度和响应速度。\n\n最后，您需要考虑笔记本电脑的价格和性价比。您可以评估其价格与性能的平衡，以及与其他同类产品相比的竞争力。\n\n综上所述，撰写一篇笔记本电脑评论需要考虑多个方面，包括性能、功能、外观、设计、价格和性价比。希望这些建议能帮助您撰写一篇有价值的评论。 | 这款笔记本电脑的外观设计非常时尚，采用了高质量的材料，手感非常舒适。它的屏幕分辨率高，色彩鲜艳，显示效果非常出色。处理器性能强劲，能够轻松处理各种任务，运行速度快。内存和存储空间足够大，能够满足用户的需求。电池续航时间长，能够满足用户的长时间使用需求。总的来说，这款笔记本电脑是一款非常值得购买的产品。它的性能和外观设计都非常出色，能够满足用户的各种需求。无论是工作还是娱乐，它都能为用户带来极好的使用体验。 |
|                                                              |                                                              |                                                              |                                                              |

总结：

- qwen1.5-0.5b效果都不太好；

- qwen1.5-0.5b-base版本出现过很多和提问无关的内容,会出现内容重复现象；

- 就人格注入来讲，qwen1.5-0.5b-chat一点都注入不进去，但base版本好很多；

- qwen1.5-0.5b推理显存占用1080ti没问题；

- qwen1.5-7b效果最好，各项评测均领先，1.8b比0.5b效果好，模型越大效果越好，基本来说14b就可以走生产了；
- llama3-chinese微调效果并不好，会出现大量重现现象；

## 项目致谢

1. [LLAMA-Factory](https://github.com/hiyouga/LLaMA-Factory)为我们提供了训练框架；
1. [Qwen1.5-0.5b-base](https://www.modelscope.cn/models/qwen/Qwen1.5-0.5B/summary)，[Qwen1.5-0.5b-chat](https://www.modelscope.cn/models/qwen/Qwen1.5-0.5B-Chat/summary)，[Qwen1.5-1.8b-base](https://www.modelscope.cn/models/qwen/Qwen1.5-1.8B/)，[Qwen1.5-7b-base](https://www.modelscope.cn/models/qwen/Qwen1.5-7B/summary)，[Llama3-Chines-sft](https://modelscope.cn/models/zhuangxialie/Llama3_Chinese_Sft/files)在modelscope上下载；

## 免责声明

本项目相关资源仅供学术研究之用，严禁用于商业用途。使用涉及第三方代码的部分时，请严格遵循相应的开源协议。模型生成的内容受模型计算、随机性和量化精度损失等因素影响，本项目无法对其准确性作出保证。

## 使用许可

此存储库遵循[CC BY-NC-SA](https://creativecommons.org/licenses/by-nc-sa/4.0/) ，请参阅许可条款。


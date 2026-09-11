# Spark-X2.5-4B LoRA 微调评测

本目录记录 Spark-X2.5-4B 在固定文本任务测试集上的一次正式 LoRA 微调对比。结果用于实验评测与复现参考，不代表通用能力或生产质量认证。

- 基模：[XHToken/Spark-X2.5-4B](https://www.modelscope.cn/models/XHToken/Spark-X2.5-4B)
- 训练框架：[XHToken/LlamaFactory](https://github.com/XHToken/LlamaFactory)
- 方法：SFT + LoRA，rank=16、alpha=32、dropout=0.05；`enable_thinking=false`
- 训练：4 × A800 80GB，1 epoch，19,555 条训练样本；训练主体约 27.24 分钟
- 选择：按验证集 loss 选择 `checkpoint-600`，最佳验证 loss 为 1.055690646
- 测试：固定留出集 1,014 条，两个模型使用相同 prompt、模板、解码方式和任务长度预算

## 主要结果

| 指标 | Spark-X2.5-4B | Spark-X2.5-4B + LoRA v1 | 变化 |
|---|---:|---:|---:|
| 意图识别集合完全匹配准确率 | 70.93% | 90.70% | +19.77 个百分点 |
| 意图识别 Micro-F1 | 82.93% | 92.51% | +9.58 个百分点 |
| 商品抽取 Micro-F1 | 75.79% | 82.62% | +6.83 个百分点 |
| 商品抽取 Micro-Precision | 66.47% | 78.82% | +12.36 个百分点 |
| 商品抽取 Micro-Recall | 88.17% | 86.80% | -1.37 个百分点 |
| 平均生成长度 | 430.0 token | 177.8 token | -252.2 token |
| 生成上限命中 | 271/1,014 | 11/1,014 | 明显减少 |

微调后的主要收益是意图识别更准、商品抽取误抽减少，以及回答更容易在长度预算内结束。Spark-X2.5-4B 在商品召回、解释展开和部分文案重复控制上仍有优势。微调并没有全面超过基模：商品抽取召回略降，完全匹配率仍为 20%，SEO 重复 4-gram 均值由 10.56% 升至 35.68%，长标题样本中由 30.42% 升至 55.49%。

参考字符 F1 和重复 4-gram 只是文本行为指标，不能替代事实准确率、人工盲评或文案质量评审。长标题仅 6 条，短标题与小红书文案各 5 条；尚未完成独立人工盲评、系统性事实评估、外部通用基准或多随机种子复验。

## 文件

- [metrics.json](metrics.json)：按任务列出的完整指标。
- [examples.md](examples.md)：4 个安全示例的并排结果，包含西班牙语问候、商品抽取、场景应答和流量概念解释。
- [training_curves.png](training_curves.png)：训练与验证 loss 曲线。

为避免发布隐私、内部路径或未经审核的原始材料，本目录不包含训练/验证/测试数据、完整预测、日志、checkpoint、LoRA 权重或历史实验压缩包。历史 GitHub 模型（qwen1.5-1.8b、qwen1.5-7b、qwen2.5-7b、qwen3-8b、llama3-chinese-sft）仅作为既有微调结果背景，本次没有重新运行它们。

# 官方实现审计与工作版改动

对照版本：[UpstageAI/cl4kt@f08aa4b5adb8eabfa4306b64e43bf3ac92ff06a2](https://github.com/UpstageAI/cl4kt/tree/f08aa4b5adb8eabfa4306b64e43bf3ac92ff06a2)。`upstream/` 未修改。根目录模型从官方源码复制后做下列修复；本实验不等于未经修改的历史代码运行。

## 模型正确性

1. **全遮挡行的未来答案泄漏。** 原始 `individual_attention` 在第一行全部遮挡时仍对一整行相同有限负数做 softmax，产生均匀注意力；多层 knowledge retriever 可将这行读到的未来响应带入后续位置。最终 softmax 后乘有效遮罩，使全遮挡行注意力归零；`monotonic_attention` 同步修复。2 层、CPU、seed=1、长度 10 的对照中，反转第 5 位及之后的响应会使原程序较早预测最大改变 `0.00018259882926940918`；修复后为 0。`tests/test_model.py` 对当前及未来回答的变更做回归检查。
2. **硬负例编码层级。** 原实现每一层都重新读取原始硬负例 embedding，仅最后一层结果用于损失。工作版将上一层输出传到下一层，与正例分支一致。
3. **mask 交互索引碰撞。** 原实现预留 padding/mask，却仍以 `num_skills` 作为响应偏移；错误 mask token 与某个答对的真实技能共享索引。工作版以完整词表大小 `num_skills+2` 为偏移，embedding 总大小保持原值。测试验证真实技能及 mask 在两种响应下索引互异。
4. **CPU 与单学生推理。** 去除导入时全局默认 CUDA tensor 类型；`get_device()` 改为 `.device`；输出 `.squeeze()` 改为 `.squeeze(-1)`，避免 batch=1 丢失批维。

这些修复会影响数值与性能，因此不能声称与原始参数或论文数值完全等价。未使用 legacy 开关混入旧行为。

可单独重现注意力泄漏对照，固定模型权重且仅修改遮罩行为：

```sh
.venv/bin/python scripts/audit_causality.py --output results/audit-causality.json
```

该脚本在临时副本中适配 CPU，自动清理，不修改保留的官方源码。

## 数据增强

- 用局部 `np.random.RandomState(seed)` 生成置换，避免官方 `np.random.seed` 每次调用污染训练和后续折划分的全局状态。
- 随机裁剪/置换允许窗口结束在最后一个交互；移除原先要求窗口结束位置严格小于序列长度的循环。原循环会排除最后一个位置，且全长置换或极短序列可能无法结束。
- 根据实际非零技能生成有效位置遮罩，修复 crop 关闭或保留全长时把 padding 算进平均池化的问题。
- 保留官方 `seed=index` / `index+1` 的两个视图策略，因此同一学生在不同 epoch 中的增强视图固定。论文文字描述为随机增强；工作版没有扩展为每轮重新采样。
- 保留官方 BERT 风格的 mask 策略：被选中的 token 中 80% 用 mask、10% 随机 token、10% 原值。保留按相邻难度技能替换、以 skill_id 作为题目 token 的实现。

## 训练与记录

新的 `main.py` 不依赖 Accelerate/yacs/iopath，直接使用单张 GPU 或 CPU。改动包括：按折重置 Python/NumPy/PyTorch seed、独立确定学生划分、每折单独保存 checkpoint、pandas 2 兼容的 CSV 写法、按 10 次未提升早停、记录完整环境与输入哈希、测试预测保存和指标汇总。原程序为超过 10 次才停，且各折 checkpoint 共用同一目录并删除旧文件。

训练与验证划分使用独立随机生成器。最初的用户打乱和 KFold 构造沿用官方做法；官方增强会污染全局随机状态，故后续折内部验证集不能保证与未经修复程序完全相同。实际划分全部保存，可直接审计。

训练保持官方的 `shuffle=False`、左侧 padding、每学生末 100 次、`output[:,1:]` 筛选规则。对于不足 100 次的学生，该筛选会包含其第一个真实交互的预测；这里没有额外改成统一排除每个学生的首个真实交互。有效响应才进入 BCE 和 AUC/RMSE。

注意力模块仍按官方逻辑处理 padding：CL 输出用有效位置平均池化，attention 本身没有另加 key padding mask。保留此项避免进一步改变架构；其影响不应与“已修复所有 padding 行为”混同。

## 本次配置的来源

论文明确：hidden=64、末 100 次、Adam lr=.001、batch512、λ=.1、增强候选 .3/.5/.7、反转候选 .1/.5/1、按学生五折、训练内 10% 验证。论文没有公布 ASSIST09 最优增强组合。

本次选用官方 README 示例的 mask=.5/crop=.3/permute=.5/replace=.5，hard negative=1。层数2、头数8、dropout=.2、温度 .05、d_ff=1024、最终隐藏512、hard-negative logit weight=1、seed12405、最大300轮来自官方配置。未运行 243 个增强组合的全网格搜索；测试集未用于调参。所报告结果是这一预设组合的五折测量，不是最佳超参数搜索后的论文 Table 2 精确复刻。

# 其余五个数据集的 GPU7 实验协议

本轮对 Spanish、Statics、Algebra05、Bridge Algebra06、Slepemapy 分别运行学生级五折交叉验证，共 25 折，按上述顺序在**物理 GPU7 串行执行**。本文件说明运行方式、验证证据与结果口径，不记录尚未完成的五折成绩。

当前状态以 [suite.json](../results/gpu7/suite.json) 和 [suite.log](../results/gpu7/suite.log) 为准；逐数据集日志在 `results/gpu7/<数据名>-fivefold.log`。最终结果须等五折完整结束、产物核验通过后另行汇总。

## 运行与中断恢复

在项目根目录执行完整套件：

```sh
.venv/bin/python -u scripts/run_remaining_gpu7.py
```

已有本轮目录时，中断后使用：

```sh
.venv/bin/python -u scripts/run_remaining_gpu7.py --resume
```

不要在套件仍运行时再启动同一目录的第二个训练进程。`--resume` 跳过通过来源校验的完整数据集；单个数据集中已完成的折也会跳过，未完成的折从该折第 1 轮重跑，并非从中断 epoch 继续。恢复要求配置、数据哈希、核心源码、checkpoint 启动器和记录的环境一致；需要改变实验时应另选新目录。

全新重跑可指定另一个输出根目录，例如：

```sh
.venv/bin/python -u scripts/run_remaining_gpu7.py \
  --output-root results/gpu7-rerun
```

单个数据集也可独立运行，以下目录应为空或尚不存在：

```sh
.venv/bin/python -u scripts/train_gpu7.py \
  --config configs/spanish.yaml \
  --device cuda:0 \
  --output-dir results/gpu7-rerun/spanish-fivefold
```

对同一单数据集恢复时追加 `--resume`。其他配置名称为 `statics`、`algebra05`、`bridge_algebra06`、`sampled_slepemapy`。论文中的 algebra06 对应 Bridge to Algebra，不能替换成 Algebra I 2006–2007。

套件在每个训练任务前运行 CPU 数据预检，检查五折各分区的学生、有效标签、技能数与时间顺序。预检显式隐藏全部 GPU。训练子进程通过 [train_gpu7.py](../scripts/train_gpu7.py) 在导入 PyTorch 前强制设置：

```text
CUDA_DEVICE_ORDER=PCI_BUS_ID
CUDA_VISIBLE_DEVICES=7
```

因此程序中的逻辑 `cuda:0` 对应所绑定的物理 GPU7。启动器要求恰有一个可见 CUDA 设备，拒绝 CPU 回退和其他逻辑设备；同时记录设备 UUID、PCI 总线号，以及 `nvidia-smi --id=7` 的只读查询结果。外部预设的其他可见设备值会被覆盖。

## 显存优化与训练语义

本轮使用 activation checkpointing：前向时减少保留的中间激活，在反向时重算相应 Transformer block。它在运行时包装 `models.modules.CL4KTTransformerLayer.forward`，**相对本项目已经修复的工作版，不修改核心模型和训练源文件**；它也不意味着回退到未经修复的作者原始程序。

包装使用 `use_reentrant=False`、`preserve_rng_state=True`，仅在 `module.training` 且启用梯度时生效；验证与测试仍走原始前向。`query`、`key`、`values` 显式作为 checkpoint 的位置参数传入，使 PyTorch 能识别 CUDA 设备并保存该设备的随机数状态。仅将这些张量放在关键字参数中，会遗漏 CUDA dropout 的随机状态，不能保证重算梯度一致。

保持的训练设置包括：

- FP32，未启用 autocast、FP16 或 BF16。
- 训练 batch size 配置为 512，不通过减小批大小改变批内负例数量；尾批或训练学生不足 512 时，仍遵循原 DataLoader 的自然批大小。
- Adam、学习率 0.001、hidden size 64、序列长度 100、梯度裁剪 2.0。
- 每折种子为 `12405 + fold`，学生划分与原工作版一致；验证 AUC 选 checkpoint，连续 10 轮不提升早停，最多 300 轮。
- 原有损失、数据增强、评估和测试标签访问规则保持不变；测试集不参与调参或最佳 epoch 选择。

本轮冻结启动器 SHA256：

```text
83821a1d121ad3edeeb37f5463c6ef54d9713ebe7849825b2fb825eee348e61f
```

后续若修改此文件，已有实验应保留其快照与哈希，不能把不同启动器行为混入同一正式套件。

## 等价性与实际显存证据

[checkpoint-equivalence-gpu7.txt](../results/gpu7/checkpoint-equivalence-gpu7.txt) 记录了 CPU 与物理 GPU7 的两项测试，均通过。测试使用含 dropout 的完整 CL4KT 小模型，batch 2、hidden 16、两层，比较普通训练与 checkpoint 训练的：

| 比较项 | CPU | GPU7 |
| --- | --- | --- |
| 预测最大绝对差 | 0 | 0 |
| 对比损失差 | 0 | 0 |
| 总损失差 | 0 | 0 |
| 98 个参数梯度的最大绝对差 | 0 | 0 |
| 反向完成后的 CPU RNG 状态 | 完全一致 | 完全一致 |
| 反向完成后的 CUDA RNG 状态 | 不适用 | 完全一致 |

对应测试代码为 [test_checkpointing.py](../tests/test_checkpointing.py)。这是所记录环境与测试实例上的精确等价证据，不代表不同硬件、不同 PyTorch 版本或全部长训练的数值都必然逐位相同。

此外，使用真实固定 Slepemapy 数据、hidden 64、batch 配置 512，在 GPU7 完成了第 0 折的单 epoch smoke。其 [execution.json](../results/gpu7/smoke-slepemapy/execution.json) 记录：

| 显存项 | 字节 | GiB（2³⁰ 字节） |
| --- | ---: | ---: |
| PyTorch 峰值 allocated | 3,066,755,584 | 2.86 |
| PyTorch 峰值 reserved | 3,596,615,680 | 3.35 |

这是 PyTorch 分配器统计，不是整张显卡上所有进程的总占用。[smoke 配置](../results/gpu7/smoke-slepemapy/config.json) 明确将 epoch 上限改为 1；该目录仅验证真实批次可运行和显存需求，**不计入正式五折结果**。

## 运行记录与来源核验

每个正式数据集目录包含以下记录：

| 文件 | 作用 |
| --- | --- |
| `execution.json` | GPU7 绑定、PCI 顺序、FP32、batch 512、checkpoint 设置、启动器及配置哈希、命令；退出时补充状态和峰值显存 |
| `launcher/train_gpu7.py` | 实际运行的启动器快照，与记录的 SHA256 对照 |
| `environment.json`、`requirements.freeze.txt` | Python/PyTorch/CUDA 等版本、可见设备、数据哈希与依赖 |
| `config.json` | 已解析的实际模型与训练配置 |
| `source_sha256.json`、`source/` | 核心训练、模型、数据加载和增强源码的哈希及快照 |
| `splits.json`、`fold_N/split.json` | 五折的训练、验证、测试学生名单 |
| `fold_N/history.csv`、`best.pt` | 训练历史和按验证 AUC 选出的最佳 checkpoint |
| `fold_N/predictions.npz`、`metrics.json` | 最佳 checkpoint 的测试预测、标签与逐折指标 |
| `summary.json` | 本数据集完成情况及逐折均值、总体标准差 |

核心源码快照不直接包含运行时的包装行为，因此必须与 `execution.json` 和 `launcher/` 一起检查。完成数据集的套件恢复校验还对照当前数据、配置、启动器及核心源码，防止改动后沿用旧结果。训练中的 `execution.json` 尚未必有退出状态，不能据此当作已经完整完成。

完整五折完成后，可以在 CPU 上执行独立产物核验，例如：

```sh
CUDA_VISIBLE_DEVICES='' .venv/bin/python scripts/verify_run.py \
  --run-dir results/gpu7/spanish-fivefold
```

核验使用该次运行保存的源码快照。当前工作区源码变化会单独提示，不能因为后来调整了辅助脚本就抹去一个来源完整的历史实验；需要当前源码也一致时可添加 `--strict-current-source`。

## 正式范围与历史结果隔离

| 路径 | 设备与用途 | 本轮口径 |
| --- | --- | --- |
| `results/gpu7/<数据名>-fivefold` | 本轮 GPU7 正式五折 | 25 折正式实验的唯一来源 |
| `results/gpu7/smoke-slepemapy` | GPU7 单 epoch smoke | 仅作验证，不计成绩 |
| `results/spanish-fivefold`、`results/statics-fivefold` | 用户指定 GPU7 之前，在 GPU0/1 上启动后中断的实验 | 保留供追溯，排除正式新汇总；不续接其折结果 |
| `results/assistments09-fivefold` | 上轮已完成的 GPU4 ASSIST09 | 仅作历史比较，不归入“本轮全部在 GPU7 完成”的结论 |

最终若将历史 ASSIST09 与本轮五个数据集放入六数据集对比表，必须逐项标出运行设备及历史性质。该表表示固定配置的跨数据集比较，不表示六个数据集均在本轮 GPU7 重新运行。

## 数据版本与超参数解释

本轮输入在截断最近 100 次之前的统计如下。四个数据集按论文表 1 的统计口径匹配；Slepemapy 使用一次固定的新抽样。

| 数据集 | 学生 | 题目/练习 token | 表 1 技能口径 | 实际模型技能 token | 交互 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Algebra05 | 571 | 173,113 | 112 | 271 | 607,014 |
| Bridge Algebra06 | 1,138 | 129,263 | 493 | 550 | 1,817,450 |
| Spanish | 182 | 409 | 221 | 221 | 578,726 |
| Statics | 333 | 1,223 个公开练习 token | 1,223 | 1,223 | 189,297 |
| Slepemapy 新样本 | 5,000 | 2,713 | 1,380 | 1,380 | 623,212 |

Algebra 的原子技能与组合技能有不同口径。Statics 采用 AKT 作者公开的完整 333 学生序列；原匿名 ID、真实时间戳及独立问题元数据不可恢复，统计匹配并不能证明与 CL4KT 作者的文件或折分完全相同。Slepemapy 固定抽样种子为 12405，论文原学生列表和种子未公开；论文统计是 5,000/2,723/1,391/625,523，不能将当前样本称为精确复原。详细来源及哈希见 [Algebra 数据说明](DATA_ALGEBRA.md)、[Spanish/Statics 数据说明](DATA_SPANISH_STATICS.md)、[Slepemapy 数据说明](DATA_SLEPEMAPY.md)。

五个数据集统一使用官方 README 示例增强组合：`mask=0.5`、`crop=0.3`、`permute=0.5`、`replace=0.5`，硬负反转率 1.0、对比权重 0.1。其余模型通用值来自公开配置。论文没有给出逐数据集最终最优组合，本轮未执行逐数据集的完整增强网格搜索，因此结果应称为**公开固定配置下的五折复现实验**，不能声称已经重建作者未公开的最佳调参过程。此轮也不包含所有 baseline、消融或显著性检验。

当本轮五个数据集完整结束且历史 ASSIST09 产物仍通过核验后，可由 `scripts/summarize_datasets.py` 生成六数据集比较的 CSV、JSON 和图表。该辅助脚本在任务未完整结束时拒绝写正式汇总；最终报告应继续保留 GPU4 历史比较、Statics 版本与 Slepemapy 新抽样标记。运行状态始终查阅 `suite.json` 与日志，本文件不提供实时完成进度。

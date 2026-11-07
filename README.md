# CL4KT 论文复现

2026-11-05：其余五个数据集已在 **物理显卡 7** 上完成全部 **25 折**。连同上轮 ASSISTments09，六数据集 CL4KT 主模型五折实验均已完成。参见 [六数据集实验报告](docs/RESULTS_ALL.md)、[汇总 CSV](results/gpu7/all_datasets.csv) 和 [7 卡运行协议](docs/GPU7_EXPERIMENTS.md)。

基于 [UpstageAI/cl4kt](https://github.com/UpstageAI/cl4kt) 的官方实现，复现 Lee 等人的 *Contrastive Learning for Knowledge Tracing*（WWW 2022）。采用公开固定参数，按学生划分五折，并保存预测、模型、划分、配置与源码快照。

六数据集均通过完整产物独立核验。上轮 ASSIST09 使用 GPU4，其 AUC 为 **0.7598 ± 0.0072**、RMSE 为 **0.4328 ± 0.0029**（± 为折间总体标准差），历史明细见 [ASSIST09 报告](docs/RESULTS.md)。本轮新增结果和论文差值见 [完整报告](docs/RESULTS_ALL.md)；Slepemapy 使用一次固定种子的新抽样，Statics 使用公开完整学生序列版本，均有明确可比性说明。

官方源码保存在 `upstream/`，commit 为 `f08aa4b5adb8eabfa4306b64e43bf3ac92ff06a2`。根目录为修复后的工作版，改动见 [代码审计与复现差异](docs/CHANGES.md)。结果属于修复后实现的实验，不能当作原始官方程序逐位一致的运行结果。论文未公布逐数据集最佳增强组合，本次采用官方 README 示例的增强配置，未重建作者的完整调参过程。

## 环境

本机已创建 `.venv/`，使用 Python 3.12.12、PyTorch 2.6.0+cu124，并复用本机已有依赖；无需修改其他 conda 环境。重新安装时：

```sh
python3.12 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
```

PyTorch CUDA wheel 需与机器环境相适配。每次实验保存完整 `requirements.freeze.txt`、GPU/库版本、原始源码快照和哈希。浮点结果在不同硬件或库版本间可能变化。

## 数据与运行

六数据集均已完成预处理，来源、统计与限制见 [数据说明](docs/DATA.md)。全新重跑本轮新增的五个数据集（启动器强制物理 GPU7、FP32、batch 配置 512，启用保留 RNG 的 activation checkpointing）：

```sh
.venv/bin/python -u scripts/run_remaining_gpu7.py \
  --output-root results/gpu7-rerun
```

ASSIST09 数据为 3,695 名学生、17,728 道题、282,071 次交互；训练输入实际有 149 种技能组合。112 与 149 的统计差异见数据说明。从原始文件重新准备：

```sh
.venv/bin/python scripts/prepare_assistments09.py \
  --raw /path/to/skill_builder_data_corrected.csv
```

单独重跑 ASSIST09 完整五折：

```sh
CUDA_VISIBLE_DEVICES=7 sh scripts/train_assistments09.sh \
  --device cuda:0 --output-dir results/my-assistments09-fivefold
```

每折训练数据内另取 10% 学生验证；每名学生仅取末 100 次交互。Adam、学习率 0.001、batch 512、hidden 64、对比损失权重 0.1。最多 300 轮，验证 AUC 连续 10 轮不提升则早停。测试集只对验证集选定的检查点评分。完整配置见 [assistments09.yaml](configs/assistments09.yaml)。

短训练检查和 CPU 运行：

```sh
CUDA_VISIBLE_DEVICES=7 .venv/bin/python main.py \
  --device cuda:0 --epochs 1 --folds 0 --output-dir results/my-smoke
CUDA_VISIBLE_DEVICES='' .venv/bin/python main.py \
  --device cpu --epochs 1 --folds 0 --output-dir results/my-cpu-smoke
```

`--folds` 使用从 0 开始的编号。短训练的结果不代表论文五折结果。中断后可加 `--resume`，跳过已完成折、从头重跑未完成折；要求数据、配置和核心源码哈希不变，避免混合不同实验。新实验使用新输出目录。

## 实验产物

本轮五数据集输出为 `results/gpu7/<数据名>-fivefold/`，每个数据集的日志在 `results/gpu7/` 同名 `.log` 文件中；套件状态为 [suite.json](results/gpu7/suite.json)。历史 ASSIST09 输出为 `results/assistments09-fivefold/`。六数据集汇总为 [CSV](results/gpu7/all_datasets.csv) / [JSON](results/gpu7/all_datasets.json)。

| 文件 | 内容 |
| --- | --- |
| `summary.json` | 所有完成折的指标、均值和总体标准差，`complete` 表示是否完整五折 |
| `config.json` / `environment.json` | 实际配置、版本、GPU、命令与数据哈希 |
| `splits.json` | 全部学生级训练/验证/测试划分 |
| `source/` / `source_sha256.json` | 运行时核心代码快照与哈希 |
| `execution.json` / `launcher/train_gpu7.py` | 本轮新增实验的物理 GPU、精度、checkpoint 包装、退出状态及启动器快照 |
| `fold_N/history.csv` | 每轮训练损失和验证 AUC/RMSE |
| `fold_N/best.pt` | 验证 AUC 最佳模型、配置与词表大小 |
| `fold_N/predictions.npz` | 测试预测与真实标签，支持独立重算指标 |
| `fold_N/metrics.json` | 测试 AUC/RMSE/ACC、最佳轮数、样本数和模型哈希 |
| `fold_N/split.json` / `data_stats.json` | 每折学生划分和截断后的数据量 |

验证程序：

```sh
CUDA_VISIBLE_DEVICES='' .venv/bin/python -m unittest discover -s tests -v
.venv/bin/python scripts/verify_run.py --run-dir results/assistments09-fivefold
.venv/bin/python scripts/plot_results.py --run-dir results/assistments09-fivefold
```

阅读路线：[论文精读](docs/PAPER.md) → [数据来源](docs/DATA.md) → [实现差异](docs/CHANGES.md) → [六数据集实验报告](docs/RESULTS_ALL.md)。已完成范围为 CL4KT 主模型的六数据集五折实验；全部基线、增强/λ 消融、uniformity 和统计显著性检验不在本次完成范围内。

`configs/` 保存六个数据集本次使用的固定复现配置；它们采用公开通用参数和 README 增强组合，不是作者逐数据集最佳配置。验证记录见 [统一产物核验](results/gpu7/aggregate-verification.txt)、[回归测试](results/gpu7/regression-tests.txt) 和 [GPU7 checkpoint 精确等价检查](results/gpu7/checkpoint-equivalence-gpu7.txt)。

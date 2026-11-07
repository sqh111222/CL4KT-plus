# 两个 KDD Algebra 数据集

本次从 SIGKDD 官方存档服务器获取原始 KDD Cup 2010 development datasets，使用 **Algebra I 2005–2006** 和 **Bridge to Algebra 2006–2007** 的原始 `*_train.txt`。没有拼接测试文件、答案文件，也没有用其他论文的预划分序列代替原始数据。

## 获取与重建

来源说明见 [SIGKDD 数据页](https://kdd.org/kdd-cup/view/kdd-cup-2010-student-performance-evaluation/Data)。本次实际可用的官方直链是：

- [algebra_2005_2006.zip](https://kdd.org/cupfiles/algebra_2005_2006.zip)，22,442,608 字节。
- [bridge_to_algebra_2006_2007.zip](https://kdd.org/cupfiles/bridge_to_algebra_2006_2007.zip)，68,480,548 字节。

原 DataShop 页面在本次环境返回 403，而上述 SIGKDD 官方存档可正常下载。数据提供者没有在上述页面授予通用 MIT/CC 数据许可；引用与使用以 [KDD Cup 数据与竞赛规则](https://kdd.org/kdd-cup/view/kdd-cup-2010-student-performance-evaluation/Rules) 及原数据提供者规定为准。

在项目根目录运行：

```sh
.venv/bin/python scripts/prepare_algebra.py --data-name algebra05
.venv/bin/python scripts/prepare_algebra.py --data-name bridge_algebra06
```

脚本自动下载或复用对应数据目录中的官方 ZIP；也可通过 `--raw /path/to/archive.zip` 或 `--raw /path/to/*_train.txt` 提供已获取原始文件。`--output-root` 可以指定另一个数据根目录，避免覆盖当前实验。若已有 `data.txt` 与输入内容不同，脚本拒绝覆盖。

预处理直接调用未修改的 `preprocess_data.prepare_kddcup10`，官方 commit 为 `f08aa4b5adb8eabfa4306b64e43bf3ac92ff06a2`。脚本固定 `PYTHONHASHSEED=0` 后重新启动自身，使官方 `set` 枚举原子技能时的列顺序可重复。采用原版稠密 Q 矩阵及 `np.unique` 实现，没有替换为另一种近似算法。

## 名称与预处理规则

| 论文名称 | 训练/预处理数据名称 | 原始 ZIP 成员 | 技能列 |
| --- | --- | --- | --- |
| algebra05 | `algebra05` | `algebra_2005_2006_train.txt` | `KC(Default)` |
| algebra06 | `bridge_algebra06` | `bridge_to_algebra_2006_2007_train.txt` | `KC(SubSkills)` |

论文中的 algebra06 指 **Bridge to Algebra**，不是同样存在于下载目录中的 Algebra I 2006–2007。两者不能互换。

官方预处理先用 `Problem Name + ':' + Step Name` 定义题目，把 `First Transaction Time` 转为相对秒数；随后仅保留二元 `Correct First Attempt`、删除没有命名技能的交互，按学生/题目/时间戳去重，再删除不足 5 次交互的学生。`~~` 分隔多个原子技能；Q 矩阵中每个独特的技能组合成为模型使用的新技能。最后先按时间排序，再按学生聚合，ID 从 0 开始。训练入口另加 1 为 padding 留出 0，并保留每学生最近 100 次交互。

论文表 1 的目标统计为：

| 数据集 | 学生 | 题目 | 原子技能 | 交互 |
| --- | ---: | ---: | ---: | ---: |
| algebra05 | 571 | 173,113 | 112 | 607,014 |
| algebra06 | 1,138 | 129,263 | 493 | 1,817,450 |

`manifest.json` 同时记录每个过滤阶段的实际统计，以及最终的原子技能数和组合技能数。论文表中的技能数是原子技能口径；`preprocessed_df.csv` 中的 `skill_id` 则是技能组合 ID。不要直接用论文表的原子技能数创建 embedding 词表；训练应从预处理表读取实际最大 ID。

本次两份数据的学生、题目、原子技能和交互数均与上表精确一致，两个清单均记录 `matches_paper_table1: true`。Algebra05 实际模型组合技能数为 **271**，Bridge 为 **550**；原子技能分别仍是论文表中的 112 和 493。最短学生序列分别为 7 和 5。原子与组合两种计数都保留在清单中。

处理后已独立检查二元标签、无缺失值、每学生的时间顺序、Q 矩阵维度，以及每一条交互的组合技能 ID 与 Q 矩阵行的一致性，均通过。

## 实验产物与审计

两个数据目录各保留原始 ZIP、选定成员的 `data.txt`、官方生成的 `preprocessed_df.csv`、`original_df.csv`、`q_mat.npz`、`question_skill_rel.pkl` 和新增的 `manifest.json`。清单包含：

- 输入存档及解压训练文件的 SHA256、官方来源 URL 和实际成员名称。
- 是否由该次脚本调用执行下载；复用已有存档会明确记录为未重新下载。
- 官方预处理源码 SHA256、环境版本、哈希种子及过滤参数。
- 各处理阶段和最终统计、与论文表 1 的逐项一致性，以及所有输出哈希。

`prepare.log` 保留本次实际运行的标准输出/错误，原始文件和预处理统计为实验结果可比性的依据。

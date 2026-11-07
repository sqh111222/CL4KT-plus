# 数据来源与预处理

主实验使用原始 **ASSISTments 2009–2010 Skill Builder corrected.csv 展开版本**，严格保留官方 CL4KT 的预处理顺序。论文表 1 的 ASSIST09 学生数、题数和交互数可由此版本精确得到。

## 2026-11-05：其余五个数据集已准备

| 数据集 | 学生 | 题目/token | 模型技能组合 | 交互 | 与论文数据统计的关系 |
| --- | ---: | ---: | ---: | ---: | --- |
| algebra05 | 571 | 173,113 | 271 | 607,014 | 表 1 精确匹配；表中 112 为原子技能 |
| algebra06 / bridge_algebra06 | 1,138 | 129,263 | 550 | 1,817,450 | 表 1 精确匹配；表中 493 为原子技能 |
| Spanish | 182 | 409 | 221 | 578,726 | 表 1 精确匹配 |
| Statics | 333 | 1,223 | 1,223 | 189,297 | AKT 作者完整学生序列版本；表 1 对题目数写 `—` |
| Slepemapy | 5,000 | 2,713 | 1,380 | 623,212 | 固定种子 12405 的新样本；作者名单与种子未公开 |

来源、哈希、预处理和可比性边界分别见 [KDD Algebra 数据](DATA_ALGEBRA.md)、[Spanish/Statics 数据](DATA_SPANISH_STATICS.md) 和 [Slepemapy 抽样](DATA_SLEPEMAPY.md)。这五个数据集均已通过所有五折训练/验证/测试分区的非空、双类别、时间顺序和 loader 条件预检。以下 ASSISTments09 内容保留此前实验的完整处理说明。

## 准备 ASSISTments09

已有原始 CSV 时运行：

```sh
.venv/bin/python scripts/prepare_assistments09.py \
  --raw /path/to/skill_builder_data_corrected.csv
```

也支持 ZIP；ZIP 中只有一个 CSV 时自动选取，多个 CSV 时用 `--zip-member` 指定成员的完整名称。脚本只读取选定成员，不把压缩包内的路径展开到磁盘。

```sh
.venv/bin/python scripts/prepare_assistments09.py \
  --raw /path/to/2009_skill_builder_data_corrected.zip \
  --zip-member skill_builder_data_corrected.csv
```

公开镜像（2026-11-04 已核验 HTTP 200，ZIP 大小 9,084,422 字节）：

```sh
curl -fL https://base.ustc.edu.cn/data/ASSISTment/2009_skill_builder_data_corrected.zip \
  -o /tmp/2009_skill_builder_data_corrected.zip
.venv/bin/python scripts/prepare_assistments09.py \
  --raw /tmp/2009_skill_builder_data_corrected.zip
```

来源说明以 [ASSISTments 官方数据页](https://sites.google.com/site/assistmentsdata/home/2009-2010-assistment-data/skill-builder-data-2009-2010) 为准。该页还提供后续发布的 collapsed 版本，每次学生作答仅一行、多技能用 `skill1_skill2` 表示。该版本与本实验使用的展开版不同，不能默默替换后继续声称使用相同数据。本脚本会记录实际输入路径和哈希，不将镜像网址冒充为本次下载记录。官方数据页要求在论文中引用该数据页及 Feng 等 2009 的 ASSISTments 系统论文；页面未给出明确的独立开放许可。

默认输出 `dataset/assistments09/`，包含官方预处理生成的制表符分隔 `preprocessed_df.csv`、`q_mat.npz` 等，以及新增的 `manifest.json`。清单记录原始文件及输出文件的 SHA256、每步样本数、预处理源码哈希与官方 commit。若已有 `data.csv`，仅当内容哈希一致才复用，避免覆盖其他版本。可通过 `--output-dir` 另存实验。为触发官方 ASSIST09 分支，输出目录名称需为 `assistments09`。

本机本次使用的公开原始 CSV 为 `/data/sqh/KT/AKT/data/2009_skill_builder_data_corrected/skill_builder_data_corrected.csv`，其 SHA256 是：

```text
1aa296e00b6c88c4d6fad4ca2ae4866484d9fe5484f38f5c8c94dfc49f045e08
```

## 必须说明的统计口径

| 处理阶段 | 行数 | 学生数 | 题目数 | 原子技能数 |
| --- | ---: | ---: | ---: | ---: |
| 原始 corrected.csv | 401,756 | 4,217 | 26,688 | 123 |
| 仅保留二元正确性 | 401,756 | 4,217 | 26,688 | 123 |
| 删除缺失技能的行 | 338,001 | 4,163 | 17,751 | 123 |
| 按此时行数删除少于 5 次交互的学生 | 336,886 | 3,695 | 17,728 | 123 |
| 按 `order_id` 去重 | 282,071 | 3,695 | 17,728 | 112 |

原始数据有 54,896 条重复 `order_id` 行；最短序列过滤后还剩 54,815 条待去重行。官方实现先建立题目到原子技能的 Q 矩阵，再按 `order_id` 去重并打印技能数，随后将每一种不同的 Q 矩阵行映射为一个新技能。因此：

- 论文的 **112** 与官方去重后打印的原子技能数一致；训练真正使用的是 **149 种技能组合**，不是 112。共有 3,017 道题的 Q 矩阵行包含多个原子技能。作者没有解释表格与实际训练词表的差异；这里给出的是本次代码与数据核对结果。
- 官方先按含重复行的序列过滤，再去重。最终有 **51 名学生少于 5 次独立交互**，最短为 2。本复现保留这一顺序，以便与官方实现比较。把过滤移到去重后会得到不同人群。
- 模型训练阶段按学生选择最近 100 次交互；这一截断不在预处理统计表中。全部原始交互数不能当作模型实际训练条数。
- 输出 `skill_id` 是从 0 开始的组合技能 ID。训练入口再加 1，为 padding 留出 0；不要在准备数据时重复偏移。

## 六个数据集的获取与可比性

| 论文名称 / 官方代码名称 | 公开来源 | 官方预处理所需文件 | 复现限制 |
| --- | --- | --- | --- |
| assist09 / `assistments09` | [ASSISTments 官方页](https://sites.google.com/site/assistmentsdata/home/2009-2010-assistment-data/skill-builder-data-2009-2010)，[corrected ZIP 镜像](https://base.ustc.edu.cn/data/ASSISTment/2009_skill_builder_data_corrected.zip) | `data.csv`，需有 `order_id,user_id,problem_id,skill_id,skill_name,correct` | 当前主实验；必须记录展开版/合并版差异 |
| algebra05 / `algebra05` | [SIGKDD 官方存档](https://kdd.org/cupfiles/algebra_2005_2006.zip) | `data.txt`，Algebra I 2005–2006，`KC(Default)` 列 | 已下载原始 train TXT 并处理，精确匹配表 1 |
| algebra06 / `bridge_algebra06` | [SIGKDD 官方存档](https://kdd.org/cupfiles/bridge_to_algebra_2006_2007.zip) | `data.txt`，Bridge to Algebra 2006–2007，`KC(SubSkills)` 列 | 已处理；论文简称 algebra06，官方入口写 `bridge_algebra06` |
| slepemapy / `sampled_slepemapy` | [Masaryk 大学原始目录](https://www.fi.muni.cz/adaptivelearning/data/slepemapy/)，[2015-05-21.zip](https://www.fi.muni.cz/adaptivelearning/data/slepemapy/2015-05-21.zip) | 分号分隔 `answer.csv` | [数据论文](https://learning-analytics.info/index.php/JLA/article/download/4462/5602/22773) 指明 ODbL 1.0；官方随机抽取 5,000 学生但未固定抽样种子，论文那批学生未发布，难以精确复原 625,523 条记录 |
| spanish / `spanish` | 数据作者的 [WCRP 仓库](https://github.com/robert-lindsey/WCRP/tree/master/datasets) | `spanish_dataset.txt` 与 `spanish_expert_labels.txt` | 三列分别为从 0 开始的学生、题目、二元正确性；顺序即时间顺序。仓库代码 MIT；README 未单列数据许可 |
| statics / `statics` | [AKT 作者完整序列](https://github.com/arghosh/AKT/blob/01b7e8d7d7c862cabc9467108e3e7f2555653a88/data/statics/statics.csv) | 由完整333名学生序列构建 `interaction_df.csv`，再调用官方减 1 规则 | 已匹配333/1223/189297；原匿名ID和时间戳未包含，见专项说明 |

本机已有原始 `answer.csv`：`/data/sqh/KT/public dataset/Slepemapy.cz/answer.csv`，本次已复制到工作区并固定抽样。Statics 此次找到的 AKT 作者完整文件有明确的 333 名完整学生边界，区别于预划分或分窗口文件；处理时逐学生、逐交互验证顺序和响应不变。

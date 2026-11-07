# Spanish 与 Statics 数据版本和处理

两个数据集均保存了固定版本的公开作者文件，准备脚本为 `scripts/prepare_spanish_statics.py`。脚本先验证来源文件的 SHA256，再调用根目录中 CL4KT 的官方预处理函数；额外的格式转换、逐步统计、源码和输出哈希写入各数据目录的 `manifest.json`。训练时仍由原有数据加载器选取每名学生最后 100 次交互，预处理表不提前截断。

```sh
.venv/bin/python scripts/prepare_spanish_statics.py --dataset all --download
```

可将 `all` 改为 `spanish` 或 `statics`。`--download` 只下载缺失文件，已有文件必须与固定作者版本的哈希一致；不允许用不同文件静默替代。脚本不会使用原作者的预划分训练/测试文件，五折由本项目训练入口重新按学生建立。

## Spanish：原作者响应与专家标签

来源是 Lindsey 等 *Automatic Discovery of Cognitive Skills to Improve the Prediction of Student Learning* 的 [WCRP 作者仓库](https://github.com/robert-lindsey/WCRP)，固定 commit `7e46c02d430b5fdc7a13dda42cc965b87f640346`。

| 原始文件 | 大小（字节） | SHA256 |
| --- | ---: | --- |
| `spanish_dataset.txt` | 5,320,569 | `47aec7f0d52d11e87d9e8e1415a7d41e02e2219075fb3b0e4d710bc011af1769` |
| `spanish_expert_labels.txt` | 1,359 | `85aaf66ad12bd315a81ce96a4d824d0bfcf7dd943d1be6bffd3fda070cb60d35` |

固定文件链接：[响应记录](https://raw.githubusercontent.com/robert-lindsey/WCRP/7e46c02d430b5fdc7a13dda42cc965b87f640346/datasets/spanish_dataset.txt)、[专家标签](https://raw.githubusercontent.com/robert-lindsey/WCRP/7e46c02d430b5fdc7a13dda42cc965b87f640346/datasets/spanish_expert_labels.txt)。[作者 README](https://github.com/robert-lindsey/WCRP/blob/7e46c02d430b5fdc7a13dda42cc965b87f640346/README.md) 规定，响应文件三列依次为学生 ID、题目 ID、二元正确性，学生与题目 ID 从 0 开始，学生内部记录由早到晚排列。专家标签第 i 行是第 i 道题的技能 ID。

处理步骤：

1. 读取 578,726 行响应，共 182 名学生、409 道题；每名学生历史最短 226 次。
2. 直接按题目索引连接原作者专家标签，共 221 个技能。没有重新聚类、合并或人为修改标签。
3. 调用官方 `prepare_spanish()`，生成零时间戳，保留学生内部原始顺序，按学生分组，输出 `preprocessed_df.csv` 和 Q 矩阵。
4. 核对每位学生的题目、响应、技能序列与输入逐项相同，核对所有 ID 连续且从 0 开始。

| 阶段 | 交互数 | 学生数 | 题目数 | 技能数 |
| --- | ---: | ---: | ---: | ---: |
| 原作者响应 | 578,726 | 182 | 409 | 尚未连接标签 |
| 连接专家标签 | 578,726 | 182 | 409 | 221 |
| CL4KT 官方输出 | 578,726 | 182 | 409 | 221 |
| 论文表 1 | 578,726 | 182 | 409 | 221 |

原文件没有真实时间戳；输出 `timestamp=0` 是官方占位规则，不能解释为同时发生的交互。仓库以 MIT 发布代码，README 未单独给出数据许可；研究使用应引用原作者论文。

## Statics：AKT 作者公开的完整学生序列

论文表 1 目标为 333 名学生、1,223 个练习 token、189,297 次交互。此前调查的 `learner-performance-prediction` 公开标准表声明 282 名学生及 98 个知识点，无法作为相同版本静默替代。

本次找到 [AKT 作者仓库](https://github.com/arghosh/AKT) 中的完整 `data/statics/statics.csv`，固定 commit `01b7e8d7d7c862cabc9467108e3e7f2555653a88`。使用的是**完整文件**，不是各折 `train/test/valid` 文件，也没有通过拼接窗口猜测学生身份。

- 固定来源：[statics.csv](https://raw.githubusercontent.com/arghosh/AKT/01b7e8d7d7c862cabc9467108e3e7f2555653a88/data/statics/statics.csv)。
- 本地文件：`dataset/statics/statics_sequences.csv`。
- 大小：1,131,524 字节。
- SHA256：`cd8b6ed640bb2e3dfd63ec681e17ec2262fb7284cea8735d1cfc2f317d17bab1`。

该文件恰有 999 行，每连续三行是一名完整学生：第一行为交互数，第二行为练习 token 序列，第三行为正确性序列。逐记录验证第一行长度与两条实际序列一致，所有响应均为 0/1。333 条学生历史合计 189,297 次交互，最短 5 次、最长 1,181 次；练习 token 连续覆盖 `1..1223`。

适配 CL4KT 的步骤：

1. 以完整三行记录的索引 `0..332` 分配本地 `user_id`，不拼接、切割或重排学生历史。
2. 对每条原始交互写一行 `interaction_df.csv`，`item_id` 与 `skill_id` 都使用原文件的练习 token，保持 `1..1223`；`timestamp=0` 仅占位。
3. 调用官方 `prepare_statics()`，由它将题目及技能 ID 减 1，输出 `0..1222` 的 `preprocessed_df.csv` 和身份 Q 矩阵。
4. 逐行验证输出只发生了规定的 ID 减 1，学生边界、练习顺序和响应均与完整输入一致。

| 阶段 | 交互数 | 学生数 | 练习/token 数 | 模型技能数 |
| --- | ---: | ---: | ---: | ---: |
| AKT 完整学生序列 | 189,297 | 333 | 1,223 | 同一套练习 token |
| 一基制 CL4KT 输入表 | 189,297 | 333 | 1,223 | 1,223 |
| CL4KT 官方零基制输出 | 189,297 | 333 | 1,223 | 1,223 |
| 论文表 1 | 189,297 | 333 | 题目列为 `—` | 1,223 |

必须保留以下可比性边界：

- 这是 **AKT 作者公开的完整序列版本**，不是从 DataShop 下载的原始事件导出。[DataShop 数据集 507](https://pslcdatashop.web.cmu.edu/DatasetInfo?datasetId=507) 在本次网页访问中返回 403，没有绕过其访问限制。
- 原匿名学生标识与真实时间戳不在序列文件中，无法还原。学生索引仅表示公开文件的记录顺序；不能声称与 CL4KT 作者随机折分中的编号相同。
- 文件只提供一套练习 token；`item_id` 是同一 token 的接口字段，不是额外恢复出的真实题目身份。它不是 98 个认知知识点标签的版本。CL4KT 主模型实际使用 `skill_id`，因此这里保留 1,223 个练习 token 作为模型输入，与论文表 1 的口径一致。
- 统计精确匹配说明可建立同规模的公开序列复现，不足以证明 CL4KT 作者使用了字节完全相同的文件或相同折分。报告实验时应将其命名为“AKT 公开完整序列版”。

两种数据的原始文件、官方函数名及源码哈希、转换器哈希、逐步统计、输出文件哈希都可在各自 `manifest.json` 中检查；实际模型结果另见运行目录，不应由数据统计推断。

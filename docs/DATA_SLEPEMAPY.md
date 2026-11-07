# Slepemapy 固定抽样数据

使用 [Masaryk 大学公开的 2015-05-21 版本](https://www.fi.muni.cz/adaptivelearning/data/slepemapy/2015-05-21.zip)；本次原始文件来自本机已有公开 `answer.csv`，并在工作区保留副本及 SHA256。数据使用 ODbL 1.0，单条内容使用 DbCL 1.0，原始 README 要求引用 Papoušek、Pelánek 与 Stanislav 的数据论文。

官方 CL4KT 先移除 `place_answered` 缺失的回答，保留至少 5 次有效交互的学生，将按原始学生 ID 排序的学生组随机打乱后取前 5,000 名。题目由 `2*(place_asked-1)+type-1` 定义，技能由 `place_asked-1` 定义；正确性为 asked 与 answered 相等，序列按时间排序。

论文没有公开其 5,000 名学生的名单或抽样种子。这里明确固定抽样种子 **12405**，抽取一次并保存名单，以此运行全部五折。

| 统计项 | 本次固定抽样 | 论文表 1 |
| --- | ---: | ---: |
| 学生 | 5,000 | 5,000 |
| 题目 | 2,713 | 2,723 |
| 技能 | 1,380 | 1,391 |
| 交互 | 623,212 | 625,523 |

这是同一公开来源、同一抽样规则下的可重复新样本，不能称为作者原始样本的精确重建；相关测试指标仅作带此限制的参考对比。没有通过反复抽样来凑论文统计。

```sh
.venv/bin/python scripts/prepare_slepemapy.py \
  --raw '/data/sqh/KT/public dataset/Slepemapy.cz/answer.csv' --seed 12405
```

输出位于 `dataset/sampled_slepemapy/`：

- `answer.csv`：原始数据副本。
- `student_mapping.csv`：抽样顺序、新学生 ID、原始匿名 ID 和交互数。
- `answer_mapping.csv`：输出行与原始回答 ID 的对应。
- `preprocessed_df.csv`、`q_mat.npz`：模型输入和题目技能映射。
- `manifest.json`：原始来源、抽样参数、统计、处理代码及输出哈希。

准备脚本先选学生再解析其时间戳，避免对所有未选学生执行 Python 逐行日期转换。已使用 5,003 名学生的语义等价测试，对缺失回答、短序列、乱序输入及并列时间戳做核对：相同种子下，选中学生的名单与顺序、最终全部列值和 Q 矩阵均与未修改的官方 `prepare_sampled_slepemapy` 完全一致。该测试只使用临时构造数据，正式训练输入均为上述公开真实数据。

```sh
CUDA_VISIBLE_DEVICES='' .venv/bin/python -m unittest discover \
  -s tests -p test_slepemapy_preparation.py -v
```

时间戳按 UTC 解释以消除运行主机时区差异；对所有记录的统一时间偏移不改变序列顺序。训练仍按学生末 100 次截断，原始交互统计不等于实际参与训练和评分的数量。

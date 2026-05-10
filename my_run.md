# my_run 使用说明

## 文档导航

- 图像级训练与标签重采样：本文档
- 病人级训练说明：[`patient_level_training.md`](./patient_level_training.md)

---

## 1. 通过 JSON 人工控制 0/1 类比例

### 脚本位置

`utils/build_binary_resampled_labels.py`

### 作用

从旧标签文件中读取指定二分类标签，按目标数量生成新的标签文件：

- class 0 数量由 `--num_class0` 控制
- class 1 数量由 `--num_class1` 控制
- 当目标数量小于原始数量时，随机下采样
- 当目标数量大于原始数量时，有放回采样，相当于复制样本
- 非 0/1 的标签值（例如 `-1`）不会写入新文件

### 基本用法

```bash
python utils/build_binary_resampled_labels.py \
  --input_json /mnt/wangbd8/workspace/DataSets/ThyroidAgent/Classifaction_Data/Malignant_ultrasound_images_cropped/train_labels.json \
  --output_json /mnt/wangbd8/workspace/DataSets/ThyroidAgent/Classifaction_Data/Malignant_ultrasound_images_cropped/train_labels_ftcptc_1000_1000.json \
  --target_key FTCPTC \
  --num_class0 1000 \
  --num_class1 1000 \
  --seed 1024
```

### 示例

#### 示例 1：生成 FTCPTC 训练标签，0 类 2000，1 类 1000

```bash
python utils/build_binary_resampled_labels.py \
  --input_json train_labels.json \
  --output_json train_labels_ftcptc_2000_1000.json \
  --target_key FTCPTC \
  --num_class0 2000 \
  --num_class1 1000
```

#### 示例 2：生成 FTCPTC 训练标签，0 类 2000，1 类 1500

```bash
python utils/build_binary_resampled_labels.py \
  --input_json train_labels.json \
  --output_json train_labels_ftcptc_2000_1500.json \
  --target_key FTCPTC \
  --num_class0 2000 \
  --num_class1 1500
```

### 说明

- 如果 `num_class1` 大于原始正类数量，脚本会自动复制正类样本
- 输出文件可以直接替换训练脚本中的 `TRAIN_LABEL_PATH`

---

## 2. 图像级二分类多任务训练

### 训练代码

`train_multitask_binary_sampler.py`

### 重要说明

- 文件名里的 `sampler` 是历史遗留命名
- 当前代码里并没有 `WeightedRandomSampler`
- 训练使用普通 `DataLoader(shuffle=True)`
- 如果你想控制 0/1 类比例，应主要依赖新的 train_labels JSON

### 核心参数

- `--train_label_path`：训练标签 JSON 文件
- `--target_key`：关注的二分类标签，例如 `FTCPTC`
- `--task_schedule`：多任务优化顺序，例如 `seg,cls` 或 `cls,cls,seg`
- `--cls_pos_weight`：分类 BCE 中正类损失权重，值越大越偏向提升 recall

### 当前可直接使用的脚本

- `scripts/binary/train_binary_sampler_FTCPTC_segcls.sh`
- `scripts/binary/train_binary_sampler_FTCPTC_clsclsseg_pf02_ns6000.sh`
- `scripts/binary/train_binary_sampler_FTCPTC.sh`

### 2.1 `seg,cls` 版本

#### 脚本位置

`scripts/binary/train_binary_sampler_FTCPTC_segcls.sh`

#### 含义

- `task_schedule="seg,cls"`
- 每个优化 step 做一次分割、一次分类
- 图像级训练
- 不做 `WeightedRandomSampler`

#### 运行方式

```bash
bash scripts/binary/train_binary_sampler_FTCPTC_segcls.sh
```

#### 如果要改训练标签文件

把脚本中的 `TRAIN_LABEL_PATH` 改成新的 JSON，例如：

```bash
TRAIN_LABEL_PATH="/mnt/wangbd8/workspace/DataSets/ThyroidAgent/Classifaction_Data/Malignant_ultrasound_images_cropped/train_labels_ftcptc_2000_1000.json"
```

#### 如果要改 recall 倾向

优先改：

```bash
CLS_POS_WEIGHT
```

#### 经验

- 想更强调 recall：可尝试调大 `CLS_POS_WEIGHT`
- 如果已经通过新 JSON 做了较强重采样，建议先把 `CLS_POS_WEIGHT` 控制在较保守范围

---

## 3. 不依赖新 JSON 的默认训练方式

### 图像级默认脚本

- `scripts/binary/train_binary_sampler_FTCPTC.sh`

### 说明

- 如果 `TRAIN_LABEL_PATH` 仍指向原始 `train_labels.json`，那么训练分布就是原始分布
- 如果 `TRAIN_LABEL_PATH` 换成你生成的新 JSON，那么训练分布就由新 JSON 决定
- 病人级默认训练方式见 [`patient_level_training.md`](./patient_level_training.md)

---

## 4. 推荐实验顺序

### 方案 A：先只测试人工重采样 JSON 的效果（图像级）

1. 用 `utils/build_binary_resampled_labels.py` 生成新 JSON
2. 修改 `scripts/binary/train_binary_sampler_FTCPTC_segcls.sh` 中的 `TRAIN_LABEL_PATH`
3. 运行：

```bash
bash scripts/binary/train_binary_sampler_FTCPTC_segcls.sh
```

### 方案 B：人工重采样 JSON + 更偏分类的图像级训练

1. 用 `utils/build_binary_resampled_labels.py` 生成新 JSON
2. 修改 `scripts/binary/train_binary_sampler_FTCPTC_clsclsseg_pf02_ns6000.sh` 中的 `TRAIN_LABEL_PATH`
3. 根据需要调整 `CLS_POS_WEIGHT`
4. 运行：

```bash
bash scripts/binary/train_binary_sampler_FTCPTC_clsclsseg_pf02_ns6000.sh
```

### 方案 C：病人级训练

病人级 BCE 训练的完整说明见 [`patient_level_training.md`](./patient_level_training.md)。

---

## 5. 参数含义速查

### `TRAIN_LABEL_PATH`

训练时使用的标签 JSON 文件路径。

### `TARGET_KEY` / `--target_key`

当前关注的二分类标签，例如 `FTCPTC` 或 `LNM_CN01`。

### `CLS_POS_WEIGHT`

分类 BCE loss 中正类样本权重。

例如 `3` 表示正类错分的代价更高，训练时更偏向提升 recall。

### `TASK_SCHEDULE`

每个 optimizer step 的任务顺序：

- `seg,cls`：一次分割 + 一次分类
- `cls,cls,seg`：两次分类 + 一次分割

> 病人级训练特有参数，如 `PATIENT_ID_DEPTH`、`MAX_IMAGES_PER_PATIENT`、`CLS_POOLING`、`FIXED_THRESHOLD`，见 [`patient_level_training.md`](./patient_level_training.md)。

---

## 6. 当前建议

如果当前目标是提升 `FTCPTC` 的 recall，可按下面顺序试：

- 先生成新的 train_labels JSON
- 先跑图像级版本：`scripts/binary/train_binary_sampler_FTCPTC_segcls.sh`
- 如果 recall 仍然偏低，再试：`scripts/binary/train_binary_sampler_FTCPTC_clsclsseg_pf02_ns6000.sh`
- 如果你更关心最终病人级判断，再看并运行 [`patient_level_training.md`](./patient_level_training.md) 中的方案

一个可作为起点的比例示例：

- `num_class0 = 2000`
- `num_class1 = 1000` 或 `1500`

---

## 7. 病人目录内图像差异性 / 异质性分析

### 脚本位置

`utils/analyze_patient_heterogeneity.py`

### 作用

分析同一病人目录下多张图像之间的差异性，输出可审计的 JSON 报告，帮助判断：

- 哪些病人的图像内部差异很大
- 哪些图像在同病人内部像离群点
- 哪些条目存在缺图、缺 mask、空 mask、标签不一致等问题

### 输入

- 现有的 label JSON（如 `train_labels.json`）
- 图像根目录 `image_root`
- mask 根目录 `mask_root`

### 输出

在 `--output_dir` 下默认生成 4 个 JSON 文件：

- `summary.<target_key>.json`
- `patients.<target_key>.json`
- `outliers.<target_key>.json`
- `audit.<target_key>.json`

其中：

- `summary`：总体统计，例如有效分析条目数、总病人数、高异质性病人数
- `patients`：每个病人的异质性结果，例如 `heterogeneity_score`、是否高异质性、离群图数量
- `outliers`：病人内离群图列表，以及触发离群的特征原因
- `audit`：无法正常分析或需要注意的条目/病人，例如缺图、缺 mask、空 mask、重复 filename、标签不一致

### 默认分析特征

- `mask_area_ratio`
- `bbox_fill_ratio`
- `bbox_area_ratio`
- `largest_component_ratio`
- `center_x_ratio`
- `center_y_ratio`
- `edge_touch_penalty`
- `masked_intensity_mean`
- `masked_intensity_std`

### 基本用法（分析 FTCPTC）

```bash
python utils/analyze_patient_heterogeneity.py \
  --input_json /mnt/wangbd8/workspace/DataSets/ThyroidAgent/Classifaction_Data/Malignant_ultrasound_images_cropped/train_labels.json \
  --image_root /mnt/wangbd8/workspace/DataSets/ThyroidAgent/Classifaction_Data/Malignant_ultrasound_images_cropped \
  --mask_root /mnt/wangbd8/workspace/DataSets/ThyroidAgent/Classifaction_Data/Malignant_ultrasound_images_cropped_predictions \
  --output_dir ./log/patient_heterogeneity_ftcptc \
  --target_key FTCPTC \
  --patient_id_depth 2
```

### 基本用法（分析 LNM_CN01）

```bash
python utils/analyze_patient_heterogeneity.py \
  --input_json /mnt/wangbd8/workspace/DataSets/ThyroidAgent/Classifaction_Data/Malignant_ultrasound_images_cropped/train_labels.json \
  --image_root /mnt/wangbd8/workspace/DataSets/ThyroidAgent/Classifaction_Data/Malignant_ultrasound_images_cropped \
  --mask_root /mnt/wangbd8/workspace/DataSets/ThyroidAgent/Classifaction_Data/Malignant_ultrasound_images_cropped_predictions \
  --output_dir ./log/patient_heterogeneity_lnm \
  --target_key LNM_CN01 \
  --patient_id_depth 2
```

### 小范围调试示例

```bash
python utils/analyze_patient_heterogeneity.py \
  --input_json /mnt/wangbd8/workspace/DataSets/ThyroidAgent/Classifaction_Data/Malignant_ultrasound_images_cropped/train_labels.json \
  --image_root /mnt/wangbd8/workspace/DataSets/ThyroidAgent/Classifaction_Data/Malignant_ultrasound_images_cropped \
  --mask_root /mnt/wangbd8/workspace/DataSets/ThyroidAgent/Classifaction_Data/Malignant_ultrasound_images_cropped_predictions \
  --output_dir ./log/patient_heterogeneity_ftcptc_debug \
  --target_key FTCPTC \
  --patient_id_depth 2 \
  --max_patients 50
```

### 常用参数

- `--target_key`：指定要结合哪个二分类标签做统计，如 `FTCPTC` 或 `LNM_CN01`
- `--patient_id_depth`：从 filename 前几段路径构造 patient_id，当前数据通常用 2
- `--min_images_per_patient`：至少有多少张有效图像才计算病人级异质性，默认 2
- `--high_heterogeneity_percentile`：把异质性分数位于全体病人前多少百分位的病人标记为高异质性，默认 90
- `--outlier_z_threshold`：样本较少或 MAD 不可用时的 z-score 离群阈值，默认 2.5
- `--outlier_mad_threshold`：MAD 可用时的 robust 离群阈值，默认 3.5
- `--max_patients`：仅分析排序后的前 N 个病人，用于快速调试
- `--report_prefix`：给输出报告文件名加前缀

### 结果怎么看

- 先看 `patients.<target_key>.json`，找 `heterogeneity_score` 高的病人
- 再看 `outliers.<target_key>.json`，找这些病人内部哪些图像最像离群图
- 再看 `audit.<target_key>.json`，确认是否存在大量缺图、缺 mask、空 mask 或标签不一致问题

### 说明

- 该脚本主要基于图像和 mask 的几何/灰度统计做差异性分析，不依赖模型训练
- 即使某些样本 `target_key` 是 `-1`，也仍可做图像差异性分析；只是这些样本不会作为有效 0/1 标签统计

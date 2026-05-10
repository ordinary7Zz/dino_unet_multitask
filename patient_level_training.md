# 病人级 BCE 二分类多任务训练

## 1. 训练入口

### 训练脚本

`scripts/binary/train_binary_patient_bce_FTCPTC.sh`

### 底层代码

`train_multitask_binary_patient_bce.py`

### 适用场景

- 你想做病人级二分类，而不是图像级二分类
- 一个病人的多张图像共同决定一个病人标签
- 不使用 GLA
- 不使用 `WeightedRandomSampler`
- 分类损失使用 patient-level BCE

---

## 2. 当前设置要点

- `DataLoader` 使用普通 `shuffle=True`
- 病人级 batch size 默认是 4
- 训练时每个病人最多采样 `MAX_IMAGES_PER_PATIENT` 张图
- 分类先对图像级 logits 做 pooling，再按病人标签计算 BCE
- 如果不手动传 `--cls_pos_weight`，会自动按病人级 `neg/pos` 计算 `pos_weight`

---

## 3. 关键参数

### `PATIENT_ID_DEPTH`

从 `filename` 前几段路径提取 `patient_id`，当前默认 2。

### `MAX_IMAGES_PER_PATIENT`

每个病人训练时最多抽多少张图，当前脚本默认 16。

### `CLS_POOLING`

病人级聚合方式，当前脚本默认 `max`。

### `FIXED_THRESHOLD`

验证时固定分类阈值，当前脚本默认 0.5。

---

## 4. 运行方式

```bash
bash scripts/binary/train_binary_patient_bce_FTCPTC.sh
```

---

## 5. 如何修改训练标签文件

修改脚本中的 `TRAIN_LABEL_PATH`，例如：

```bash
TRAIN_LABEL_PATH="/mnt/wangbd8/workspace/DataSets/ThyroidAgent/Classifaction_Data/Malignant_ultrasound_images_cropped/train_labels_ftcptc_2000_1500.json"
```

---

## 6. 如何手动指定病人级正类权重

在脚本里取消注释：

```bash
# CLS_POS_WEIGHT=10.89
```

并在命令末尾追加：

```bash
#     --cls_pos_weight $CLS_POS_WEIGHT
```

---

## 7. 调参经验

- 如果病人级 recall 很低，可先尝试把 `TASK_SCHEDULE` 改成 `cls,cls,seg`
- 如果 `max` pooling 太激进，可再试 `CLS_POOLING="mean"`
- 如果关键阳性图容易漏采，可适当增大 `MAX_IMAGES_PER_PATIENT`

---

## 8. 默认训练方式

### 病人级默认脚本

- `scripts/binary/train_binary_patient_bce_FTCPTC.sh`

### 说明

- 如果 `TRAIN_LABEL_PATH` 仍指向原始 `train_labels.json`，那么训练分布就是原始分布
- 如果 `TRAIN_LABEL_PATH` 换成你生成的新 JSON，那么训练分布就由新 JSON 决定

---

## 9. 推荐实验顺序

### 方案 C：人工重采样 JSON + 病人级 BCE 训练

1. 用 `utils/build_binary_resampled_labels.py` 生成新 JSON
2. 修改 `scripts/binary/train_binary_patient_bce_FTCPTC.sh` 中的 `TRAIN_LABEL_PATH`
3. 根据需要调整：
   - `TASK_SCHEDULE`
   - `MAX_IMAGES_PER_PATIENT`
   - `CLS_POOLING`
   - `CLS_POS_WEIGHT`
4. 运行：

```bash
bash scripts/binary/train_binary_patient_bce_FTCPTC.sh
```

---

## 10. 参数含义速查

### `PATIENT_ID_DEPTH`

病人级训练时，从 `filename` 前几段路径构造 `patient_id`。

### `MAX_IMAGES_PER_PATIENT`

病人级训练时，每个病人最多抽取的图像数。

### `CLS_POOLING`

病人级训练时，把图像级 logits 聚合成病人级 logit 的方式：

- `max`：只看最强响应
- `mean`：看整体平均证据

### `FIXED_THRESHOLD`

验证时固定使用的病人级分类阈值。

---

## 11. 当前建议

病人级 BCE 版本可先从下面组合开始：

- `TASK_SCHEDULE="seg,cls"` 或 `"cls,cls,seg"`
- `MAX_IMAGES_PER_PATIENT=16`
- `CLS_POOLING="max"`
- `CLS_POS_WEIGHT` 使用自动计算，或手动给一个病人级 `neg/pos` 比值附近的数

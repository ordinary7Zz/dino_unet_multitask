# 病人级二分类训练设计方案

## 目标

在现有多任务框架上，只实现**病人级二分类训练**，不扩展到病人级 TI-RADS 多分类。

本方案的核心目标是：

- 保留现有分割分支与分割监督方式
- 将二分类监督从**图像级**改为**病人级**
- 一个病人的多张图像组成一个 bag
- 模型仍逐图像前向
- 分类分支对同一病人的实例 logits 做 pooling 后，再计算病人级二分类 loss
- 验证与测试阶段的二分类指标也按病人级统计

---

## 当前代码基础

当前仓库已经具备以下基础能力：

1. `dataset.py`
   - 现有 `MultiTaskDataset` 是图像级数据集
   - 一次返回一张图像、一张 mask，以及该图像对应的分类标签

2. `patient_dataset.py`
   - 已有 `PatientMultiTaskDataset` 雏形
   - 已支持将同一病人的多张图组织成一个 group / bag
   - 已提供 `collate_patient_bags()`，可把多个病人的 bag 合并为一个训练 batch

3. `utils/patient_metrics.py`
   - 已有 `pool_instance_logits()`，可根据 `bag_sizes` 对实例级 logit 做 pooling
   - 已有 `evaluate_model_binary_target_patient()`，可按病人级评估二分类结果

4. `dino_unet_multitask.py`
   - 当前模型输入仍是普通图像 batch：`[B, C, H, W]`
   - 输出为：
     - 分割 logits
     - 二分类 logits
     - TI-RADS logits

因此，当前最适合的改造路径不是重写模型输入结构，而是：

- 数据层按病人组织
- 前向时仍将 bag 内图像展平后送入模型
- 分类损失和分类评估改为病人级

---

## 病人级训练的定义

### 1. 训练单位

训练时，一个样本不再是一张图，而是一个病人。

即：

- 一个病人 = 多张超声图像
- 每张图像有自己的 segmentation mask
- 一个病人对应一个二分类标签，例如：
  - `LNM_CN01`
  - `FTCPTC`

### 2. 监督粒度

- **分割任务：图像级监督**
- **二分类任务：病人级监督**

这意味着：

- 分割 loss 仍对 bag 中每张图分别计算
- 分类 loss 不再对每张图计算，而是对整个病人的聚合预测计算

### 3. 本质建模方式

这是一个典型的轻量级 MIL（Multiple Instance Learning）/ bag-level classification 方案：

- instance = 单张图像
- bag = 单个病人
- bag label = 病人级标签

---

## 病人 ID 的定义

当前标签 JSON 中的 `filename` 形式例如：

```json
"2016/刘惠银/刘惠银_01_0001_0001.jpg"
```

建议当前版本采用：

- `patient_id = 年份/姓名`

即：

```text
2016/刘惠银
```

这与 `patient_dataset.py` 中现有 `derive_patient_id(..., depth=2)` 的逻辑一致。

### 约束

同一个 `patient_id` 下的所有图像必须满足：

- 二分类标签一致
- 若标签不一致，则直接报错，不允许静默跳过

---

## 数据组织设计

### 1. 训练集数据集

使用 `PatientMultiTaskDataset`，而不是 `MultiTaskDataset`。

每次 `__getitem__()` 返回一个病人的完整 bag，包含：

- `image`: `[num_images, C, H, W]`
- `label`: `[num_images, 1, H, W]`
- `target`: 病人级二分类标签
- `patient_id`
- `filenames`
- `num_images`

### 2. collate 逻辑

使用 `collate_patient_bags()`。

其输出设计为：

- `image`: 将多个病人的所有图像拼接后得到 `[sum_instances, C, H, W]`
- `label`: `[sum_instances, 1, H, W]`
- `target`: `[num_patients]`
- `bag_sizes`: `[num_patients]`
- `patient_ids`
- `filenames`

### 3. 为什么采用扁平化 batch

因为现有模型 `DINOv3_S_UNet_MULTITASK` 只接受标准图像 batch 输入：

```python
[B, C, H, W]
```

因此不改模型输入签名，而是在数据层把 bag 展平，是当前工程里改动最小、最稳定的方案。

---

## 模型前向与损失设计

## 1. 前向方式

训练时，直接把扁平后的图像 batch 送入现有模型：

```python
pred_seg, pred_cls, _ = model(batch['image'])
```

其中：

- `pred_seg`: 对每张图像的分割 logits
- `pred_cls`: 对每张图像的二分类 logits

这里的 `pred_cls` 仍然是**实例级输出**，还不是病人级输出。

## 2. 分割损失

分割仍保持图像级：

```python
seg_loss = structure_loss(pred_seg, batch['label'])
```

因为 `pred_seg` 与 `batch['label']` 都已经按图像实例对齐。

## 3. 病人级分类 pooling

分类分支要先将实例级 logits 聚合成病人级 logits。

建议复用：

- `utils/patient_metrics.py` 中的 `pool_instance_logits()`

计算方式：

```python
pooled_logits = pool_instance_logits(pred_cls, batch['bag_sizes'], pooling=cls_pooling)
```

聚合后：

- `pooled_logits.shape == [num_patients]`
- `batch['target'].shape == [num_patients]`

此时才可计算病人级二分类损失。

## 4. 分类损失

若使用普通 BCE：

```python
cls_loss = benign_malignant_loss(
    pooled_logits.unsqueeze(1),
    batch['target']
)
```

若使用 GLA 版本：

```python
cls_loss = benign_malignant_loss_gla(
    pooled_logits.unsqueeze(1),
    batch['target'],
    p_pos=p_pos,
    p_neg=p_neg,
    tau=gla_tau,
)
```

注意：

- 类别频率统计必须按**病人数**统计，而不是按图像数统计
- 否则病人级训练目标与 loss weighting 口径不一致

## 5. 总 loss

建议仍保持双任务训练：

```python
total_loss = seg_weight * seg_loss + cls_weight * cls_loss
```

如果沿用现有 GA-MTL 风格，也可以保留 `seg,cls` 的 task schedule，但分类任务必须改成病人级 pooling 后再算 loss。

---

## pooling 策略设计

### 默认策略

首版默认使用：

- `max pooling`

即：

```python
bag_logit = max(instance_logits)
```

### 原因

- 若病人的多张图中只有少数图强烈提示阳性，`max` 更容易保留关键阳性信号
- 更符合弱监督 / MIL 的经典设定
- 改动小，易于作为 baseline

### 备选策略

同时建议保留参数化支持：

- `mean pooling`

命令行参数建议：

```text
--cls_pooling max
--cls_pooling mean
```

### 结论

首版先实现：

- `max` 为默认
- `mean` 为可选实验项

---

## 训练脚本设计

建议新增独立脚本，而不是直接覆盖现有图像级脚本。

建议文件名：

```text
train_multitask_binary_patient.py
```

### 新脚本职责

该脚本只负责：

- 病人级二分类 + 图像级分割联合训练
- 指定 `target_key`（例如 `LNM_CN01` 或 `FTCPTC`）
- 病人级验证与测试

### 建议复用的现有能力

可直接复用：

- 现有模型 `DINOv3_S_UNet_MULTITASK`
- `structure_loss`
- `benign_malignant_loss` / `benign_malignant_loss_gla`
- 日志与 checkpoint 逻辑
- AMP、scheduler、optimizer 的现有实现

### 需要新增或改造的参数

建议支持：

- `--target_key`
- `--patient_id_depth`
- `--max_images_per_patient`
- `--cls_pooling`
- `--task_schedule`
- `--gla_tau`（如果保留 GLA）

其中：

- `patient_id_depth` 默认可设为 `2`
- `max_images_per_patient` 仅训练阶段生效，用于控制显存
- 测试阶段默认不裁剪病人图像数

---

## 训练阶段的数据细节

## 1. batch 含义

病人级训练时：

- `batch_size` 应表示**病人数**，不是图像数

例如：

- `batch_size=4`
- 表示每个 batch 取 4 个病人
- 实际送进模型的图像数为这 4 个病人的图像总和

因此显存占用会随病人的图像数波动。

## 2. max_images_per_patient

建议训练阶段支持：

```text
--max_images_per_patient N
```

作用：

- 当某个病人的图像数量过多时，随机采样 N 张图参与训练
- 控制显存与训练稳定性
- 同时保留病人级监督的核心特性

注意：

- 该参数只在训练集生效
- 验证 / 测试阶段应默认使用全部图像

## 3. shuffle 粒度

训练集 shuffle 应在**病人级**进行，而不是图像级。

也就是说：

- DataLoader 打乱的是 `patient_id`
- 不应再把同一病人的图像拆散后独立打乱

---

## 验证与测试设计

### 1. 分类评估必须改成病人级

验证阶段不能继续使用图像级分类评估函数，而应改用：

- `evaluate_model_binary_target_patient()`

该函数的正确口径是：

1. 对所有实例图像逐张前向
2. 按 `bag_sizes` 将实例 logits 聚合成病人级 logits
3. 计算病人级概率
4. 按病人标签统计：
   - accuracy
   - precision
   - recall
   - f1
   - auroc
   - auprc
   - ece
   - sensitivity / specificity / youden（若保留）

### 2. 分割评估口径

首版建议保留当前分割评估口径：

- 仍按图像实例逐张计算 Dice / HD95
- 最终对所有图像实例汇总

原因：

- 当前任务重点是“分类从图像级改到病人级”
- 分割监督与分割评估无需同时做病人级重定义

---

## 类别不平衡统计口径

病人级训练下，正负样本统计必须从“图像数”切换到“病人数”。

### 正确做法

对每个病人只统计一次标签：

- negative patient count
- positive patient count
- missing patient count

### 原因

如果继续按图像数统计，会导致：

- 图像多的病人被重复加权
- GLA / pos_weight 与实际训练目标不一致
- 病人级 loss 被隐式扭曲

因此，无论是：

- `pos_weight`
- `gla_params_binary`
- 训练日志中的 label stats

都应基于**病人级计数**重算。

---

## 首版实现边界

本次只做以下范围：

1. 仅支持**二分类病人级训练**
2. 保留**图像级分割监督**
3. 保留现有模型结构，不重写 backbone / head
4. 保留实例级分割输出
5. 将二分类 loss 与二分类评估改为病人级

本次不做：

1. 不实现病人级 TI-RADS 多分类
2. 不实现 attention-based MIL pooling
3. 不修改模型输入为 `[B, N, C, H, W]`
4. 不重构整个多任务框架
5. 不定义病人级分割指标

---

## 推荐实现步骤

### 第一步：清理并确认病人级数据接口

- 检查 `PatientMultiTaskDataset` 是否只服务二分类场景
- 确认 `target_key` 为单一二分类标签
- 确认同病人标签一致性校验
- 确认 `patient_id_depth=2` 在当前数据集上成立

### 第二步：新增病人级训练脚本

新增：

```text
train_multitask_binary_patient.py
```

并完成：

- train loader 使用 `PatientMultiTaskDataset`
- test loader 使用 `PatientMultiTaskDataset`
- `collate_fn=collate_patient_bags`

### 第三步：接入病人级分类 loss

训练时：

- 前向得到实例级 `pred_cls`
- 使用 `bag_sizes` 聚合为病人级 logits
- 用病人级 `target` 计算分类损失

### 第四步：接入病人级评估

验证时：

- 改用 `evaluate_model_binary_target_patient()`
- 日志中明确标注分类指标为 patient-level

### 第五步：补充病人级统计与参数

包括：

- 病人数统计
- 每个病人的图像数量统计
- `cls_pooling`
- `max_images_per_patient`
- 病人级正负样本比例

---

## 风险点与注意事项

### 1. 病人图像数不均衡

不同病人的图像数量可能差异很大，导致：

- batch 内实际图像总数波动
- 显存不稳定
- 训练 step 耗时不稳定

解决方向：

- 增加 `max_images_per_patient`
- 减小病人级 `batch_size`

### 2. 分类 head 当前仍是实例级输出

这是预期行为，不是 bug。

当前设计并不要求模型直接输出病人级 logits，而是：

- 先输出实例级 logits
- 再在训练 / 评估阶段做 bag-level pooling

### 3. segmentation loss 与 classification loss 的样本粒度不同

这是该方案的核心特征：

- segmentation：instance-level
- classification：patient-level

因此日志与实现中必须清楚区分，避免误把 `batch_size` 当成图像数理解。

### 4. GLA / 类别权重口径必须统一

如果保留 GLA 或 `pos_weight`：

- 必须使用病人级标签统计
- 不能复用图像级统计结果

---

## 最终结论

当前工程中，病人级二分类训练的最优落地路径是：

- 使用 `PatientMultiTaskDataset` 按病人构造 bag
- 使用 `collate_patient_bags()` 将病人 bag 展平为实例 batch + `bag_sizes`
- 模型保持逐图像前向，不修改输入签名
- 分割任务继续使用图像级 mask 监督
- 二分类任务对实例 logits 做 pooling，转成病人级 logits 后计算 loss
- 验证与测试阶段按病人级输出二分类指标

这是当前代码基础上改动最小、最容易稳定实现、并且与病人级标签定义一致的方案。

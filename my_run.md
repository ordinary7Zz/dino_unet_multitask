# my_run 使用说明

## 文档导航

- 图像级训练与标签重采样：本文档
- 病人级训练说明：[`patient_level_training.md`](./patient_level_training.md)

---

## 1. 分类推理脚本

### 脚本位置

`infer_classification.py`

### 作用

使用训练好的 `DINOv3_S_UNet_MULTITASK` 权重，对图像目录进行批量分类推理。根据分类数量自动选择分类头：

- `num_classes=2`：使用 `benign_malignant_head`（良恶性二分类，sigmoid + threshold 0.5）
- `num_classes=5`：使用 `tirads_head`（TIRADS 五分类，softmax + argmax）

输出 CSV 文件，包含每张图像的文件名、预测类别和各类别概率/置信度分数。

### 基本用法

#### 二分类推理

```bash
python infer_classification.py \
    --image_dir /mnt/wangbd8/workspace/DataSets/ThyroidAgent/train_val_test/sample/images \
    --checkpoint /mnt/wangbd8/workspace/ThyroidAgent/dino_unet_multitask/checkpoints/gamtl/dino_unet_gamtl_train_multitask_dataset_4_epoch_50.pth \
    --num_classes 2 \
    --output results/binary_preds.csv \
    --img_size 224
```

#### 五分类推理

```bash
python infer_classification.py \
    --image_dir /path/to/test/images \
    --checkpoint logs/default/GA-MTL_TIRADS_20250601_120000/GA-MTL_default_TIRADS_epoch_50.pth \
    --num_classes 5 \
    --output results/tirads_preds.csv \
    --img_size 224
```

### 参数说明

| 参数 | 必填 | 默认值 | 说明 |
|------|------|--------|------|
| `--image_dir` | ✓ | — | 待推理图像所在目录 |
| `--checkpoint` | ✓ | — | 模型权重文件路径 (.pth) |
| `--num_classes` | ✓ | — | 分类类别数：`2` 或 `5` |
| `--output` | ✓ | — | 输出 CSV 文件路径 |
| `--img_size` | | `224` | 输入图像尺寸，需与训练时一致 |
| `--dino_pretrained` | | `True` | DINO backbone 是否使用预训练权重，需与训练时一致 |
| `--use_dilation` | | `False` | 模型是否使用 dilation 层，需与训练时一致 |
| `--cuda_device` | | `0` | CUDA 设备索引 |
| `--batch_size` | | `16` | 推理批大小 |
| `--num_workers` | | `4` | DataLoader 子进程数 |

### 输出说明

**二分类 CSV 格式**：`filename | predicted_class | prob_0 | prob_1`

**五分类 CSV 格式**：`filename | predicted_class | prob_0 | prob_1 | prob_2 | prob_3 | prob_4`

脚本会在控制台打印预测类别分布统计。

---

## 2. 分类结果指标汇总

### 脚本位置

`compute_single_task_binary_metrics.py`

### 作用

读取单个分类模型导出的标准化 JSON，直接计算常见二分类指标及其 CI95：

- AUROC
- AUPRC
- precision
- recall
- accuracy
- F1
- sensitivity / specificity
- Youden index
- ECE

### 支持的输入格式

该脚本支持本文档里约定的两种样本级格式：

1. `true_label` + `prob_class_1`
2. `ground_truth_label` + `malignant_probability`

### 基本用法

```bash
python compute_single_task_binary_metrics.py \
  /path/to/model_results.json
```

### 可选参数

- `--threshold`：计算 precision / recall / F1 / sensitivity / specificity 时使用的阈值，默认 `0.5`
- `--n-boot`：bootstrap 次数，默认 `2000`
- `--ci`：置信区间水平，默认 `0.95`
- `--seed`：bootstrap 随机种子，默认 `0`
- `--output`：指定输出汇总 JSON 的路径；不指定时默认输出到输入文件同目录下的 `*_binary_metrics.json`

### 示例

#### 示例 1：直接计算并输出同目录结果文件

```bash
python compute_single_task_binary_metrics.py \
  my_model_results.json
```

#### 示例 2：指定阈值和输出路径

```bash
python compute_single_task_binary_metrics.py \
  my_model_results.json \
  --threshold 0.4 \
  --n-boot 5000 \
  --seed 1024 \
  --output my_model_metrics_summary.json
```

### 输出说明

脚本会在控制台打印一份简洁摘要，并生成一个 JSON 汇总文件，里面包含：

- 样本数、正负样本数
- 各指标的 mean 与 CI95
- Youden 最优阈值及其对应统计量
- bootstrap 参数记录

---

## 3. 按单个标签值复制图像并重命名

### 脚本位置

`copy_images_by_label.py`

### 作用

根据分类标签 JSON 和指定标签名，把原始图像从 `image_root` 复制到 `save_root/0` 或 `save_root/1` 下，并把文件名重命名为把 `filename` 各级路径用下划线拼接后的形式。

例如：

- `filename = 2016/叶永强/叶永强_01_0003_0003.jpg`
- 重命名后：`2016_叶永强_叶永强_01_0003_0003.jpg`
- 若 `LNM_CN01 = 1`，则保存到：`save_root/1/2016_叶永强_叶永强_01_0003_0003.jpg`

### 基本用法

```bash
python copy_images_by_label.py \
  --json_file "/mnt/wangbd8/workspace/DataSets/ThyroidAgent/train_val_test/finall_data/data_label.json" \
  --label_name malignancy \
  --image_root "/mnt/wangbd8/workspace/DataSets/ThyroidAgent/train_val_test/finall_data/image" \
  --save_root "/mnt/wangbd8/workspace/DataSets/ThyroidAgent/train_val_test/finall_data"
```

### 参数说明

- `--json_file`：分类标签 JSON 文件路径
- `--label_name`：要筛选的标签名，例如 `LNM_CN01`、`FTCPTC`
- `--image_root`：原始图像根目录，脚本会用 `image_root + filename` 定位源图像
- `--save_root`：保存目录，脚本会自动创建 `0`、`1` 子目录

### 说明

- 只处理标签值为 `0/1` 的样本
- 如果源图像不存在，会打印缺失信息并跳过
- 输出文件名不保留原来的目录结构，避免重复文件名冲突

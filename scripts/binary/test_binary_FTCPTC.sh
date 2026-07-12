#!/bin/bash

# ---------------------- Configuration ----------------------
TARGET_KEY="FTCPTC"
DATASET_NAME="${TARGET_KEY}"
CUDA_VISIBLE_DEVICES="1"

# Model checkpoint path
CHECKPOINT_PATH="/mnt/wangbd8/workspace/ThyroidAgent/dino_unet_multitask/checkpoints/FTCPTC_4795/train_multitask_FTCPTC_4795/20260712_185826/dino_unet_train_multitask_FTCPTC_4795_FTCPTC_epoch_15.pth"

# Configure test dataset paths
# 测试数据集名称数组
TEST_DATASET_NAMES=(
    "FTCPTC_FangDai"
)

# 测试图像路径数组
TEST_IMAGE_PATHS=(
    "/mnt/wangbd8/workspace/DataSets/ThyroidAgent/Classifaction_Data/FangDai_Thyroid_Ultrasound_Images_cropped/"
)

# 测试掩码路径数组
TEST_MASK_PATHS=(
    "/mnt/wangbd8/workspace/DataSets/ThyroidAgent/Classifaction_Data/FangDai_Thyroid_Ultrasound_Images_cropped_predictions/"
)

# 测试分类标签路径数组
TEST_LABEL_PATHS=(
    "/mnt/wangbd8/workspace/DataSets/ThyroidAgent/Classifaction_Data/FangDai_Thyroid_Ultrasound_Images_cropped/FangDai_all_labels.json"
)

if [ ${#TEST_DATASET_NAMES[@]} -ne ${#TEST_IMAGE_PATHS[@]} ] || [ ${#TEST_DATASET_NAMES[@]} -ne ${#TEST_MASK_PATHS[@]} ] || [ ${#TEST_DATASET_NAMES[@]} -ne ${#TEST_LABEL_PATHS[@]} ]; then
    echo "Error: Arrays must have the same length"
    exit 1
fi

SAVE_PATH="./predictions/binary/test_${DATASET_NAME}"
SAVE_RESULTS="false"
LOG_DIR="./logs/test_logs/binary/test_${DATASET_NAME}_5012"
IMG_SIZE=224
DINO_PRETRAINED="true"

# Fixed threshold for malignancy classification.
THRESHOLD_MALIGNANCY="0.5"

# ---------------------- Execution ----------------------
mkdir -p "$SAVE_PATH"

TEST_IMAGE_ARGS=()
for img_path in "${TEST_IMAGE_PATHS[@]}"; do
    TEST_IMAGE_ARGS+=("--test_image_paths" "$img_path")
done

TEST_MASK_ARGS=()
for mask_path in "${TEST_MASK_PATHS[@]}"; do
    TEST_MASK_ARGS+=("--test_gt_paths" "$mask_path")
done

TEST_LABEL_ARGS=()
for label_path in "${TEST_LABEL_PATHS[@]}"; do
    TEST_LABEL_ARGS+=("--test_label_paths" "$label_path")
done

TEST_NAMES_ARGS=()
for dataset_name in "${TEST_DATASET_NAMES[@]}"; do
    TEST_NAMES_ARGS+=("--test_dataset_names" "$dataset_name")
done

CMD=(
    python -u test_paralled_binary.py
    --cuda_device "$CUDA_VISIBLE_DEVICES"
    --checkpoint "$CHECKPOINT_PATH"
    --target_key "$TARGET_KEY"
    "${TEST_IMAGE_ARGS[@]}"
    "${TEST_MASK_ARGS[@]}"
    "${TEST_LABEL_ARGS[@]}"
    "${TEST_NAMES_ARGS[@]}"
    --save_path "$SAVE_PATH"
    --save_results "$SAVE_RESULTS"
    --log_dir "$LOG_DIR"
    --img_size "$IMG_SIZE"
    --dino_pretrained "$DINO_PRETRAINED"
)

CMD+=(--threshold_malignancy "$THRESHOLD_MALIGNANCY")

"${CMD[@]}"

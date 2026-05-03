#!/bin/bash

# ---------------------- Configuration ----------------------
TARGET_KEY="LNM_CN01"
DATASET_NAME="${TARGET_KEY}"
CUDA_VISIBLE_DEVICES="1"

# Model checkpoint path
CHECKPOINT_PATH="/mnt/wangbd8/workspace/ThyroidAgent/dino_unet_multitask/checkpoints/train_multitask_${TARGET_KEY}/YOUR_TIMESTAMP/dino_unet_train_multitask_${TARGET_KEY}_${TARGET_KEY}_epoch_50.pth"

# Validation dataset paths (used when --threshold_malignancy is not provided)
VAL_IMAGE_PATH="/mnt/wangbd8/workspace/DataSets/ThyroidAgent/Classifaction_Data/Malignant_ultrasound_images_cropped/"
VAL_MASK_PATH="/mnt/wangbd8/workspace/DataSets/ThyroidAgent/Classifaction_Data/Malignant_ultrasound_images_cropped_predictions/"
VAL_LABEL_PATH="/mnt/wangbd8/workspace/DataSets/ThyroidAgent/Classifaction_Data/Malignant_ultrasound_images_cropped/test_labels.json"

# Configure test dataset paths
TEST_DATASET_NAMES=(
    "LNM_CN01"
)

TEST_IMAGE_PATHS=(
    "/mnt/wangbd8/workspace/DataSets/ThyroidAgent/Classifaction_Data/Malignant_ultrasound_images_cropped/"
)

TEST_MASK_PATHS=(
    "/mnt/wangbd8/workspace/DataSets/ThyroidAgent/Classifaction_Data/Malignant_ultrasound_images_cropped_predictions/"
)

TEST_LABEL_PATHS=(
    "/mnt/wangbd8/workspace/DataSets/ThyroidAgent/Classifaction_Data/Malignant_ultrasound_images_cropped/test_labels.json"
)

if [ ${#TEST_DATASET_NAMES[@]} -ne ${#TEST_IMAGE_PATHS[@]} ] || [ ${#TEST_DATASET_NAMES[@]} -ne ${#TEST_MASK_PATHS[@]} ] || [ ${#TEST_DATASET_NAMES[@]} -ne ${#TEST_LABEL_PATHS[@]} ]; then
    echo "Error: Arrays must have the same length"
    exit 1
fi

SAVE_PATH="./predictions/binary/test_${DATASET_NAME}"
SAVE_RESULTS="false"
LOG_DIR="./logs/test_logs/binary/test_${DATASET_NAME}"
IMG_SIZE=224
DINO_PRETRAINED="true"

# Set THRESHOLD_MALIGNANCY to a numeric value such as 0.5 to use a fixed threshold.
# Leave it empty to compute the threshold from the validation set above.
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

if [ -n "$THRESHOLD_MALIGNANCY" ]; then
    CMD+=(--threshold_malignancy "$THRESHOLD_MALIGNANCY")
else
    CMD+=(
        --val_image_path "$VAL_IMAGE_PATH"
        --val_gt_path "$VAL_MASK_PATH"
        --val_label_path "$VAL_LABEL_PATH"
    )
fi

"${CMD[@]}"

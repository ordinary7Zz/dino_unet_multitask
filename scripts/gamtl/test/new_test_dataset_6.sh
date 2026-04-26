#!/bin/bash

# ---------------------- Configuration ----------------------
# Set CUDA device
DATASET_NAME="dataset_6"
CUDA_VISIBLE_DEVICES="1"

# Model checkpoint path
CHECKPOINT_PATH="/mnt/wangbd8/workspace/ThyroidAgent/dino_unet_multitask/checkpoints/gamtl/dino_unet_gamtl_train_multitask_${DATASET_NAME}_epoch_50.pth"

# Validation dataset paths (used to select malignancy threshold by Youden)
VAL_IMAGE_PATH="/mnt/wangbd8/workspace/DataSets/ThyroidAgent/Superimposed_multitask/${DATASET_NAME}/images/"
VAL_MASK_PATH="/mnt/wangbd8/workspace/DataSets/ThyroidAgent/Superimposed_multitask/${DATASET_NAME}/masks/"
VAL_LABEL_PATH="/mnt/wangbd8/workspace/DataSets/ThyroidAgent/Superimposed_multitask/${DATASET_NAME}/${DATASET_NAME}_label.json"

# Configure multiple test dataset paths
TEST_DATASET_NAMES=(
    "DDTI_Classification"
    "TN3K"
    "ThyroidXL"
    "TN5K"
)

TEST_IMAGE_PATHS=(
    "/mnt/wangbd8/workspace/DataSets/ThyroidAgent/train_val_test/DDTI_Classification/valid_test/images/"
    "/mnt/wangbd8/workspace/DataSets/ThyroidAgent/train_val_test/TN3K/test/images/"
    "/mnt/wangbd8/workspace/DataSets/ThyroidAgent/train_val_test/ThyroidXL/test/images/"
    "/mnt/wangbd8/workspace/DataSets/ThyroidAgent/train_val_test/TN5K/test/images/"
)

TEST_MASK_PATHS=(
    "/mnt/wangbd8/workspace/DataSets/ThyroidAgent/train_val_test/DDTI_Classification/valid_test/masks/"
    "/mnt/wangbd8/workspace/DataSets/ThyroidAgent/train_val_test/TN3K/test/masks/"
    "/mnt/wangbd8/workspace/DataSets/ThyroidAgent/train_val_test/ThyroidXL/test/masks/"
    "/mnt/wangbd8/workspace/DataSets/ThyroidAgent/train_val_test/TN5K/test/masks/"
)

TEST_LABEL_PATHS=(
    "/mnt/wangbd8/workspace/DataSets/ThyroidAgent/train_val_test/DDTI_Classification/valid_test/DDTI_Classification_valid_test_label.json"
    "/mnt/wangbd8/workspace/DataSets/ThyroidAgent/train_val_test/TN3K/test/TN3K_test_label.json"
    "/mnt/wangbd8/workspace/DataSets/ThyroidAgent/train_val_test/ThyroidXL/test/ThyroidXL_test_label.json"
    "/mnt/wangbd8/workspace/DataSets/ThyroidAgent/train_val_test/TN5K/test/TN5K_test_label.json"
)

# Ensure arrays have the same length
if [ ${#TEST_DATASET_NAMES[@]} -ne ${#TEST_IMAGE_PATHS[@]} ] || [ ${#TEST_DATASET_NAMES[@]} -ne ${#TEST_MASK_PATHS[@]} ] || [ ${#TEST_DATASET_NAMES[@]} -ne ${#TEST_LABEL_PATHS[@]} ]; then
    echo "Error: Arrays must have the same length"
    exit 1
fi

# Prediction results save path
SAVE_PATH="./predictions/new_test_${DATASET_NAME}"

# Whether to save prediction results (true/false)
SAVE_RESULTS="false"

# Log directory
LOG_DIR="./logs/test_logs/gamtl/new_test_${DATASET_NAME}"

# ---------------------- Execution ----------------------

# Create save directory if it doesn't exist
mkdir -p "$SAVE_PATH"

# Build test image paths arguments
TEST_IMAGE_ARGS=()
for img_path in "${TEST_IMAGE_PATHS[@]}"; do
    if [ -d "$img_path" ]; then
        TEST_IMAGE_ARGS+=("--test_image_paths" "$img_path")
    fi
done

# Build test mask paths arguments
TEST_MASK_ARGS=()
for mask_path in "${TEST_MASK_PATHS[@]}"; do
    if [ -d "$mask_path" ]; then
        TEST_MASK_ARGS+=("--test_gt_paths" "$mask_path")
    fi
done

# Build test label paths arguments
TEST_LABEL_ARGS=()
for label_path in "${TEST_LABEL_PATHS[@]}"; do
    if [ -f "$label_path" ]; then
        TEST_LABEL_ARGS+=("--test_label_paths" "$label_path")
    else
        # If label file doesn't exist, still add it as empty to maintain order
        TEST_LABEL_ARGS+=("--test_label_paths" "")
    fi
done

# Build test dataset names arguments
TEST_NAMES_ARGS=()
for dataset_name in "${TEST_DATASET_NAMES[@]}"; do
    TEST_NAMES_ARGS+=("--test_dataset_names" "$dataset_name")
done

# Execute the test command
python -u test_parallel.py \
    --threshold_malignancy 0.5 \
    --cuda_device $CUDA_VISIBLE_DEVICES \
    --checkpoint "$CHECKPOINT_PATH" \
    --val_image_path "$VAL_IMAGE_PATH" \
    --val_gt_path "$VAL_MASK_PATH" \
    --val_label_path "$VAL_LABEL_PATH" \
    "${TEST_IMAGE_ARGS[@]}" \
    "${TEST_MASK_ARGS[@]}" \
    "${TEST_LABEL_ARGS[@]}" \
    "${TEST_NAMES_ARGS[@]}" \
    --save_path "$SAVE_PATH" \
    --save_results "$SAVE_RESULTS" \
    --log_dir "$LOG_DIR" \
    --img_size 224 \
    --dino_pretrained "true"
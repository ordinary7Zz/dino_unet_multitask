#!/bin/bash

# ---------------------- Configuration ----------------------
# Set CUDA device
CUDA_VISIBLE_DEVICES="1"

# Model checkpoint path
CHECKPOINT_PATH="/mnt/wangbd8/workspace/ThyroidAgent/dino_unet_multitask/checkpoints/nopretrained/nopretrained_train_multitask_dataset4/20251225_230954/dino_unet_nopretrained_train_multitask_dataset4_epoch_50.pth"

# Validation dataset paths (used to select malignancy threshold by Youden)
VAL_IMAGE_PATH="/mnt/wangbd8/workspace/DataSets/ThyroidAgent/Superimposed_multitask/dataset_4/images/"
VAL_MASK_PATH="/mnt/wangbd8/workspace/DataSets/ThyroidAgent/Superimposed_multitask/dataset_4/masks/"
VAL_LABEL_PATH="/mnt/wangbd8/workspace/DataSets/ThyroidAgent/Superimposed_multitask/dataset_4/dataset4_label.json"

# Configure multiple test dataset paths
TEST_DATASET_NAMES=(
    "TN3K"
    "DDTI"
    "ThyroidXL"
    "PKTN"
    "TN5K"
    "test_dataset4"
)

TEST_IMAGE_PATHS=(
    "/mnt/wangbd8/workspace/DataSets/ThyroidAgent/TN3K/test-image/"
    "/mnt/wangbd8/workspace/DataSets/ThyroidAgent/DDTI/2_preprocessed_data/stage1/p_image/"
    "/mnt/wangbd8/workspace/DataSets/ThyroidAgent/ThyroidXL/test/images/"
    "/mnt/wangbd8/workspace/DataSets/ThyroidAgent/PKTN_processed/images/"
    "/mnt/wangbd8/workspace/DataSets/ThyroidAgent/TN5K_processed/test/images/"
    "/mnt/wangbd8/workspace/DataSets/ThyroidAgent/Superimposed_multitask/test_dataset_4/images/"
)

TEST_MASK_PATHS=(
    "/mnt/wangbd8/workspace/DataSets/ThyroidAgent/TN3K/test-mask/"
    "/mnt/wangbd8/workspace/DataSets/ThyroidAgent/DDTI/2_preprocessed_data/stage1/p_mask/"
    "/mnt/wangbd8/workspace/DataSets/ThyroidAgent/ThyroidXL/test/masks/"
    "/mnt/wangbd8/workspace/DataSets/ThyroidAgent/PKTN_processed/masks/"
    "/mnt/wangbd8/workspace/DataSets/ThyroidAgent/TN5K_processed/test/masks/"
    "/mnt/wangbd8/workspace/DataSets/ThyroidAgent/Superimposed_multitask/test_dataset_4/masks/"
)

TEST_LABEL_PATHS=(
    "/mnt/wangbd8/workspace/DataSets/ThyroidAgent/TN3K/tn3k_test_label.json"
    "/mnt/wangbd8/workspace/DataSets/ThyroidAgent/DDTI/2_preprocessed_data/stage1/DDTI_label.json"
    "/mnt/wangbd8/workspace/DataSets/ThyroidAgent/ThyroidXL/test/thyroidxl_test_labels.json"
    "/mnt/wangbd8/workspace/DataSets/ThyroidAgent/PKTN_processed/PKTN_processed_label.json"
    "/mnt/wangbd8/workspace/DataSets/ThyroidAgent/TN5K_processed/test/TN5K_test_label.json"
    "/mnt/wangbd8/workspace/DataSets/ThyroidAgent/Superimposed_multitask/test_dataset_4/test_dataset4_label.json"
)

# Ensure arrays have the same length
if [ ${#TEST_DATASET_NAMES[@]} -ne ${#TEST_IMAGE_PATHS[@]} ] || [ ${#TEST_DATASET_NAMES[@]} -ne ${#TEST_MASK_PATHS[@]} ] || [ ${#TEST_DATASET_NAMES[@]} -ne ${#TEST_LABEL_PATHS[@]} ]; then
    echo "Error: Arrays must have the same length"
    exit 1
fi

# Prediction results save path
SAVE_PATH="./predictions/test_dataset4"

# Whether to save prediction results (true/false)
SAVE_RESULTS="false"

# Log directory
LOG_DIR="./logs/test_logs/nopretrained/test_dataset4"

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

# Choose test script based on whether to use Dilation
USE_Dilation="false"

if [ "$USE_Dilation" = "true" ] ; then
    TEST_SCRIPT="test_parallel_Dilation.py"
else
    TEST_SCRIPT="test_parallel.py"
fi
# Execute the test command
python -u "$TEST_SCRIPT" \
    --threshold_malignancy None \
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
    --dino_pretrained "false"
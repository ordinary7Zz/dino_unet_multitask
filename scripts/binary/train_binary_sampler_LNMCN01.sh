#!/bin/bash

# 设置训练参数
Train_DATASET="LNM_CN01"
CUDA_VISIBLE_DEVICES="1"
METHOD="dino_unet"
TRAIN_IMAGE_PATH="/mnt/wangbd8/workspace/DataSets/ThyroidAgent/Classifaction_Data/Malignant_ultrasound_images_cropped/"
TRAIN_MASK_PATH="/mnt/wangbd8/workspace/DataSets/ThyroidAgent/Classifaction_Data/Malignant_ultrasound_images_cropped_predictions/"
TRAIN_LABEL_PATH="/mnt/wangbd8/workspace/DataSets/ThyroidAgent/Classifaction_Data/Malignant_ultrasound_images_cropped/train_labels.json"
EPOCH=50
LR=1e-4
BATCH_SIZE=12
CHECKPOINT_DIR="/mnt/wangbd8/workspace/ThyroidAgent/dino_unet_multitask/checkpoints"
CHECKPOINT_INTERVAL=5
EVAL_INTERVAL=50
DATASET_NAME="train_multitask_${Train_DATASET}_sampler"

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

DATASET_NAMES_ARGS=""
IMAGE_PATHS_ARGS=""
MASK_PATHS_ARGS=""
LABEL_PATHS_ARGS=""

for i in "${!TEST_DATASET_NAMES[@]}"; do
    DATASET_NAMES_ARGS="$DATASET_NAMES_ARGS ${TEST_DATASET_NAMES[$i]}"
done

for i in "${!TEST_IMAGE_PATHS[@]}"; do
    IMAGE_PATHS_ARGS="$IMAGE_PATHS_ARGS ${TEST_IMAGE_PATHS[$i]}"
done

for i in "${!TEST_MASK_PATHS[@]}"; do
    MASK_PATHS_ARGS="$MASK_PATHS_ARGS ${TEST_MASK_PATHS[$i]}"
done

for i in "${!TEST_LABEL_PATHS[@]}"; do
    LABEL_PATHS_ARGS="$LABEL_PATHS_ARGS ${TEST_LABEL_PATHS[$i]}"
done

python train_multitask_binary_sampler.py \
    --cuda_device $CUDA_VISIBLE_DEVICES \
    --method "$METHOD" \
    --train_image_path "$TRAIN_IMAGE_PATH" \
    --train_mask_path "$TRAIN_MASK_PATH" \
    --train_label_path "$TRAIN_LABEL_PATH" \
    --test_image_paths $IMAGE_PATHS_ARGS \
    --test_mask_paths $MASK_PATHS_ARGS \
    --test_label_paths $LABEL_PATHS_ARGS \
    --test_dataset_names $DATASET_NAMES_ARGS \
    --target_key "LNM_CN01" \
    --epoch $EPOCH \
    --lr $LR \
    --batch_size $BATCH_SIZE \
    --dir_checkpoint "$CHECKPOINT_DIR" \
    --checkpoint_interval $CHECKPOINT_INTERVAL \
    --eval_interval $EVAL_INTERVAL \
    --dataset_name "$DATASET_NAME" \
    --task_schedule "seg,cls"

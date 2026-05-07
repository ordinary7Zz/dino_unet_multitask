#!/bin/bash

CUDA_VISIBLE_DEVICES="0"
METHOD="dino_unet"
TARGET_KEY="FTCPTC"
TRAIN_IMAGE_PATH="/mnt/wangbd8/workspace/DataSets/ThyroidAgent/Classifaction_Data/Malignant_ultrasound_images_cropped/"
TRAIN_MASK_PATH="/mnt/wangbd8/workspace/DataSets/ThyroidAgent/Classifaction_Data/Malignant_ultrasound_images_cropped_predictions/"
TRAIN_LABEL_PATH="/mnt/wangbd8/workspace/DataSets/ThyroidAgent/Classifaction_Data/Malignant_ultrasound_images_cropped/train_labels_ftcptc_3000_1000.json"
TEST_IMAGE_PATH="/mnt/wangbd8/workspace/DataSets/ThyroidAgent/Classifaction_Data/Malignant_ultrasound_images_cropped/"
TEST_MASK_PATH="/mnt/wangbd8/workspace/DataSets/ThyroidAgent/Classifaction_Data/Malignant_ultrasound_images_cropped_predictions/"
TEST_LABEL_PATH="/mnt/wangbd8/workspace/DataSets/ThyroidAgent/Classifaction_Data/Malignant_ultrasound_images_cropped/test_labels.json"
TEST_DATASET_NAME="FTCPTC"
EPOCH=30
LR=2e-5
BATCH_SIZE=12
CHECKPOINT_DIR="/mnt/wangbd8/workspace/ThyroidAgent/dino_unet_multitask/checkpoints"
CHECKPOINT_INTERVAL=5
EVAL_INTERVAL=2
CLS_POS_WEIGHT=1
DATASET_NAME="train_multitask_FTCPTC_segcls"
TASK_SCHEDULE="seg,cls"

# 如果你已经生成了新的重采样标签文件，只需要修改 TRAIN_LABEL_PATH。
# 例如：
# TRAIN_LABEL_PATH="/mnt/wangbd8/workspace/ThyroidAgent/dino_unet_multitask/train_labels_ftcptc_2000_1000.json"

CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES" python train_multitask_binary_sampler.py \
    --cuda_device 0 \
    --method "$METHOD" \
    --train_image_path "$TRAIN_IMAGE_PATH" \
    --train_mask_path "$TRAIN_MASK_PATH" \
    --train_label_path "$TRAIN_LABEL_PATH" \
    --test_image_paths "$TEST_IMAGE_PATH" \
    --test_mask_paths "$TEST_MASK_PATH" \
    --test_label_paths "$TEST_LABEL_PATH" \
    --test_dataset_names "$TEST_DATASET_NAME" \
    --target_key "$TARGET_KEY" \
    --epoch $EPOCH \
    --lr $LR \
    --batch_size $BATCH_SIZE \
    --dir_checkpoint "$CHECKPOINT_DIR" \
    --checkpoint_interval $CHECKPOINT_INTERVAL \
    --eval_interval $EVAL_INTERVAL \
    --dataset_name "$DATASET_NAME" \
    --task_schedule "$TASK_SCHEDULE" \
    --cls_pos_weight $CLS_POS_WEIGHT

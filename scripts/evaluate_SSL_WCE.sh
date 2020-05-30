# Where the TFRecords are saved to.
DATASET_DIR=./tfrecord/

# Where the checkpoint and logs will be saved to.
DATASET_NAME=train
SAVE_NAME=models  
ATT_DIR=${SAVE_NAME}/result
CKPT_DIR=${SAVE_NAME}/checkpoint
LOG_DIR=${SAVE_NAME}/logs

# Model setting
MODEL_NAME=densenet,densenet
SPLIT_NAME=train

# Run training.
python3 eval_image_classifier.py \
    --dataset_name=${DATASET_NAME}\
    --split_name=${SPLIT_NAME} \
    --tfdata_path=${DATASET_DIR} \
    --attention_map=${ATT_DIR} \
    --checkpoint_dir=${CKPT_DIR} \
    --log_dir=${LOG_DIR} \
    --model_name=${MODEL_NAME} \
    --preprocessing_name=reid \
    --num_networks=2

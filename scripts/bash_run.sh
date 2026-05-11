#!/bin/bash

conda activate matryoshka_lora
cd ~/MatryoshkaLoRA

SCRIPT_PATH=""~/MatryoshkaLoRA/train.py"

LR=8e-5

DEVICE_BATCH_SIZE=8
GRAD_ACC_STEPS=4
EPOCHS=3

RANK=32
TRAIN_EVAL_RANKS="1,2,4,8,16,32,64,128,256" # removes ranks greater than $RANK

DATASET_NAME="gsm8k"
MODEL_NAME="llama3.2-1B-i"
ADAPTER_TYPE="matryoshka" # lora / dylora / matryoshka
MATRYOSHKA_MASK_TYPE="diag" # use "diag" for the Matryoshka: W = W_0 + A P B, with P = sum_{r in S} s_r A P_r B
EVAL_SHOTS="3,8"
SEED=42

WANDB_ENTITY=$1
WANDB_PROJECT="matryoshka_lora"

WANDB_GROUP="${MODEL_NAME}_${DATASET_NAME}_${ADAPTER_TYPE}_mask=${MATRYOSHKA_MASK_TYPE}_r=${RANK}_E=${EPOCHS}_s=${SCALING}_ter=${TRAIN_EVAL_RANKS}"
WANDB_JOB_TYPE="lr=${LR}"
WANDB_NAME="${WANDB_GROUP}_${WANDB_JOB_TYPE}_seed=${SEED}"

OUTPUT_FOLDER="./MatryoshkaLoRA/results/${WANDB_PROJECT}/${WANDB_NAME}"
mkdir -p ${OUTPUT_FOLDER}

python ${SCRIPT_PATH} \
    --wandb_entity ${WANDB_ENTITY} \
    --wandb_project ${WANDB_PROJECT} \
    --wandb_group ${WANDB_GROUP} \
    --wandb_job_type ${WANDB_JOB_TYPE} \
    --wandb_name ${WANDB_NAME} \
    \
    --dataset_name ${DATASET_NAME} \
    --device_batch_size ${device_batch_size} \
    --grad_acc_steps ${GRAD_ACC_STEPS} \
    --epochs ${EPOCHS} \
    --eval_shots ${EVAL_SHOTS} \
    --lr ${LR} \
    \
    --model_name ${MODEL_NAME} \
    --target_layers q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj \
    \
    --adapter_type ${ADAPTER_TYPE} \
    --matryoshka_mask_type ${MATRYOSHKA_MASK_TYPE} \
    --lora_scaling sqrt \
    \
    --rank ${RANK} \
    --train_ranks ${TRAIN_EVAL_RANKS} \
    --eval_ranks ${TRAIN_EVAL_RANKS} \
    \
    --seed ${SEED} \
    \
    --output_dir ${OUTPUT_FOLDER}

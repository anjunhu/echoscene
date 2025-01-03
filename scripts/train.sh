#!/bin/bash

# Define the experiment directory
expdir=/home/ubuntu/scene/EchoScene/scripts/checkpoint/finetune-20250102-linearprobe-echoscene-layoutlossonly
expdir=/home/ubuntu/scene/EchoScene/scripts/checkpoint/finetune-20250102-linearprobe-echoscene-overlaploss
ls $expdir/checkpoint

# Clear previous checkpoints in the experiment directory
if [ -d "$expdir/checkpoint" ]; then
    echo "Removing existing checkpoints in $expdir/checkpoint"
    rm -f $expdir/checkpoint/*
else
    echo "Checkpoint directory does not exist. Creating $expdir/checkpoint"
    mkdir -p $expdir/checkpoint
fi

# Copy the pre-trained model to the experiment directory
model_path=/home/ubuntu/scene/EchoScene/scripts/checkpoint/released_full_model/checkpoint/model2050.pth
if [ -f "$model_path" ]; then
    echo "Copying model from $model_path to $expdir/checkpoint"
    cp $model_path $expdir/checkpoint
else
    echo "Error: Model file $model_path does not exist."
    exit 1
fi

# Run the training script
python train_3dfront.py --exp $expdir \
       --room_type all \
       --dataset /home/ubuntu/datasets/FRONT \
       --loadepoch 2050 \
       --residual True \
       --network_type echoscene \
       --with_SDF True \
       --with_CLIP True \
       --separated False \
       --batchSize 64 \
       --workers 0 \
       --loadmodel True \
       --nepoch 20000 \
       --large False \
       --use_scene_rels True

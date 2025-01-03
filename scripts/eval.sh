#!/bin/bash

# Define starting and ending epochs
start_epoch=2050
end_epoch=2960
step=50
exp=finetune-20250102-linearprobe-echoscene-layoutlossonly

# Log file
log_file="/home/ubuntu/scene/EchoScene/scripts/checkpoint/finetune-20250102-linearprobe-echoscene-layoutlossonly/checkpoint/eval_log_$exp.txt"

# Clear the log file if it exists
> $log_file

# Loop through epochs
for (( epoch=$start_epoch; epoch<=$end_epoch; epoch+=$step ))
do
    echo "Running evaluation for epoch $epoch..."
    python eval_3dfront.py \
        --epoch $epoch --visualize True \
        --exp "checkpoint/$exp" \
        --dataset "/home/ubuntu/datasets/FRONT" \
        --room_type all --render_type onlybox --gen_shape True \
    | tail -n 10 >> $log_file
done

echo "All evaluations complete. Log saved to $log_file."

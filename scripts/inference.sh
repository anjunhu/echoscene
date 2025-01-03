python eval_3dfront.py \
  --epoch 2050 --visualize True \
  --exp "checkpoint/released_full_model" \
  --dataset "/home/ubuntu/datasets/FRONT" --visualize True \
  --room_type all --render_type onlybox --gen_shape True

# python eval_3dfront.py \
#   --epoch 2050 --visualize True \
#   --exp "checkpoint/finetune-20241231-linearprobe-echoscene" \
#   --dataset "/home/ubuntu/datasets/FRONT" --visualize True \
#   --room_type all --render_type onlybox --gen_shape True

# python eval_3dfront.py \
#   --epoch 2070 --visualize True \
#   --exp "checkpoint/finetune-20241230-lowerlr-obj_embeddings_ec" \
#   --dataset "/home/ubuntu/datasets/FRONT" --visualize True \
#   --room_type all --render_type onlybox --gen_shape True

# python eval_3dfront.py \
#   --epoch 2090 --visualize True \
#   --exp "checkpoint/finetune-20241228-lowerlr-gcn_layout_df_params" \
#   --dataset "/home/ubuntu/datasets/FRONT" --visualize True \
#   --room_type all --render_type onlybox --gen_shape True

# python eval_3dfront.py \
#   --epoch 2000 --visualize True \
#   --exp "checkpoint/released_layout_model" \
#   --dataset "/home/ubuntu/datasets/FRONT" --visualize True \
#   --room_type all --render_type onlybox --gen_shape True

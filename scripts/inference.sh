# python eval_3dfront.py \
#   --epoch 2050 --visualize True \
#   --exp "checkpoint/released_full_model" \
#   --dataset "/home/ubuntu/datasets/FRONT" --visualize True \
#   --room_type all --render_type echoscene --gen_shape True

python eval_3dfront.py \
  --epoch 2050 --visualize True \
  --exp "checkpoint/released_full_model" \
  --dataset "/home/ubuntu/datasets/FRONT" --visualize True \
  --room_type all --render_type onlybox --gen_shape True

# python eval_3dfront.py \
#   --epoch 2000 --visualize True \
#   --exp "checkpoint/released_layout_model" \
#   --dataset "/home/ubuntu/datasets/FRONT" --visualize True \
#   --room_type all --render_type onlybox --gen_shape True

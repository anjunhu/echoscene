import numpy as np
from plyfile import PlyData, PlyElement
pcd_path = "/home/ubuntu/scene/GaussianDreamer/outputs/MasterBedroom-33296/5_sweet_marjoram_bed@20241207-194240/save/it1200-test-color.ply"
pcd_path = "/home/ubuntu/scene/GaussianDreamer/outputs/MasterBedroom-33296/1_flower_laden_bower@20241207-190743/save/last_3dgs.ply"
pcd_data = PlyData.read(pcd_path)
print(pcd_data)

# import torch
# checkpoint_path = '/home/ubuntu/scene/GaussianDreamer/outputs/MasterBedroom-33296/1_flower_laden_bower@20241207-190743/ckpts/epoch=0-step=1200.ckpt'
# checkpoint = torch.load(checkpoint_path, map_location='cpu')
# print("Checkpoint loaded successfully.")
# print(checkpoint['state_dict'])

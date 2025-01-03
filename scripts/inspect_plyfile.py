import numpy as np
from plyfile import PlyData

# Read the PLY file
pcd_path = "/home/ubuntu/scene/EchoScene/scripts/checkpoint/released_full_model/vis-bkp-20250102/2050/combined_gs_MasterBedroom-33296.ply"
pcd_data = PlyData.read(pcd_path)

# Inspect the fields in the file
print("Available fields in the PLY data:")
print(pcd_data.elements[0].data.dtype.names)

# Extract and print rot_* data
for field_name in pcd_data.elements[0].data.dtype.names:
    if field_name.startswith("rot_"):
        print(f"{field_name}:")
        print(pcd_data.elements[0].data[field_name])


# import torch
# checkpoint_path = '/home/ubuntu/scene/GaussianDreamer/outputs/MasterBedroom-33296/1_flower_laden_bower@20241207-190743/ckpts/epoch=0-step=1200.ckpt'
# checkpoint = torch.load(checkpoint_path, map_location='cpu')
# print("Checkpoint loaded successfully.")
# print(checkpoint['state_dict'])

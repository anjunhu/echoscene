import torch
                                            
ckpt = torch.load("/home/ubuntu/scene/EchoScene/scripts/checkpoint/finetune-20241220/checkpoint/model90.pth")
for key, value in ckpt.items():
    try:
        size = value.size() 
    except AttributeError:
        size = value
    if "obj_embeddings" in key or "pred_embeddings" in key:
        print(f"{key}: {size}")
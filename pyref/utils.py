import torch
from PIL import Image
import numpy as np
import os

def save_tensor_as_png(tensor: torch.Tensor, filename: str) -> None:
    """
    Save a single-channel torch Tensor of shape (1, H, W) or (H, W) to a PNG file.

    Args:
        tensor:   torch.Tensor, shape (1, H, W) or (H, W), dtype float or uint8.
        filename: str, path to save the PNG (e.g. "out.png").
    """
    tensor = tensor.detach().cpu()
    
    if tensor.dim() == 3 and tensor.size(0) == 1:
        img = tensor.squeeze(0)
    elif tensor.dim() == 2:
        img = tensor
    else:
        raise ValueError(f"Expected tensor of shape (1,H,W) or (H,W), got {tensor.shape}")

    arr = img.numpy()

    if arr.dtype in [np.float32, np.float64]:
        arr = (arr * 255.0).clip(0, 255).astype(np.uint8)

    pil_img = Image.fromarray(arr, mode='L')

    pil_img.save(filename)

def save_image_batch(batch, save_folder='exp/figs/'):
    os.makedirs(save_folder, exist_ok=True)
    for i in range(len(batch)):
        img = batch[i]
        save_tensor_as_png(img, os.path.join(save_folder, f'sample{i}.png'))
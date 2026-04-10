
import os
import numpy as np
from PIL import Image
import torch
import matplotlib.pyplot as plt

def apply_colormap(label_img):
    """
    Apply RGB colormap to a 2D label image with class values [0, 1, 2]
    
    Args:
        label_img: 2D numpy array with class labels
    
    Returns:
        RGB image [H, W, 3]
    """
    colormap = {
        0: (200, 200, 200),  # Class 0 → Gray (Background)
        1: (230, 25, 75),    # Class 1 → Red (Shirorekha)
        2: (60, 180, 75),    # Class 2 → Green (Text body)
    }
    
    color_img = np.zeros((label_img.shape[0], label_img.shape[1], 3), dtype=np.uint8)
    for label, color in colormap.items():
        color_img[label_img == label] = color
    
    return color_img

def postprocess(output):
    """Convert 3-channel logits to class predictions [0, 1, 2]"""
    preds = torch.argmax(output, dim=1)
    return preds

def save_required_images(real_A, fake_B, real_B=None, epoch=None, batch_idx=None, sample_idx=0):
    """Save input, ground truth, generated image, and generated label"""
    try:
        os.makedirs('output_images', exist_ok=True)
        
        # Get predictions
        pred_B = postprocess(fake_B)
        
        # Process first sample in batch
        real_A_np = real_A[sample_idx, 0].cpu().numpy()
        pred_B_np = pred_B[sample_idx].cpu().numpy()
        
        # Convert input image from [-1,1] to [0,255]
        input_img = (real_A_np * 127.5 + 127.5).astype(np.uint8)
        
        # Create filename prefix
        if epoch is not None and batch_idx is not None:
            prefix = f"output_images/epoch{epoch}_batch{batch_idx}_sample{sample_idx}"
        else:
            prefix = f"output_images/sample{sample_idx}"
        
        # Save input image
        Image.fromarray(input_img).save(f'{prefix}_input.png')
        
        # Save generated label (color representation)
        color_pred = apply_colormap(pred_B_np)
        Image.fromarray(color_pred).save(f'{prefix}_generated_label.png')
        
        # Save ground truth if available
        if real_B is not None:
            real_B_np = real_B[sample_idx, 0].cpu().numpy()
            gt_img = (real_B_np * 127.5).astype(np.uint8)
            Image.fromarray(gt_img).save(f'{prefix}_ground_truth.png')
            
        print(f"Saved images with prefix: {prefix}")
        
    except Exception as e:
        print(f"Error saving images: {str(e)}")

def visualize_samples(real_A, fake_B, real_B=None, epoch=None, batch_idx=None, max_samples=1):
    """Visualize and save samples"""
    try:
        save_required_images(real_A, fake_B, real_B, epoch, batch_idx)
    except Exception as e:
        print(f"Visualization error: {str(e)}")

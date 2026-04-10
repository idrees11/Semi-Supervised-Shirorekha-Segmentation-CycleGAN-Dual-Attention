
import torch

def calculate_iou(pred, target, num_classes, ignore_index=None):
    """
    Compute mean IoU over num_classes
    
    Args:
        pred: Prediction tensor [B, H, W] or [H, W]
        target: Target tensor [B, H, W] or [H, W]
        num_classes: Number of classes
        ignore_index: Index to ignore in calculation
    """
    pred = pred.view(-1)
    target = target.view(-1)

    if ignore_index is not None:
        mask = target != ignore_index
        pred = pred[mask]
        target = target[mask]

    ious = []
    for c in range(num_classes):
        pred_c = pred == c
        target_c = target == c

        intersection = torch.sum(pred_c & target_c).float()
        union = torch.sum(pred_c | target_c).float()

        if union > 0:
            iou = intersection / (union + 1e-7)
            ious.append(iou)

    if len(ious) == 0:
        return torch.tensor(0.0, device=pred.device)

    return torch.mean(torch.stack(ious))

def evaluate(model, loader, num_classes=3, device='cuda'):
    """
    Evaluate model on validation/test set
    
    Args:
        model: Trained model
        loader: DataLoader
        num_classes: Number of classes
        device: Device to use
    """
    model.eval()
    intersection = torch.zeros(num_classes, device=device)
    union = torch.zeros(num_classes, device=device)

    with torch.no_grad():
        for real_A, real_B, is_paired in loader:
            if not is_paired.any():
                continue

            real_A = real_A.to(device)
            real_B = real_B.to(device)

            # Get predictions
            output = model.G_A2B(real_A)
            preds = torch.argmax(output, dim=1)
            targets = real_B.squeeze(1).long()

            for cls in range(num_classes):
                pred_mask = (preds == cls)
                target_mask = (targets == cls)

                intersection[cls] += (pred_mask & target_mask).sum()
                union[cls] += (pred_mask | target_mask).sum()

    # Compute per-class IoU
    iou_per_class = torch.zeros(num_classes, device=device)
    valid = union > 0
    iou_per_class[valid] = intersection[valid] / (union[valid] + 1e-7)

    mIoU = iou_per_class[valid].mean().item() if valid.any() else 0.0
    return mIoU, iou_per_class

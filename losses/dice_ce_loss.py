
import torch
import torch.nn as nn
import torch.nn.functional as F

class DiceCELoss(nn.Module):
    def __init__(self, alpha=0.5, smooth=1e-6):
        """
        Combined Dice and Cross Entropy Loss
        
        Args:
            alpha: Weight for Dice loss (1-alpha for CE loss)
            smooth: Smoothing factor for Dice loss
        """
        super().__init__()
        self.alpha = alpha
        self.smooth = smooth
        self.ce = nn.CrossEntropyLoss()

    def forward(self, pred, target):
        """
        Args:
            pred: Model predictions (logits) [B, C, H, W]
            target: Ground truth labels [B, H, W]
        """
        # Cross Entropy Loss
        ce_loss = self.ce(pred, target.long())

        # Dice Loss for multi-class
        pred_softmax = torch.softmax(pred, dim=1)
        target_onehot = F.one_hot(target, num_classes=pred.shape[1]).permute(0, 3, 1, 2).float()
        
        intersection = (pred_softmax * target_onehot).sum()
        union = pred_softmax.sum() + target_onehot.sum()
        dice_loss = 1 - (2. * intersection + self.smooth) / (union + self.smooth)

        return self.alpha * dice_loss + (1 - self.alpha) * ce_loss

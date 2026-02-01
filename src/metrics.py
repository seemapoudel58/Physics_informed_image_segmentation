import torch


def compute_dice_score(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    threshold: float = 0.5,
    smooth: float = 1e-6
) -> torch.Tensor:
    """
    Compute Dice score (F1 score) for binary segmentation.
    
    Args:
        predictions: Model output tensor (B, 1, H, W) with values in (0,1)
        targets: Ground truth binary mask (B, 1, H, W) with values in {0,1}
        threshold: Threshold for binarizing predictions (default: 0.5)
        smooth: Smoothing factor to avoid division by zero (default: 1e-6)
        
    Returns:
        Dice score (scalar tensor)
    """
    # Binarize predictions
    predictions_binary = (predictions > threshold).float()
    
    # Flatten tensors
    predictions_flat = predictions_binary.view(-1)
    targets_flat = targets.view(-1)
    
    # Compute intersection and union
    intersection = (predictions_flat * targets_flat).sum()
    dice = (2.0 * intersection + smooth) / (
        predictions_flat.sum() + targets_flat.sum() + smooth
    )
    
    return dice


def compute_dice_score_batch(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    threshold: float = 0.5,
    smooth: float = 1e-6
) -> torch.Tensor:
    """
    Compute Dice score for each sample in a batch.
    
    Args:
        predictions: Model output tensor (B, 1, H, W) with values in (0,1)
        targets: Ground truth binary mask (B, 1, H, W) with values in {0,1}
        threshold: Threshold for binarizing predictions (default: 0.5)
        smooth: Smoothing factor to avoid division by zero (default: 1e-6)
        
    Returns:
        Tensor of Dice scores for each sample (B,)
    """
    # Binarize predictions
    predictions_binary = (predictions > threshold).float()
    
    # Compute per-sample Dice scores
    batch_size = predictions.shape[0]
    dice_scores = torch.zeros(batch_size, device=predictions.device)
    
    for i in range(batch_size):
        pred_flat = predictions_binary[i].view(-1)
        target_flat = targets[i].view(-1)
        
        intersection = (pred_flat * target_flat).sum()
        dice = (2.0 * intersection + smooth) / (
            pred_flat.sum() + target_flat.sum() + smooth
        )
        dice_scores[i] = dice
    
    return dice_scores


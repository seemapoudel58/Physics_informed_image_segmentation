import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from scipy import stats
from scipy.spatial.distance import directed_hausdorff
import cv2
from torch.utils.data import DataLoader

# Handle both relative and absolute imports
try:
    from .metrics import compute_dice_score_batch
    from .dataset import CellSegmentationDataset
except ImportError:
    # Fallback for direct execution
    import sys
    from pathlib import Path as PathLib
    sys.path.insert(0, str(PathLib(__file__).parent.parent))
    from src.metrics import compute_dice_score_batch
    from src.dataset import CellSegmentationDataset


# Primary Segmentation Accuracy Metrics

def compute_iou(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    threshold: float = 0.5,
    smooth: float = 1e-6
) -> torch.Tensor:
    """
    Compute Intersection over Union (IoU) for binary segmentation.
    
    IoU = |A ∩ B| / |A ∪ B|
    
    Args:
        predictions: Model output tensor (B, 1, H, W) with values in (0,1)
        targets: Ground truth binary mask (B, 1, H, W) with values in {0,1}
        threshold: Threshold for binarizing predictions (default: 0.5)
        smooth: Smoothing factor to avoid division by zero (default: 1e-6)
        
    Returns:
        IoU score (scalar tensor)
    """
    # Binarize predictions
    predictions_binary = (predictions > threshold).float()
    
    # Flatten tensors
    predictions_flat = predictions_binary.view(-1)
    targets_flat = targets.view(-1)
    
    # Compute intersection and union
    intersection = (predictions_flat * targets_flat).sum()
    union = predictions_flat.sum() + targets_flat.sum() - intersection
    
    iou = (intersection + smooth) / (union + smooth)
    
    return iou


def compute_iou_batch(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    threshold: float = 0.5,
    smooth: float = 1e-6
) -> torch.Tensor:
    """
    Compute IoU for each sample in a batch.
    
    Args:
        predictions: Model output tensor (B, 1, H, W) with values in (0,1)
        targets: Ground truth binary mask (B, 1, H, W) with values in {0,1}
        threshold: Threshold for binarizing predictions (default: 0.5)
        smooth: Smoothing factor to avoid division by zero (default: 1e-6)
        
    Returns:
        Tensor of IoU scores for each sample (B,)
    """
    # Binarize predictions
    predictions_binary = (predictions > threshold).float()
    
    # Compute per-sample IoU scores
    batch_size = predictions.shape[0]
    iou_scores = torch.zeros(batch_size, device=predictions.device)
    
    for i in range(batch_size):
        pred_flat = predictions_binary[i].view(-1)
        target_flat = targets[i].view(-1)
        
        intersection = (pred_flat * target_flat).sum()
        union = pred_flat.sum() + target_flat.sum() - intersection
        
        iou = (intersection + smooth) / (union + smooth)
        iou_scores[i] = iou
    
    return iou_scores


# Boundary-Aware Metrics

def extract_boundaries(mask: np.ndarray) -> np.ndarray:
    """
    Extract boundary pixels from a binary mask.
    
    Args:
        mask: Binary mask (H, W) with values in {0, 1}
        
    Returns:
        Binary mask of boundary pixels (H, W)
    """
    # Convert to uint8 for OpenCV
    mask_uint8 = (mask * 255).astype(np.uint8)
    
    # Find contours
    contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    
    # Create boundary mask
    boundary_mask = np.zeros_like(mask_uint8)
    cv2.drawContours(boundary_mask, contours, -1, 255, 1)
    
    return (boundary_mask > 0).astype(np.float32)


def compute_boundary_f1(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    threshold: float = 0.5,
    tolerance: int = 2,
    smooth: float = 1e-6
) -> torch.Tensor:
    """
    Compute Boundary F1 Score within a tolerance distance.
    
    Args:
        predictions: Model output tensor (B, 1, H, W) with values in (0,1)
        targets: Ground truth binary mask (B, 1, H, W) with values in {0,1}
        threshold: Threshold for binarizing predictions (default: 0.5)
        tolerance: Pixel tolerance for boundary matching (default: 2)
        smooth: Smoothing factor to avoid division by zero (default: 1e-6)
        
    Returns:
        Boundary F1 score (scalar tensor)
    """
    # Binarize predictions
    predictions_binary = (predictions > threshold).float()
    
    # Convert to numpy for boundary extraction
    pred_np = predictions_binary[0, 0].cpu().numpy()
    target_np = targets[0, 0].cpu().numpy()
    
    # Extract boundaries
    pred_boundary = extract_boundaries(pred_np)
    target_boundary = extract_boundaries(target_np)
    
    # Create distance transform for tolerance
    if tolerance > 0:
        # Distance transform from target boundary
        target_dist = cv2.distanceTransform(
            (1 - target_boundary).astype(np.uint8),
            cv2.DIST_L2,
            5
        )
        
        # Predicted boundary pixels within tolerance
        pred_in_tolerance = (target_dist <= tolerance).astype(np.float32) * pred_boundary
        
        # Precision: predicted boundary pixels within tolerance / total predicted boundary
        precision = (pred_in_tolerance.sum() + smooth) / (pred_boundary.sum() + smooth)
        
        # Distance transform from predicted boundary
        pred_dist = cv2.distanceTransform(
            (1 - pred_boundary).astype(np.uint8),
            cv2.DIST_L2,
            5
        )
        
        # Target boundary pixels within tolerance
        target_in_tolerance = (pred_dist <= tolerance).astype(np.float32) * target_boundary
        
        # Recall: target boundary pixels within tolerance / total target boundary
        recall = (target_in_tolerance.sum() + smooth) / (target_boundary.sum() + smooth)
        
        # F1 score
        f1 = (2.0 * precision * recall + smooth) / (precision + recall + smooth)
    else:
        # Exact boundary matching
        intersection = (pred_boundary * target_boundary).sum()
        f1 = (2.0 * intersection + smooth) / (
            pred_boundary.sum() + target_boundary.sum() + smooth
        )
    
    return torch.tensor(f1, dtype=torch.float32)


def compute_boundary_f1_batch(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    threshold: float = 0.5,
    tolerance: int = 2,
    smooth: float = 1e-6
) -> torch.Tensor:
    """
    Compute Boundary F1 Score for each sample in a batch.
    
    Args:
        predictions: Model output tensor (B, 1, H, W) with values in (0,1)
        targets: Ground truth binary mask (B, 1, H, W) with values in {0,1}
        threshold: Threshold for binarizing predictions (default: 0.5)
        tolerance: Pixel tolerance for boundary matching (default: 2)
        smooth: Smoothing factor to avoid division by zero (default: 1e-6)
        
    Returns:
        Tensor of Boundary F1 scores for each sample (B,)
    """
    batch_size = predictions.shape[0]
    f1_scores = torch.zeros(batch_size)
    
    for i in range(batch_size):
        f1 = compute_boundary_f1(
            predictions[i:i+1],
            targets[i:i+1],
            threshold=threshold,
            tolerance=tolerance,
            smooth=smooth
        )
        f1_scores[i] = f1
    
    return f1_scores


def compute_hausdorff_distance(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    threshold: float = 0.5
) -> float:
    """
    Compute Hausdorff Distance between predicted and target boundaries.
    
    H(A, B) = max(sup_{a in A} inf_{b in B} d(a,b), sup_{b in B} inf_{a in A} d(b,a))
    
    Args:
        predictions: Model output tensor (B, 1, H, W) with values in (0,1)
        targets: Ground truth binary mask (B, 1, H, W) with values in {0,1}
        threshold: Threshold for binarizing predictions (default: 0.5)
        
    Returns:
        Hausdorff distance (float)
    """
    # Binarize predictions
    predictions_binary = (predictions > threshold).float()
    
    # Convert to numpy
    pred_np = predictions_binary[0, 0].cpu().numpy()
    target_np = targets[0, 0].cpu().numpy()
    
    # Extract boundaries
    pred_boundary = extract_boundaries(pred_np)
    target_boundary = extract_boundaries(target_np)
    
    # Get boundary coordinates
    pred_coords = np.column_stack(np.where(pred_boundary > 0))
    target_coords = np.column_stack(np.where(target_boundary > 0))
    
    if len(pred_coords) == 0 or len(target_coords) == 0:
        # If no boundaries found, return a large distance
        return float('inf')
    
    # Compute Hausdorff distance
    hausdorff_dist = max(
        directed_hausdorff(pred_coords, target_coords)[0],
        directed_hausdorff(target_coords, pred_coords)[0]
    )
    
    return hausdorff_dist


# Evaluation Function
def evaluate_model(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    threshold: float = 0.5
) -> Dict[str, np.ndarray]:
    """
    Evaluate a model on a dataset and compute all metrics.
    
    Args:
        model: Trained segmentation model
        dataloader: DataLoader for the evaluation dataset
        device: Device to run evaluation on
        threshold: Threshold for binarizing predictions (default: 0.5)
        
    Returns:
        Dictionary containing arrays of per-image metrics:
        - dice_scores: Dice coefficient for each image
        - iou_scores: IoU for each image
        - boundary_f1_scores: Boundary F1 score for each image
        - hausdorff_distances: Hausdorff distance for each image
    """
    model.eval()
    
    all_dice = []
    all_iou = []
    all_boundary_f1 = []
    all_hausdorff = []
    
    with torch.no_grad():
        for images, masks in dataloader:
            images = images.to(device)
            masks = masks.to(device)
            
            # Forward pass
            outputs = model(images)
            
            # Compute Dice scores
            dice_batch = compute_dice_score_batch(outputs, masks, threshold=threshold)
            all_dice.extend(dice_batch.cpu().numpy())
            
            # Compute IoU scores
            iou_batch = compute_iou_batch(outputs, masks, threshold=threshold)
            all_iou.extend(iou_batch.cpu().numpy())
            
            # Compute Boundary F1 scores
            boundary_f1_batch = compute_boundary_f1_batch(
                outputs, masks, threshold=threshold, tolerance=2
            )
            all_boundary_f1.extend(boundary_f1_batch.numpy())
            
            # Compute Hausdorff distances (slower, so we do it per image)
            for i in range(outputs.shape[0]):
                hausdorff = compute_hausdorff_distance(
                    outputs[i:i+1], masks[i:i+1], threshold=threshold
                )
                if np.isfinite(hausdorff):
                    all_hausdorff.append(hausdorff)
                else:
                    all_hausdorff.append(np.nan)
    
    return {
        'dice_scores': np.array(all_dice),
        'iou_scores': np.array(all_iou),
        'boundary_f1_scores': np.array(all_boundary_f1),
        'hausdorff_distances': np.array(all_hausdorff)
    }


# Statistical Analysis
def compute_statistics(metric_array: np.ndarray) -> Dict[str, float]:
    """
    Compute mean and standard deviation for a metric array.
    
    Args:
        metric_array: Array of metric values
        
    Returns:
        Dictionary with 'mean' and 'std'
    """
    # Filter out NaN values
    valid_values = metric_array[~np.isnan(metric_array)]
    
    if len(valid_values) == 0:
        return {'mean': np.nan, 'std': np.nan, 'count': 0}
    
    return {
        'mean': float(np.mean(valid_values)),
        'std': float(np.std(valid_values, ddof=1)),  # Sample std
        'count': len(valid_values)
    }


def compare_models_statistically(
    metrics_baseline: Dict[str, np.ndarray],
    metrics_pde: Dict[str, np.ndarray],
    alpha: float = 0.05
) -> Dict[str, Dict[str, float]]:
    """
    Perform statistical comparison between baseline and PDE-constrained models.
    
    Uses paired t-test and Wilcoxon signed-rank test for each metric.
    
    Args:
        metrics_baseline: Dictionary of metric arrays for baseline model
        metrics_pde: Dictionary of metric arrays for PDE-constrained model
        alpha: Significance level (default: 0.05)
        
    Returns:
        Dictionary with statistical test results for each metric
    """
    results = {}
    
    for metric_name in metrics_baseline.keys():
        baseline_values = metrics_baseline[metric_name]
        pde_values = metrics_pde[metric_name]
        
        # Filter out NaN values
        valid_mask = ~(np.isnan(baseline_values) | np.isnan(pde_values))
        baseline_clean = baseline_values[valid_mask]
        pde_clean = pde_values[valid_mask]
        
        if len(baseline_clean) < 2:
            results[metric_name] = {
                't_statistic': np.nan,
                't_pvalue': np.nan,
                'wilcoxon_statistic': np.nan,
                'wilcoxon_pvalue': np.nan,
                'significant': False
            }
            continue
        
        # Paired t-test
        t_stat, t_pvalue = stats.ttest_rel(baseline_clean, pde_clean)
        
        # Wilcoxon signed-rank test
        wilcoxon_stat, wilcoxon_pvalue = stats.wilcoxon(
            baseline_clean, pde_clean, alternative='two-sided'
        )
        
        # Determine significance
        significant = (t_pvalue < alpha) or (wilcoxon_pvalue < alpha)
        
        baseline_stats = compute_statistics(baseline_clean)
        pde_stats = compute_statistics(pde_clean)
        
        results[metric_name] = {
            't_statistic': float(t_stat),
            't_pvalue': float(t_pvalue),
            'wilcoxon_statistic': float(wilcoxon_stat),
            'wilcoxon_pvalue': float(wilcoxon_pvalue),
            'significant': significant,
            'baseline_mean': baseline_stats['mean'],
            'baseline_std': baseline_stats['std'],
            'pde_mean': pde_stats['mean'],
            'pde_std': pde_stats['std'],
            'improvement': float(np.mean(pde_clean) - np.mean(baseline_clean))
        }
    
    return results


def format_metric_report(
    metrics: Dict[str, np.ndarray],
    model_name: str = "Model"
) -> str:
    """
    Format metrics as a report string with mean ± std format.
    
    Args:
        metrics: Dictionary of metric arrays
        model_name: Name of the model for the report
        
    Returns:
        Formatted report string
    """
    report_lines = [f"\n{model_name} Performance:"]
    report_lines.append("=" * 60)
    
    for metric_name, metric_array in metrics.items():
        stats_dict = compute_statistics(metric_array)
        
        if stats_dict['count'] > 0:
            report_lines.append(
                f"{metric_name.replace('_', ' ').title()}: "
                f"{stats_dict['mean']:.4f} ± {stats_dict['std']:.4f} "
                f"(n={stats_dict['count']})"
            )
        else:
            report_lines.append(
                f"{metric_name.replace('_', ' ').title()}: N/A"
            )
    
    return "\n".join(report_lines)


# Main Evaluation Function
def evaluate_on_test_set(
    model: nn.Module,
    test_dir: Path,
    test_json: Path,
    device: torch.device,
    batch_size: int = 8,
    threshold: float = 0.5,
    model_name: str = "Model"
) -> Dict[str, np.ndarray]:
    """
    Evaluate a model on the test set (BV2 cell type).
    
    Args:
        model: Trained segmentation model
        test_dir: Directory containing test images
        test_json: Path to test annotations JSON file
        device: Device to run evaluation on
        batch_size: Batch size for evaluation (default: 8)
        threshold: Threshold for binarizing predictions (default: 0.5)
        model_name: Name of the model for reporting
        
    Returns:
        Dictionary containing arrays of per-image metrics
    """
    print(f"\nEvaluating {model_name} on test set...")
    print("=" * 70)
    
    # Create test dataset
    test_dataset = CellSegmentationDataset(test_dir, test_json)
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=True if torch.cuda.is_available() else False
    )
    
    print(f"Test samples: {len(test_dataset)}")
    
    # Evaluate model
    metrics = evaluate_model(model, test_loader, device, threshold=threshold)
    
    # Print report
    report = format_metric_report(metrics, model_name=model_name)
    print(report)
    
    return metrics


"""
Two-stage training strategy for PDE-constrained cell segmentation.

Stage I: Baseline (unconstrained) training with Dice + BCE
Stage II: PDE-constrained fine-tuning with Dice + BCE + PDE regularization
"""

import torch
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from pathlib import Path
import numpy as np
import csv
import json
from datetime import datetime
from typing import Optional, Tuple, Dict, List

from .unet import UNet
from .dataset import CellSegmentationDataset
from .loss import DiceBCELoss, DiceBCEPDELoss
from .metrics import compute_dice_score, compute_dice_score_batch
from .evaluate import (
    compute_iou_batch, 
    compute_boundary_f1_batch,
    evaluate_model,
    evaluate_on_test_set,
    compute_statistics
)
from .plot import plot_training_results


class EarlyStopping:
    """Early stopping based on validation Dice score."""
    
    def __init__(
        self,
        patience: int = 10,
        min_delta: float = 1e-4,
        mode: str = 'max'
    ):
        """
        Args:
            patience: Number of epochs to wait before stopping
            min_delta: Minimum change to qualify as improvement
            mode: 'max' for Dice score (higher is better), 'min' for loss
        """
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_score = None
        self.best_epoch = 0
        self.early_stop = False
    
    def __call__(self, score: float, epoch: int) -> bool:
        """
        Check if training should stop.
        
        Returns:
            True if training should stop, False otherwise
        """
        if self.best_score is None:
            self.best_score = score
            self.best_epoch = epoch
            return False
        
        if self.mode == 'max':
            improved = score > self.best_score + self.min_delta
        else:
            improved = score < self.best_score - self.min_delta
        
        if improved:
            self.best_score = score
            self.best_epoch = epoch
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        
        return self.early_stop


def train_epoch(
    model,
    dataloader,
    criterion,
    optimizer,
    device,
    return_components: bool = False,
    compute_metrics: bool = True
) -> Dict[str, float]:
    """Train for one epoch."""
    model.train()
    total_loss = 0.0
    total_dice_loss = 0.0
    total_bce_loss = 0.0
    total_pde_loss = 0.0
    total_phase_field_loss = 0.0
    
    # Metrics accumulators
    all_dice_scores = []
    all_iou_scores = []
    all_boundary_f1_scores = []
    
    num_batches = 0
    
    for images, masks in dataloader:
        images = images.to(device)
        masks = masks.to(device)
        
        # Forward pass
        optimizer.zero_grad()
        outputs = model(images)
        
        # Compute loss
        loss = criterion(outputs, masks)
        
        # Extract loss components and compute metrics if requested
        if return_components or compute_metrics:
            with torch.no_grad():
                # Compute individual components (works for both DiceBCELoss and DiceBCEPDELoss)
                predictions_flat = outputs.view(-1)
                targets_flat = masks.view(-1)
                smooth = criterion.smooth
                
                # Dice loss
                intersection = (predictions_flat * targets_flat).sum()
                dice = (2.0 * intersection + smooth) / (
                    predictions_flat.sum() + targets_flat.sum() + smooth
                )
                dice_loss = 1 - dice
                
                # BCE loss
                bce_loss = criterion.bce(outputs, masks)
                
                if return_components:
                    total_dice_loss += dice_loss.item()
                    total_bce_loss += bce_loss.item()
                
                # PDE losses (only for DiceBCEPDELoss)
                if return_components and isinstance(criterion, DiceBCEPDELoss):
                    if criterion.pde_weight > 0:
                        pde_loss = criterion.pde_regularization.compute_loss(outputs)
                        total_pde_loss += pde_loss.item()
                    if criterion.phase_field_weight > 0:
                        phase_field_loss = criterion.pde_regularization.compute_phase_field_loss(
                            outputs, epsilon=criterion.epsilon
                        )
                        total_phase_field_loss += phase_field_loss.item()
                
                # Compute additional metrics
                if compute_metrics:
                    dice_batch = compute_dice_score_batch(outputs, masks, threshold=0.5)
                    iou_batch = compute_iou_batch(outputs, masks, threshold=0.5)
                    boundary_f1_batch = compute_boundary_f1_batch(outputs, masks, threshold=0.5, tolerance=2)
                    
                    all_dice_scores.extend(dice_batch.cpu().numpy())
                    all_iou_scores.extend(iou_batch.cpu().numpy())
                    all_boundary_f1_scores.extend(boundary_f1_batch.cpu().numpy())
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        num_batches += 1
    
    results = {'loss': total_loss / num_batches}
    
    if return_components:
        results['dice_loss'] = total_dice_loss / num_batches
        results['bce_loss'] = total_bce_loss / num_batches
        if isinstance(criterion, DiceBCEPDELoss):
            if criterion.pde_weight > 0:
                results['pde_loss'] = total_pde_loss / num_batches
            if criterion.phase_field_weight > 0:
                results['phase_field_loss'] = total_phase_field_loss / num_batches
    
    if compute_metrics:
        results['dice_score'] = np.mean(all_dice_scores) if all_dice_scores else 0.0
        results['iou_score'] = np.mean(all_iou_scores) if all_iou_scores else 0.0
        results['boundary_f1_score'] = np.mean(all_boundary_f1_scores) if all_boundary_f1_scores else 0.0
    
    return results


def validate(
    model,
    dataloader,
    criterion,
    device,
    return_components: bool = False,
    compute_metrics: bool = True
) -> Dict[str, float]:
    """Validate the model."""
    model.eval()
    total_loss = 0.0
    total_dice_score = 0.0
    total_dice_loss = 0.0
    total_bce_loss = 0.0
    total_pde_loss = 0.0
    total_phase_field_loss = 0.0
    
    # Metrics accumulators
    all_dice_scores = []
    all_iou_scores = []
    all_boundary_f1_scores = []
    
    num_batches = 0
    
    with torch.no_grad():
        for images, masks in dataloader:
            images = images.to(device)
            masks = masks.to(device)
            
            outputs = model(images)
            loss = criterion(outputs, masks)
            
            # Compute Dice score (for backward compatibility)
            dice_score = compute_dice_score(outputs, masks)
            total_dice_score += dice_score.item()
            
            # Extract loss components and compute metrics if requested
            if return_components or compute_metrics:
                predictions_flat = outputs.view(-1)
                targets_flat = masks.view(-1)
                smooth = criterion.smooth
                
                # Dice loss
                intersection = (predictions_flat * targets_flat).sum()
                dice = (2.0 * intersection + smooth) / (
                    predictions_flat.sum() + targets_flat.sum() + smooth
                )
                dice_loss = 1 - dice
                
                # BCE loss
                bce_loss = criterion.bce(outputs, masks)
                
                if return_components:
                    total_dice_loss += dice_loss.item()
                    total_bce_loss += bce_loss.item()
                
                # PDE losses (only for DiceBCEPDELoss)
                if return_components and isinstance(criterion, DiceBCEPDELoss):
                    if criterion.pde_weight > 0:
                        pde_loss = criterion.pde_regularization.compute_loss(outputs)
                        total_pde_loss += pde_loss.item()
                    if criterion.phase_field_weight > 0:
                        phase_field_loss = criterion.pde_regularization.compute_phase_field_loss(
                            outputs, epsilon=criterion.epsilon
                        )
                        total_phase_field_loss += phase_field_loss.item()
                
                # Compute additional metrics
                if compute_metrics:
                    dice_batch = compute_dice_score_batch(outputs, masks, threshold=0.5)
                    iou_batch = compute_iou_batch(outputs, masks, threshold=0.5)
                    boundary_f1_batch = compute_boundary_f1_batch(outputs, masks, threshold=0.5, tolerance=2)
                    
                    all_dice_scores.extend(dice_batch.cpu().numpy())
                    all_iou_scores.extend(iou_batch.cpu().numpy())
                    all_boundary_f1_scores.extend(boundary_f1_batch.cpu().numpy())
            
            total_loss += loss.item()
            num_batches += 1
    
    results = {
        'loss': total_loss / num_batches,
        'dice_score': total_dice_score / num_batches
    }
    
    if return_components:
        results['dice_loss'] = total_dice_loss / num_batches
        results['bce_loss'] = total_bce_loss / num_batches
        if isinstance(criterion, DiceBCEPDELoss):
            if criterion.pde_weight > 0:
                results['pde_loss'] = total_pde_loss / num_batches
            if criterion.phase_field_weight > 0:
                results['phase_field_loss'] = total_phase_field_loss / num_batches
    
    if compute_metrics:
        results['iou_score'] = np.mean(all_iou_scores) if all_iou_scores else 0.0
        results['boundary_f1_score'] = np.mean(all_boundary_f1_scores) if all_boundary_f1_scores else 0.0
    
    return results


def train_stage(
    model,
    train_loader,
    val_loader,
    criterion,
    optimizer,
    device,
    num_epochs: int,
    stage_name: str,
    early_stopping: Optional[EarlyStopping] = None,
    verbose: bool = True,
    csv_path: Optional[Path] = None
) -> Tuple[Dict, int, List[Dict]]:
    """
    Train for a single stage.
    
    Args:
        csv_path: Optional path to save metrics CSV file
    
    Returns:
        Tuple of (best_metrics, best_epoch, all_epoch_metrics)
    """
    best_val_dice = 0.0
    best_epoch = 0
    best_metrics = {}
    all_metrics = []  # Store metrics for all epochs
    
    for epoch in range(num_epochs):
        # Training
        train_results = train_epoch(
            model, train_loader, criterion, optimizer, device,
            return_components=True,
            compute_metrics=True
        )
        
        # Validation
        val_results = validate(
            model, val_loader, criterion, device,
            return_components=True,
            compute_metrics=True
        )
        
        # Track best model
        if val_results['dice_score'] > best_val_dice:
            best_val_dice = val_results['dice_score']
            best_epoch = epoch + 1
            best_metrics = {
                'train': train_results,
                'val': val_results
            }
        
        # Store metrics for this epoch
        epoch_metrics = {
            'epoch': epoch + 1,
            'train_loss': train_results['loss'],
            'train_dice_loss': train_results.get('dice_loss', 0.0),
            'train_bce_loss': train_results.get('bce_loss', 0.0),
            'train_pde_loss': train_results.get('pde_loss', 0.0),
            'train_phase_field_loss': train_results.get('phase_field_loss', 0.0),
            'train_dice_score': train_results.get('dice_score', 0.0),
            'train_iou_score': train_results.get('iou_score', 0.0),
            'train_boundary_f1_score': train_results.get('boundary_f1_score', 0.0),
            'val_loss': val_results['loss'],
            'val_dice_score': val_results['dice_score'],
            'val_dice_loss': val_results.get('dice_loss', 0.0),
            'val_bce_loss': val_results.get('bce_loss', 0.0),
            'val_pde_loss': val_results.get('pde_loss', 0.0),
            'val_phase_field_loss': val_results.get('phase_field_loss', 0.0),
            'val_iou_score': val_results.get('iou_score', 0.0),
            'val_boundary_f1_score': val_results.get('boundary_f1_score', 0.0)
        }
        all_metrics.append(epoch_metrics)
        
        # Save to CSV after each epoch
        if csv_path is not None:
            save_metrics_to_csv(all_metrics, csv_path)
        
        # Print progress
        if verbose:
            print(f"\n{stage_name} - Epoch {epoch+1}/{num_epochs}")
            print(f"  Train Loss: {train_results['loss']:.6f}")
            if 'dice_loss' in train_results:
                print(f"    - Dice Loss: {train_results['dice_loss']:.6f}")
                print(f"    - BCE Loss: {train_results['bce_loss']:.6f}")
                if 'pde_loss' in train_results:
                    print(f"    - PDE Loss: {train_results['pde_loss']:.6f}")
            print(f"  Val Loss: {val_results['loss']:.6f}")
            print(f"  Val Dice Score: {val_results['dice_score']:.6f}")
            if 'dice_loss' in val_results:
                print(f"    - Dice Loss: {val_results['dice_loss']:.6f}")
                print(f"    - BCE Loss: {val_results['bce_loss']:.6f}")
                if 'pde_loss' in val_results:
                    print(f"    - PDE Loss: {val_results['pde_loss']:.6f}")
        
        # Early stopping
        if early_stopping is not None:
            if early_stopping(val_results['dice_score'], epoch + 1):
                if verbose:
                    print(f"\nEarly stopping triggered at epoch {epoch+1}")
                    print(f"Best validation Dice score: {best_val_dice:.6f} at epoch {best_epoch}")
                break
    
    return best_metrics, best_epoch, all_metrics


def save_metrics_to_csv(metrics: List[Dict], csv_path: Path):
    """
    Save training metrics to a CSV file.
    
    Args:
        metrics: List of dictionaries containing metrics for each epoch
        csv_path: Path to CSV file
    """
    if not metrics:
        return
    
    # Create output directory if it doesn't exist
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Get all possible keys (handles different stages with/without PDE)
    fieldnames = [
        'epoch',
        'train_loss',
        'train_dice_loss',
        'train_bce_loss',
        'train_pde_loss',
        'train_phase_field_loss',
        'train_dice_score',
        'train_iou_score',
        'train_boundary_f1_score',
        'val_loss',
        'val_dice_score',
        'val_dice_loss',
        'val_bce_loss',
        'val_pde_loss',
        'val_phase_field_loss',
        'val_iou_score',
        'val_boundary_f1_score'
    ]
    
    # Write CSV file (overwrite each time to ensure latest data)
    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(metrics)


def save_test_metrics(
    test_metrics: Dict[str, np.ndarray],
    output_path: Path,
    model_name: str = "Model"
):
    """
    Save test metrics to both CSV and JSON files.
    
    Args:
        test_metrics: Dictionary containing arrays of per-image metrics
        output_path: Base path for output files (without extension)
        model_name: Name of the model for reporting
    """
    # Create output directory if it doesn't exist
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Compute statistics
    stats_dict = {}
    for metric_name, metric_array in test_metrics.items():
        stats = compute_statistics(metric_array)
        stats_dict[metric_name] = stats
    
    # Save as JSON (with statistics)
    json_path = output_path.with_suffix('.json')
    json_data = {
        'model_name': model_name,
        'statistics': {k: {'mean': float(v['mean']), 'std': float(v['std']), 'count': int(v['count'])} 
                      for k, v in stats_dict.items()},
        'per_image_metrics': {k: v.tolist() for k, v in test_metrics.items()}
    }
    
    with open(json_path, 'w') as f:
        json.dump(json_data, f, indent=2)
    
    # Save as CSV (per-image metrics)
    csv_path = output_path.with_suffix('.csv')
    # Get the maximum length to pad shorter arrays
    max_len = max(len(v) for v in test_metrics.values())
    
    # Create rows for CSV
    rows = []
    for i in range(max_len):
        row = {}
        for metric_name, metric_array in test_metrics.items():
            if i < len(metric_array):
                value = metric_array[i]
                row[metric_name] = value if np.isfinite(value) else np.nan
            else:
                row[metric_name] = np.nan
        rows.append(row)
    
    # Write CSV
    if rows:
        fieldnames = list(test_metrics.keys())
        with open(csv_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for row in rows:
                # Convert numpy types to Python types, handling NaN and inf
                clean_row = {}
                for k, v in row.items():
                    if isinstance(v, (np.floating, float)):
                        if np.isnan(v) or np.isinf(v):
                            clean_row[k] = ''
                        else:
                            clean_row[k] = float(v)
                    else:
                        clean_row[k] = v
                writer.writerow(clean_row)
    
    print(f"Test metrics saved to:")
    print(f"  CSV: {csv_path}")
    print(f"  JSON: {json_path}")


def create_subset_dataset(
    dataset: CellSegmentationDataset,
    fraction: float
) -> Subset:
    """
    Create a subset of the dataset for low-label training.
    
    Args:
        dataset: Full dataset
        fraction: Fraction of data to use (e.g., 0.1 for 10%, 0.25 for 25%)
        
    Returns:
        Subset of the dataset
    """
    total_size = len(dataset)
    subset_size = int(total_size * fraction)
    indices = np.random.choice(total_size, subset_size, replace=False)
    return Subset(dataset, indices)


def train(
    use_two_stage: bool = True,
    pde_weight: float = 1e-4,
    diffusion_coeff: float = 5.0,
    reaction_threshold: float = 0.5,
    phase_field_weight: float = 1e-4,
    epsilon: float = 0.05,
    batch_size: int = 8,
    learning_rate: float = 1e-4,
    stage1_epochs: int = 50,
    stage2_epochs: int = 50,
    early_stopping_patience: int = 10,
    train_fraction: Optional[float] = None,
    seed: int = 42
):
    """
    Two-stage training strategy for PDE-constrained cell segmentation.
    
    Stage I: Baseline training (Dice + BCE only)
    Stage II: PDE-constrained fine-tuning (Dice + BCE + PDE + phase-field)
    
    Args:
        use_two_stage: If True, use two-stage training. If False, use single-stage
                       with PDE from the start (default: True)
        pde_weight: Weight for reaction-diffusion PDE regularization λ_RD (default: 1e-4, optimal)
        diffusion_coeff: Diffusion coefficient D > 0 for PDE (default: 5.0, optimal)
        reaction_threshold: Reaction term threshold a ∈ (0,1) for PDE (default: 0.5, optimal)
        phase_field_weight: Weight for phase-field energy λ_PF (default: 1e-4, optimal)
        epsilon: Interface width parameter for phase-field energy (default: 0.05, optimal)
        batch_size: Batch size for training (default: 8, recommended: 8-16)
        learning_rate: Learning rate for AdamW optimizer (default: 1e-4)
        stage1_epochs: Maximum epochs for Stage I (baseline) (default: 50)
        stage2_epochs: Maximum epochs for Stage II (PDE fine-tuning) (default: 50)
        early_stopping_patience: Patience for early stopping (default: 10)
        train_fraction: Fraction of training data to use (None = 100%, 0.1 = 10%, 0.25 = 25%)
        seed: Random seed for reproducibility (default: 42)
    """
    # Set random seed
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
    
    # Configuration
    BASE_DIR = Path(__file__).parent.parent
    IMG_DIR = BASE_DIR / 'images'
    TRAIN_DIR = IMG_DIR / 'training'
    VAL_DIR = IMG_DIR / 'validation'
    TEST_DIR = IMG_DIR / 'testing'
    ANN_DIR = IMG_DIR / 'annotation'
    OUTPUT_DIR = BASE_DIR / 'output'
    
    TRAIN_JSON = ANN_DIR / 'training_annotation.json'
    VAL_JSON = ANN_DIR / 'validation_annotation.json'
    TEST_JSON = ANN_DIR / 'testing_annotation.json'
    
    # Create output directory
    OUTPUT_DIR.mkdir(exist_ok=True)
    
    # Generate timestamp for unique file names
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print("=" * 70)
    print("PDE-CONSTRAINED CELL SEGMENTATION TRAINING")
    print("=" * 70)
    print(f"Device: {device}")
    print(f"Training strategy: {'Two-stage' if use_two_stage else 'Single-stage (PDE from start)'}")
    
    # Create datasets
    print("\nLoading datasets...")
    train_dataset = CellSegmentationDataset(TRAIN_DIR, TRAIN_JSON)
    val_dataset = CellSegmentationDataset(VAL_DIR, VAL_JSON)
    
    # Apply low-label training if specified
    if train_fraction is not None:
        print(f"Using {train_fraction*100:.1f}% of training data ({int(len(train_dataset) * train_fraction)} samples)")
        train_dataset = create_subset_dataset(train_dataset, train_fraction)
    
    # Create CSV file paths
    if train_fraction is not None:
        fraction_str = f"_frac{train_fraction:.2f}"
    else:
        fraction_str = ""
    
    csv_path_stage1 = OUTPUT_DIR / f"metrics_stage1_{timestamp}{fraction_str}.csv"
    csv_path_stage2 = OUTPUT_DIR / f"metrics_stage2_{timestamp}{fraction_str}.csv"
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=True if torch.cuda.is_available() else False
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=True if torch.cuda.is_available() else False
    )
    
    print(f"Training samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")
    print(f"Batch size: {batch_size}")
    
    # Create UNet model
    print(f"\nCreating UNet model...")
    model = UNet(
        in_channels=1,
        out_channels=1,
        base_channels=64
    )
    model = model.to(device)
    
    # ========================================================================
    # STAGE I: Baseline Training (Unconstrained)
    # ========================================================================
    print("\n" + "=" * 70)
    print("STAGE I: BASELINE TRAINING (Unconstrained)")
    print("=" * 70)
    print("Objective: L = L_Dice + L_BCE")
    
    criterion_stage1 = DiceBCELoss(dice_weight=0.5, bce_weight=0.5)
    criterion_stage1 = criterion_stage1.to(device)
    optimizer_stage1 = optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=1e-5
    )
    
    early_stopping_stage1 = EarlyStopping(
        patience=early_stopping_patience,
        min_delta=1e-4,
        mode='max'
    )
    
    best_stage1_metrics, best_stage1_epoch, stage1_all_metrics = train_stage(
        model,
        train_loader,
        val_loader,
        criterion_stage1,
        optimizer_stage1,
        device,
        num_epochs=stage1_epochs,
        stage_name="Stage I",
        early_stopping=early_stopping_stage1,
        verbose=True,
        csv_path=csv_path_stage1
    )
    
    print(f"\nStage I complete. Best validation Dice: {best_stage1_metrics['val']['dice_score']:.6f} at epoch {best_stage1_epoch}")
    print(f"Stage I metrics saved to: {csv_path_stage1}")
    
    # Save Stage I model
    model_path_stage1 = BASE_DIR / 'models' / 'unet_baseline.pth'
    model_path_stage1.parent.mkdir(exist_ok=True)
    torch.save(model.state_dict(), model_path_stage1)
    print(f"Stage I model saved to: {model_path_stage1}")
    
    # ========================================================================
    # STAGE II: PDE-Constrained Fine-Tuning
    # ========================================================================
    if use_two_stage:
        print("\n" + "=" * 70)
        print("STAGE II: PDE-CONSTRAINED FINE-TUNING")
        print("=" * 70)
        print("Objective: L = L_Dice + L_BCE + λ_RD * L_RD + λ_PF * L_PF")
        print(f"  λ_RD (reaction-diffusion): {pde_weight}")
        print(f"  λ_PF (phase-field): {phase_field_weight}")
        print(f"  Diffusion coefficient (D): {diffusion_coeff}")
        print(f"  Reaction threshold (a): {reaction_threshold}")
        if phase_field_weight > 0:
            print(f"  Phase-field epsilon (ε): {epsilon}")
        
        criterion_stage2 = DiceBCEPDELoss(
            dice_weight=0.5,
            bce_weight=0.5,
            pde_weight=pde_weight,
            phase_field_weight=phase_field_weight,
            diffusion_coeff=diffusion_coeff,
            reaction_threshold=reaction_threshold,
            epsilon=epsilon
        )
        criterion_stage2 = criterion_stage2.to(device)
        
        # Use reduced learning rate for fine-tuning (10% of original)
        stage2_learning_rate = learning_rate * 0.1
        print(f"  Learning rate for Stage II: {stage2_learning_rate:.2e} (reduced from {learning_rate:.2e})")
        optimizer_stage2 = optim.AdamW(
            model.parameters(),
            lr=stage2_learning_rate,
            weight_decay=1e-5
        )
        
        early_stopping_stage2 = EarlyStopping(
            patience=early_stopping_patience,
            min_delta=1e-4,
            mode='max'
        )
        
        best_stage2_metrics, best_stage2_epoch, stage2_all_metrics = train_stage(
            model,
            train_loader,
            val_loader,
            criterion_stage2,
            optimizer_stage2,
            device,
            num_epochs=stage2_epochs,
            stage_name="Stage II",
            early_stopping=early_stopping_stage2,
            verbose=True,
            csv_path=csv_path_stage2
        )
        
        print(f"\nStage II complete. Best validation Dice: {best_stage2_metrics['val']['dice_score']:.6f} at epoch {best_stage2_epoch}")
        print(f"Stage II metrics saved to: {csv_path_stage2}")
        
        # Stability checks
        print("\nStability checks:")
        print(f"  Final PDE loss: {best_stage2_metrics['val']['pde_loss']:.6f}")
        print(f"  Final Dice loss: {best_stage2_metrics['val']['dice_loss']:.6f}")
        print(f"  Final BCE loss: {best_stage2_metrics['val']['bce_loss']:.6f}")
        
        # Check for improvement
        dice_improvement = best_stage2_metrics['val']['dice_score'] - best_stage1_metrics['val']['dice_score']
        print(f"\nPDE regularization effect:")
        print(f"  Dice score improvement: {dice_improvement:+.6f}")
        
        # Save Stage II model
        model_path_stage2 = BASE_DIR / 'models' / 'unet_pde_regularized.pth'
        torch.save(model.state_dict(), model_path_stage2)
        print(f"Stage II model saved to: {model_path_stage2}")
        
        # Generate plots
        print("\n" + "=" * 70)
        print("GENERATING TRAINING PLOTS")
        print("=" * 70)
        plot_training_results(
            csv_path_stage1=csv_path_stage1,
            csv_path_stage2=csv_path_stage2,
            output_dir=OUTPUT_DIR,
            show_plots=False
        )
    else:
        # Single-stage training with PDE from start
        print("\n" + "=" * 70)
        print("SINGLE-STAGE TRAINING (PDE from start)")
        print("=" * 70)
        print("Objective: L = L_Dice + L_BCE + λ_RD * L_RD + λ_PF * L_PF")
        print(f"  λ_RD (reaction-diffusion): {pde_weight}")
        print(f"  λ_PF (phase-field): {phase_field_weight}")
        print(f"  Diffusion coefficient (D): {diffusion_coeff}")
        print(f"  Reaction threshold (a): {reaction_threshold}")
        if phase_field_weight > 0:
            print(f"  Phase-field epsilon (ε): {epsilon}")
        
        criterion = DiceBCEPDELoss(
            dice_weight=0.5,
            bce_weight=0.5,
            pde_weight=pde_weight,
            phase_field_weight=phase_field_weight,
            diffusion_coeff=diffusion_coeff,
            reaction_threshold=reaction_threshold,
            epsilon=epsilon
        )
        criterion = criterion.to(device)
        optimizer = optim.AdamW(
            model.parameters(),
            lr=learning_rate,
            weight_decay=1e-5
        )
        
        early_stopping = EarlyStopping(
            patience=early_stopping_patience,
            min_delta=1e-4,
            mode='max'
        )
        
        csv_path_single = OUTPUT_DIR / f"metrics_single_stage_{timestamp}{fraction_str}.csv"
        
        best_metrics, best_epoch, single_stage_all_metrics = train_stage(
            model,
            train_loader,
            val_loader,
            criterion,
            optimizer,
            device,
            num_epochs=stage1_epochs,
            stage_name="Training",
            early_stopping=early_stopping,
            verbose=True,
            csv_path=csv_path_single
        )
        
        model_path = BASE_DIR / 'models' / 'unet_pde_regularized.pth'
        model_path.parent.mkdir(exist_ok=True)
        torch.save(model.state_dict(), model_path)
        print(f"Model saved to: {model_path}")
        print(f"Single-stage metrics saved to: {csv_path_single}")
        
        # Generate plots
        print("\n" + "=" * 70)
        print("GENERATING TRAINING PLOTS")
        print("=" * 70)
        plot_training_results(
            csv_path_stage1=csv_path_single,
            csv_path_stage2=None,
            output_dir=OUTPUT_DIR,
            show_plots=False
        )
    
    # ========================================================================
    # TEST SET EVALUATION
    # ========================================================================
    print("\n" + "=" * 70)
    print("TEST SET EVALUATION")
    print("=" * 70)
    
    # Evaluate on test set
    if TEST_JSON.exists() and TEST_DIR.exists():
        # Determine which model to evaluate (Stage II if two-stage, otherwise single-stage)
        if use_two_stage:
            model_name = "PDE-Constrained (Stage II)"
            test_metrics = evaluate_on_test_set(
                model,
                TEST_DIR,
                TEST_JSON,
                device,
                batch_size=batch_size,
                threshold=0.5,
                model_name=model_name
            )
            
            # Save test metrics
            test_metrics_path = OUTPUT_DIR / f"test_metrics_stage2_{timestamp}{fraction_str}"
            save_test_metrics(test_metrics, test_metrics_path, model_name=model_name)
            
            # Also evaluate Stage I model for comparison
            print("\n" + "=" * 70)
            print("EVALUATING STAGE I MODEL ON TEST SET")
            print("=" * 70)
            
            # Load Stage I model
            stage1_model = UNet(in_channels=1, out_channels=1, base_channels=64)
            stage1_model.load_state_dict(torch.load(model_path_stage1, map_location=device))
            stage1_model = stage1_model.to(device)
            
            stage1_test_metrics = evaluate_on_test_set(
                stage1_model,
                TEST_DIR,
                TEST_JSON,
                device,
                batch_size=batch_size,
                threshold=0.5,
                model_name="Baseline (Stage I)"
            )
            
            # Save Stage I test metrics
            stage1_test_metrics_path = OUTPUT_DIR / f"test_metrics_stage1_{timestamp}{fraction_str}"
            save_test_metrics(stage1_test_metrics, stage1_test_metrics_path, model_name="Baseline (Stage I)")
        else:
            model_name = "Single-Stage PDE-Constrained"
            test_metrics = evaluate_on_test_set(
                model,
                TEST_DIR,
                TEST_JSON,
                device,
                batch_size=batch_size,
                threshold=0.5,
                model_name=model_name
            )
            
            # Save test metrics
            test_metrics_path = OUTPUT_DIR / f"test_metrics_single_stage_{timestamp}{fraction_str}"
            save_test_metrics(test_metrics, test_metrics_path, model_name=model_name)
    else:
        print(f"Warning: Test set not found at {TEST_DIR} or {TEST_JSON}")
        print("Skipping test set evaluation.")
    
    print("\n" + "=" * 70)
    print("TRAINING COMPLETE")
    print("=" * 70)

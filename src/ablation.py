import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
import json
from datetime import datetime
from dataclasses import dataclass, asdict

from .unet import UNet
from .dataset import CellSegmentationDataset
from .loss import DiceBCELoss, DiceBCEPDELoss
from .train import train_stage, EarlyStopping, create_subset_dataset
from .evaluate import evaluate_on_test_set, compare_models_statistically
from .metrics import compute_dice_score_batch


@dataclass
class AblationConfig:
    """Configuration for an ablation study variant."""
    name: str
    description: str
    use_pde: bool = False
    pde_weight: float = 1e-4  # alpha_RD: Reaction-Diffusion weight
    phase_field_weight: float = 1e-4  # alpha_PF: Phase-field weight
    epsilon: float = 0.05  # Interface width for phase-field
    diffusion_coeff: float = 5.0  # D: Diffusion coefficient
    reaction_threshold: float = 0.5
    use_reaction_term: bool = True
    use_two_stage: bool = True
    use_three_stage: bool = False  # Three-stage: baseline -> PDE -> baseline
    train_fraction: Optional[float] = None
    # Optional override for Stage I epochs (baseline training).
    # If None, the global stage1_epochs argument is used.
    stage1_epochs: Optional[int] = None
    # Optional override for Stage II epochs (PDE fine-tuning).
    # If None, the global stage2_epochs argument is used.
    stage2_epochs: Optional[int] = None
    # Optional override for Stage III epochs (baseline continuation).
    # If None and use_three_stage=True, uses stage2_epochs.
    stage3_epochs: Optional[int] = None
    output_activation: str = 'sigmoid'  # Output activation function ('sigmoid' or 'tanh')
    intermediate_activation: str = 'relu'  # Intermediate activation function ('relu', 'leaky_relu', 'elu', 'gelu', 'swish', 'mish', 'prelu')
    seed: int = 42
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON serialization."""
        return asdict(self)


class PDERegularizationAblation:
    """
    Custom PDE regularization that can disable reaction term.
    This is used for ablation A3 (reaction vs diffusion only).
    """
    def __init__(
        self,
        diffusion_coeff: float = 1.0,
        reaction_threshold: float = 0.5,
        use_reaction_term: bool = True
    ):
        try:
            from .pde import PDERegularization
        except ImportError:
            from src.pde import PDERegularization
        
        self.pde_reg = PDERegularization(
            diffusion_coeff=diffusion_coeff,
            reaction_threshold=reaction_threshold
        )
        self.use_reaction_term = use_reaction_term
    
    def compute_loss(self, u: torch.Tensor) -> torch.Tensor:
        """Compute PDE loss with optional reaction term."""
        laplacian = self.pde_reg.compute_laplacian(u)
        
        if self.use_reaction_term:
            reaction = self.pde_reg.reaction_term(u)
            residual = self.pde_reg.diffusion_coeff * laplacian + reaction
        else:
            # Pure diffusion: f(u) = 0
            residual = self.pde_reg.diffusion_coeff * laplacian
        
        return torch.mean(residual ** 2)


def create_ablation_loss(
    config: AblationConfig
) -> torch.nn.Module:
    """
    Create loss function based on ablation configuration.
    
    Args:
        config: Ablation configuration
        
    Returns:
        Loss function module
    """
    if not config.use_pde:
        return DiceBCELoss(dice_weight=0.5, bce_weight=0.5)
    
    # For ablation A3 (reaction term ablation), we need custom PDE
    if not config.use_reaction_term:
        # Create custom loss with diffusion only
        class DiffusionOnlyLoss(torch.nn.Module):
            def __init__(self, config):
                super().__init__()
                self.dice_weight = 0.5
                self.bce_weight = 0.5
                self.pde_weight = config.pde_weight
                self.smooth = 1e-6
                self.pde_reg = PDERegularizationAblation(
                    diffusion_coeff=config.diffusion_coeff,
                    reaction_threshold=config.reaction_threshold,
                    use_reaction_term=False
                )
                self.bce = torch.nn.BCELoss()
            
            def forward(self, predictions, targets):
                # Dice loss
                predictions_flat = predictions.view(-1)
                targets_flat = targets.view(-1)
                intersection = (predictions_flat * targets_flat).sum()
                dice = (2.0 * intersection + self.smooth) / (
                    predictions_flat.sum() + targets_flat.sum() + self.smooth
                )
                dice_loss = 1 - dice
                
                # BCE loss
                bce_loss = self.bce(predictions, targets)
                
                # PDE loss (diffusion only)
                pde_loss = self.pde_reg.compute_loss(predictions)
                
                return (
                    self.dice_weight * dice_loss +
                    self.bce_weight * bce_loss +
                    self.pde_weight * pde_loss
                )
        
        return DiffusionOnlyLoss(config)
    else:
        # Standard PDE loss with reaction term and optional phase-field
        return DiceBCEPDELoss(
            dice_weight=0.5,
            bce_weight=0.5,
            pde_weight=config.pde_weight,
            phase_field_weight=config.phase_field_weight,
            diffusion_coeff=config.diffusion_coeff,
            reaction_threshold=config.reaction_threshold,
            epsilon=config.epsilon
        )


def run_ablation_variant(
    config: AblationConfig,
    train_dir: Path,
    train_json: Path,
    val_dir: Path,
    val_json: Path,
    in_dist_test_dir: Path,
    in_dist_test_json: Path,
    out_dist_test_dir: Path,
    out_dist_test_json: Path,
    device: torch.device,
    batch_size: int = 8,
    learning_rate: float = 1e-4,
    stage1_epochs: int = 50,
    stage2_epochs: int = 50,
    early_stopping_patience: int = 10,
    output_dir: Optional[Path] = None,
    ablation_folder: Optional[Path] = None
) -> Dict:
    """
    Run a single ablation variant and return evaluation metrics.
    
    Args:
        config: Ablation configuration
        train_dir: Training images directory
        train_json: Training annotations JSON
        val_dir: Validation images directory
        val_json: Validation annotations JSON
        in_dist_test_dir: In-distribution test images directory
        in_dist_test_json: In-distribution test annotations JSON
        out_dist_test_dir: Out-of-distribution test images directory
        out_dist_test_json: Out-of-distribution test annotations JSON
        device: Device to run on
        batch_size: Batch size for training
        learning_rate: Learning rate
        stage1_epochs: Max epochs for stage 1
        stage2_epochs: Max epochs for stage 2
        early_stopping_patience: Early stopping patience
        output_dir: Directory to save results
        
    Returns:
        Dictionary containing model path and evaluation metrics for both test sets
    """
    # Use ablation_folder if provided, otherwise use output_dir
    if ablation_folder is not None:
        variant_output_dir = ablation_folder
    elif output_dir is not None:
        variant_output_dir = Path(output_dir)
    else:
        variant_output_dir = Path(__file__).parent.parent / 'output' / 'ablation'
    variant_output_dir = Path(variant_output_dir)
    variant_output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n{'='*70}")
    print(f"ABLATION VARIANT: {config.name}")
    print(f"{'='*70}")
    print(f"Description: {config.description}")
    print(f"Configuration: {config.to_dict()}")
    
    # Set random seed
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(config.seed)
    
    # Create datasets
    train_dataset = CellSegmentationDataset(train_dir, train_json)
    val_dataset = CellSegmentationDataset(val_dir, val_json)
    
    # Apply low-label training if specified
    if config.train_fraction is not None:
        train_dataset = create_subset_dataset(train_dataset, config.train_fraction)
    
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
    
    # Create UNet model
    print(f"Using model architecture: UNet")
    model = UNet(
        in_channels=1, 
        out_channels=1, 
        base_channels=64,
        output_activation=config.output_activation,
        intermediate_activation=config.intermediate_activation
    )
    model = model.to(device)
    
    # Initialize variables for A2 stage comparison (two-stage with PDE)
    baseline_test_metrics = None
    pde_test_metrics = None
    comparison_results = None
    baseline_model_path = None
    pde_model_path = None
    
    # Stage I: Baseline training (if two-stage or three-stage)
    if (config.use_two_stage and config.use_pde) or config.use_three_stage:
        print(f"\nStage I: Baseline Training")
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
        
        # Determine number of epochs for Stage I:
        # - If config.stage1_epochs is set, use that override
        # - For three-stage, default to 50 if not specified
        # - Otherwise, fall back to global stage1_epochs
        if config.stage1_epochs is not None:
            stage1_epochs_to_use = config.stage1_epochs
        elif config.use_three_stage:
            stage1_epochs_to_use = 50  # Default for three-stage
        else:
            stage1_epochs_to_use = stage1_epochs
        
        # CSV path for Stage I metrics
        stage1_csv = variant_output_dir / f"{config.name.replace(' ', '_').lower()}_stage1_metrics.csv"
        
        _, _, stage1_all_metrics = train_stage(
            model,
            train_loader,
            val_loader,
            criterion_stage1,
            optimizer_stage1,
            device,
            num_epochs=stage1_epochs_to_use,
            stage_name="Stage I",
            early_stopping=early_stopping_stage1,
            verbose=False,
            csv_path=stage1_csv
        )
        
        # Save baseline model after Stage I (before PDE integration)
        # Save for both three-stage training and two-stage with PDE (e.g., A2)
        if config.use_three_stage or (config.use_two_stage and config.use_pde):
            baseline_model_path = variant_output_dir / f"{config.name.replace(' ', '_').lower()}_baseline_after_stage1.pth"
            torch.save(model.state_dict(), baseline_model_path)
            print(f"Saved baseline model (after Stage I) to: {baseline_model_path}")
            
            # Evaluate baseline model on test sets for A2 (two-stage with PDE) and A6 (three-stage)
            if (config.use_two_stage and config.use_pde and not config.use_three_stage) or config.use_three_stage:
                print(f"\nEvaluating baseline model (Stage I) on test sets...")
                
                try:
                    from .evaluate import evaluate_model
                except ImportError:
                    from src.evaluate import evaluate_model
                
                # Evaluate on in-distribution test set
                print(f"\n  In-distribution test set:")
                in_dist_test_dataset = CellSegmentationDataset(in_dist_test_dir, in_dist_test_json)
                in_dist_test_loader = DataLoader(
                    in_dist_test_dataset,
                    batch_size=batch_size,
                    shuffle=False,
                    num_workers=2,
                    pin_memory=True if torch.cuda.is_available() else False
                )
                baseline_in_dist_metrics = evaluate_model(
                    model,
                    in_dist_test_loader,
                    device,
                    threshold=0.5
                )
                
                # Evaluate on out-of-distribution test set
                print(f"\n  Out-of-distribution test set:")
                out_dist_test_dataset = CellSegmentationDataset(out_dist_test_dir, out_dist_test_json)
                out_dist_test_loader = DataLoader(
                    out_dist_test_dataset,
                    batch_size=batch_size,
                    shuffle=False,
                    num_workers=2,
                    pin_memory=True if torch.cuda.is_available() else False
                )
                baseline_out_dist_metrics = evaluate_model(
                    model,
                    out_dist_test_loader,
                    device,
                    threshold=0.5
                )
                
                # Combine metrics
                baseline_test_metrics = {
                    'in_dist': baseline_in_dist_metrics,
                    'out_dist': baseline_out_dist_metrics
                }
                print(f"Baseline model evaluation complete.")
    
    # Initialize variable for tracking actual Stage II epochs (needed for three-stage)
    actual_stage2_epochs = None
    
    # Stage II: PDE-constrained training (or single-stage if not two-stage)
    if config.use_pde or not config.use_two_stage or config.use_three_stage:
        stage_name = "Stage II (PDE)" if config.use_two_stage else "Training"
        print(f"\n{stage_name}: {'PDE-Constrained' if config.use_pde else 'Baseline'} Training")
        
        criterion = create_ablation_loss(config)
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
        
        # Determine number of epochs for Stage II:
        # - If two-stage and config.stage2_epochs is set, use that override
        # - Otherwise, fall back to global stage2_epochs (or stage1_epochs for single-stage)
        if config.use_two_stage:
            effective_stage2_epochs = (
                config.stage2_epochs if config.stage2_epochs is not None else stage2_epochs
            )
        else:
            effective_stage2_epochs = stage1_epochs
        
        # CSV path for Stage II metrics
        stage2_csv = variant_output_dir / f"{config.name.replace(' ', '_').lower()}_stage2_metrics.csv"
        
        _, _, stage2_all_metrics = train_stage(
            model,
            train_loader,
            val_loader,
            criterion,
            optimizer,
            device,
            num_epochs=effective_stage2_epochs,
            stage_name=stage_name,
            early_stopping=early_stopping,
            verbose=False,
            csv_path=stage2_csv
        )
        
        # Track actual epochs run in Stage II (for three-stage)
        actual_stage2_epochs = len(stage2_all_metrics) if stage2_all_metrics else effective_stage2_epochs
        if config.use_three_stage:
            print(f"Stage II completed: {actual_stage2_epochs} epochs (out of {effective_stage2_epochs} max)")
        
        # For three-stage training, save and evaluate Stage II (PDE) model
        if config.use_three_stage and config.use_pde:
            pde_model_path = variant_output_dir / f"{config.name.replace(' ', '_').lower()}_after_pde_stage2.pth"
            torch.save(model.state_dict(), pde_model_path)
            print(f"Saved PDE model (after Stage II) to: {pde_model_path}")
            
            # Evaluate PDE model on both test sets for three-stage comparison
            print(f"\nEvaluating PDE model (Stage II) on test sets...")
            
            try:
                from .evaluate import evaluate_model
            except ImportError:
                from src.evaluate import evaluate_model
            
            # Evaluate on in-distribution test set
            print(f"\n  In-distribution test set:")
            in_dist_test_dataset = CellSegmentationDataset(in_dist_test_dir, in_dist_test_json)
            in_dist_test_loader = DataLoader(
                in_dist_test_dataset,
                batch_size=batch_size,
                shuffle=False,
                num_workers=2,
                pin_memory=True if torch.cuda.is_available() else False
            )
            pde_in_dist_metrics = evaluate_model(
                model,
                in_dist_test_loader,
                device,
                threshold=0.5
            )
            
            # Evaluate on out-of-distribution test set
            print(f"\n  Out-of-distribution test set:")
            out_dist_test_dataset = CellSegmentationDataset(out_dist_test_dir, out_dist_test_json)
            out_dist_test_loader = DataLoader(
                out_dist_test_dataset,
                batch_size=batch_size,
                shuffle=False,
                num_workers=2,
                pin_memory=True if torch.cuda.is_available() else False
            )
            pde_out_dist_metrics = evaluate_model(
                model,
                out_dist_test_loader,
                device,
                threshold=0.5
            )
            
            # Store for later comparison with Stage III
            pde_test_metrics = {
                'in_dist': pde_in_dist_metrics,
                'out_dist': pde_out_dist_metrics
            }
            print(f"PDE model (Stage II) evaluation complete.")
        
        # Save model after PDE integration (after Stage II)
        # Save for two-stage with PDE (e.g., A2) - this is the final model with PDE
        if config.use_two_stage and config.use_pde and not config.use_three_stage:
            pde_model_path = variant_output_dir / f"{config.name.replace(' ', '_').lower()}_after_pde_stage2.pth"
            torch.save(model.state_dict(), pde_model_path)
            print(f"Saved model after PDE integration (after Stage II) to: {pde_model_path}")
            
            # Evaluate PDE model on both test sets for A2
            print(f"\nEvaluating PDE model (Stage II) on test sets...")
            
            try:
                from .evaluate import evaluate_model, compare_models_statistically
            except ImportError:
                from src.evaluate import evaluate_model, compare_models_statistically
            
            # Evaluate on in-distribution test set
            print(f"\n  In-distribution test set:")
            in_dist_test_dataset = CellSegmentationDataset(in_dist_test_dir, in_dist_test_json)
            in_dist_test_loader = DataLoader(
                in_dist_test_dataset,
                batch_size=batch_size,
                shuffle=False,
                num_workers=2,
                pin_memory=True if torch.cuda.is_available() else False
            )
            pde_in_dist_metrics = evaluate_model(
                model,
                in_dist_test_loader,
                device,
                threshold=0.5
            )
            
            # Evaluate on out-of-distribution test set
            print(f"\n  Out-of-distribution test set:")
            out_dist_test_dataset = CellSegmentationDataset(out_dist_test_dir, out_dist_test_json)
            out_dist_test_loader = DataLoader(
                out_dist_test_dataset,
                batch_size=batch_size,
                shuffle=False,
                num_workers=2,
                pin_memory=True if torch.cuda.is_available() else False
            )
            pde_out_dist_metrics = evaluate_model(
                model,
                out_dist_test_loader,
                device,
                threshold=0.5
            )
            
            # Combine metrics
            pde_test_metrics = {
                'in_dist': pde_in_dist_metrics,
                'out_dist': pde_out_dist_metrics
            }
            print(f"PDE model evaluation complete.")
            
            # Compare Stage 1 vs Stage 2 models for both test sets
            comparison_results = {}
            
            # In-distribution comparison
            print(f"\n{'='*70}")
            print(f"COMPARING STAGE 1 vs STAGE 2 FOR {config.name} - IN-DISTRIBUTION")
            print(f"{'='*70}")
            
            in_dist_comparison = compare_models_statistically(
                baseline_test_metrics['in_dist'],
                pde_test_metrics['in_dist'],
                alpha=0.05
            )
            comparison_results['in_dist'] = in_dist_comparison
            
            # Print in-distribution comparison summary
            print("\nStatistical Comparison Results - In-Distribution (α = 0.05):")
            print("-" * 70)
            for metric_name, results in in_dist_comparison.items():
                metric_display = metric_name.replace('_', ' ').title()
                print(f"\n{metric_display}:")
                print(f"  Stage 1 (Baseline) Mean: {results['baseline_mean']:.4f} ± {results['baseline_std']:.4f}")
                print(f"  Stage 2 (PDE) Mean:     {results['pde_mean']:.4f} ± {results['pde_std']:.4f}")
                print(f"  Improvement:            {results['improvement']:+.4f}")
                print(f"  Paired t-test p-value:  {results['t_pvalue']:.4f}")
                print(f"  Wilcoxon p-value:       {results['wilcoxon_pvalue']:.4f}")
                print(f"  Statistically Significant: {'Yes' if results['significant'] else 'No'}")
            
            # Out-of-distribution comparison
            print(f"\n{'='*70}")
            print(f"COMPARING STAGE 1 vs STAGE 2 FOR {config.name} - OUT-OF-DISTRIBUTION")
            print(f"{'='*70}")
            
            out_dist_comparison = compare_models_statistically(
                baseline_test_metrics['out_dist'],
                pde_test_metrics['out_dist'],
                alpha=0.05
            )
            comparison_results['out_dist'] = out_dist_comparison
            
            # Print out-of-distribution comparison summary
            print("\nStatistical Comparison Results - Out-of-Distribution (α = 0.05):")
            print("-" * 70)
            for metric_name, results in out_dist_comparison.items():
                metric_display = metric_name.replace('_', ' ').title()
                print(f"\n{metric_display}:")
                print(f"  Stage 1 (Baseline) Mean: {results['baseline_mean']:.4f} ± {results['baseline_std']:.4f}")
                print(f"  Stage 2 (PDE) Mean:     {results['pde_mean']:.4f} ± {results['pde_std']:.4f}")
                print(f"  Improvement:            {results['improvement']:+.4f}")
                print(f"  Paired t-test p-value:  {results['t_pvalue']:.4f}")
                print(f"  Wilcoxon p-value:       {results['wilcoxon_pvalue']:.4f}")
                print(f"  Statistically Significant: {'Yes' if results['significant'] else 'No'}")
            
            # Save comparison results to CSV for both test sets
            # In-distribution comparison CSV
            in_dist_comparison_csv = variant_output_dir / f"{config.name.replace(' ', '_').lower()}_stage1_vs_stage2_comparison_in_dist.csv"
            in_dist_comparison_data = []
            for metric_name, results in in_dist_comparison.items():
                in_dist_comparison_data.append({
                    'metric': metric_name,
                    'stage1_mean': results['baseline_mean'],
                    'stage1_std': results['baseline_std'],
                    'stage2_mean': results['pde_mean'],
                    'stage2_std': results['pde_std'],
                    'improvement': results['improvement'],
                    't_pvalue': results['t_pvalue'],
                    'wilcoxon_pvalue': results['wilcoxon_pvalue'],
                    'significant': results['significant']
                })
            in_dist_comparison_df = pd.DataFrame(in_dist_comparison_data)
            in_dist_comparison_df.to_csv(in_dist_comparison_csv, index=False)
            print(f"\nIn-distribution comparison results saved to: {in_dist_comparison_csv}")
            
            # Out-of-distribution comparison CSV
            out_dist_comparison_csv = variant_output_dir / f"{config.name.replace(' ', '_').lower()}_stage1_vs_stage2_comparison_out_dist.csv"
            out_dist_comparison_data = []
            for metric_name, results in out_dist_comparison.items():
                out_dist_comparison_data.append({
                    'metric': metric_name,
                    'stage1_mean': results['baseline_mean'],
                    'stage1_std': results['baseline_std'],
                    'stage2_mean': results['pde_mean'],
                    'stage2_std': results['pde_std'],
                    'improvement': results['improvement'],
                    't_pvalue': results['t_pvalue'],
                    'wilcoxon_pvalue': results['wilcoxon_pvalue'],
                    'significant': results['significant']
                })
            out_dist_comparison_df = pd.DataFrame(out_dist_comparison_data)
            out_dist_comparison_df.to_csv(out_dist_comparison_csv, index=False)
            print(f"Out-of-distribution comparison results saved to: {out_dist_comparison_csv}")
    
    # Stage III: Baseline continuation (if three-stage training)
    # IMPORTANT: Load the baseline model from AFTER Stage I (before PDE)
    # and continue training for the same number of epochs that Stage II ran
    if config.use_three_stage:
        print(f"\nStage III: Baseline Continuation Training")
        print(f"Loading baseline model from after Stage I (before PDE optimization)")
        
        # Load the saved baseline model (from after Stage I, before PDE)
        baseline_model_path = variant_output_dir / f"{config.name.replace(' ', '_').lower()}_baseline_after_stage1.pth"
        if not baseline_model_path.exists():
            raise FileNotFoundError(f"Baseline model not found: {baseline_model_path}")
        
        model.load_state_dict(torch.load(baseline_model_path))
        print(f"Loaded baseline model from: {baseline_model_path}")
        
        criterion_stage3 = DiceBCELoss(dice_weight=0.5, bce_weight=0.5)
        criterion_stage3 = criterion_stage3.to(device)
        optimizer_stage3 = optim.AdamW(
            model.parameters(),
            lr=learning_rate,
            weight_decay=1e-5
        )
        
        # Use the actual number of epochs that Stage II ran
        if actual_stage2_epochs is None:
            raise ValueError("actual_stage2_epochs not set - Stage II must run before Stage III")
        effective_stage3_epochs = actual_stage2_epochs
        print(f"Stage III will train for {effective_stage3_epochs} epochs (same as Stage II)")
        
        # CSV path for Stage III metrics
        stage3_csv = variant_output_dir / f"{config.name.replace(' ', '_').lower()}_stage3_metrics.csv"
        
        # No early stopping for Stage III (train for full n epochs)
        train_stage(
            model,
            train_loader,
            val_loader,
            criterion_stage3,
            optimizer_stage3,
            device,
            num_epochs=effective_stage3_epochs,
            stage_name="Stage III (Baseline)",
            early_stopping=None,  # No early stopping in Stage III
            verbose=False,
            csv_path=stage3_csv
        )
        
        # Evaluate Stage III (baseline continuation) model on both test sets
        print(f"\nEvaluating Stage III (baseline continuation) model on test sets...")
        
        try:
            from .evaluate import evaluate_model, compare_models_statistically
        except ImportError:
            from src.evaluate import evaluate_model, compare_models_statistically
        
        # Evaluate on in-distribution test set
        print(f"\n  In-distribution test set:")
        in_dist_test_dataset = CellSegmentationDataset(in_dist_test_dir, in_dist_test_json)
        in_dist_test_loader = DataLoader(
            in_dist_test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=2,
            pin_memory=True if torch.cuda.is_available() else False
        )
        stage3_in_dist_metrics = evaluate_model(
            model,
            in_dist_test_loader,
            device,
            threshold=0.5
        )
        
        # Evaluate on out-of-distribution test set
        print(f"\n  Out-of-distribution test set:")
        out_dist_test_dataset = CellSegmentationDataset(out_dist_test_dir, out_dist_test_json)
        out_dist_test_loader = DataLoader(
            out_dist_test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=2,
            pin_memory=True if torch.cuda.is_available() else False
        )
        stage3_out_dist_metrics = evaluate_model(
            model,
            out_dist_test_loader,
            device,
            threshold=0.5
        )
        
        # Compare Stage II (PDE) vs Stage I (baseline) for both test sets
        print(f"\n{'='*70}")
        print(f"COMPARING STAGE II (PDE) vs STAGE I (BASELINE) FOR {config.name} - IN-DISTRIBUTION")
        print(f"{'='*70}")
        
        stage2_vs_stage1_in_dist = compare_models_statistically(
            baseline_test_metrics['in_dist'],
            pde_test_metrics['in_dist'],
            alpha=0.05
        )
        
        print("\nStatistical Comparison Results - In-Distribution (α = 0.05):")
        print("-" * 70)
        for metric_name, results in stage2_vs_stage1_in_dist.items():
            metric_display = metric_name.replace('_', ' ').title()
            baseline_mean = results['baseline_mean']
            improvement_abs = results['improvement']
            improvement_pct = (improvement_abs / baseline_mean) * 100 if baseline_mean > 0 else 0.0
            print(f"\n{metric_display}:")
            print(f"  Stage I (Baseline) Mean:   {results['baseline_mean']:.4f} ± {results['baseline_std']:.4f}")
            print(f"  Stage II (PDE) Mean:        {results['pde_mean']:.4f} ± {results['pde_std']:.4f}")
            print(f"  Absolute Improvement:      {results['improvement']:+.4f}")
            print(f"  Percentage Improvement:     {improvement_pct:+.2f}%")
            print(f"  Paired t-test p-value:      {results['t_pvalue']:.4f}")
            print(f"  Wilcoxon p-value:           {results['wilcoxon_pvalue']:.4f}")
            print(f"  Statistically Significant:  {'Yes' if results['significant'] else 'No'}")
        
        print(f"\n{'='*70}")
        print(f"COMPARING STAGE II (PDE) vs STAGE I (BASELINE) FOR {config.name} - OUT-OF-DISTRIBUTION")
        print(f"{'='*70}")
        
        stage2_vs_stage1_out_dist = compare_models_statistically(
            baseline_test_metrics['out_dist'],
            pde_test_metrics['out_dist'],
            alpha=0.05
        )
        
        print("\nStatistical Comparison Results - Out-of-Distribution (α = 0.05):")
        print("-" * 70)
        for metric_name, results in stage2_vs_stage1_out_dist.items():
            metric_display = metric_name.replace('_', ' ').title()
            baseline_mean = results['baseline_mean']
            improvement_abs = results['improvement']
            # For Hausdorff, improvement is negative (reduction), so we need to handle it
            if metric_name == 'hausdorff_distances':
                improvement_pct = (-improvement_abs / baseline_mean) * 100 if baseline_mean > 0 else 0.0
            else:
                improvement_pct = (improvement_abs / baseline_mean) * 100 if baseline_mean > 0 else 0.0
            print(f"\n{metric_display}:")
            print(f"  Stage I (Baseline) Mean:   {results['baseline_mean']:.4f} ± {results['baseline_std']:.4f}")
            print(f"  Stage II (PDE) Mean:        {results['pde_mean']:.4f} ± {results['pde_std']:.4f}")
            print(f"  Absolute Improvement:      {results['improvement']:+.4f}")
            print(f"  Percentage Improvement:     {improvement_pct:+.2f}%")
            print(f"  Paired t-test p-value:      {results['t_pvalue']:.4f}")
            print(f"  Wilcoxon p-value:           {results['wilcoxon_pvalue']:.4f}")
            print(f"  Statistically Significant:  {'Yes' if results['significant'] else 'No'}")
        
        # Compare Stage III (baseline continuation) vs Stage I (baseline) for both test sets
        print(f"\n{'='*70}")
        print(f"COMPARING STAGE III (BASELINE CONTINUATION) vs STAGE I (BASELINE) FOR {config.name} - IN-DISTRIBUTION")
        print(f"{'='*70}")
        
        stage3_vs_stage1_in_dist = compare_models_statistically(
            baseline_test_metrics['in_dist'],
            stage3_in_dist_metrics,
            alpha=0.05
        )
        
        print("\nStatistical Comparison Results - In-Distribution (α = 0.05):")
        print("-" * 70)
        for metric_name, results in stage3_vs_stage1_in_dist.items():
            metric_display = metric_name.replace('_', ' ').title()
            baseline_mean = results['baseline_mean']
            improvement_abs = results['improvement']
            improvement_pct = (improvement_abs / baseline_mean) * 100 if baseline_mean > 0 else 0.0
            print(f"\n{metric_display}:")
            print(f"  Stage I (Baseline) Mean:        {results['baseline_mean']:.4f} ± {results['baseline_std']:.4f}")
            print(f"  Stage III (Baseline Cont.) Mean: {results['pde_mean']:.4f} ± {results['pde_std']:.4f}")
            print(f"  Absolute Improvement:           {results['improvement']:+.4f}")
            print(f"  Percentage Improvement:          {improvement_pct:+.2f}%")
            print(f"  Paired t-test p-value:          {results['t_pvalue']:.4f}")
            print(f"  Wilcoxon p-value:                {results['wilcoxon_pvalue']:.4f}")
            print(f"  Statistically Significant:      {'Yes' if results['significant'] else 'No'}")
        
        print(f"\n{'='*70}")
        print(f"COMPARING STAGE III (BASELINE CONTINUATION) vs STAGE I (BASELINE) FOR {config.name} - OUT-OF-DISTRIBUTION")
        print(f"{'='*70}")
        
        stage3_vs_stage1_out_dist = compare_models_statistically(
            baseline_test_metrics['out_dist'],
            stage3_out_dist_metrics,
            alpha=0.05
        )
        
        print("\nStatistical Comparison Results - Out-of-Distribution (α = 0.05):")
        print("-" * 70)
        for metric_name, results in stage3_vs_stage1_out_dist.items():
            metric_display = metric_name.replace('_', ' ').title()
            baseline_mean = results['baseline_mean']
            improvement_abs = results['improvement']
            # For Hausdorff, improvement is negative (reduction), so we need to handle it
            if metric_name == 'hausdorff_distances':
                improvement_pct = (-improvement_abs / baseline_mean) * 100 if baseline_mean > 0 else 0.0
            else:
                improvement_pct = (improvement_abs / baseline_mean) * 100 if baseline_mean > 0 else 0.0
            print(f"\n{metric_display}:")
            print(f"  Stage I (Baseline) Mean:        {results['baseline_mean']:.4f} ± {results['baseline_std']:.4f}")
            print(f"  Stage III (Baseline Cont.) Mean: {results['pde_mean']:.4f} ± {results['pde_std']:.4f}")
            print(f"  Absolute Improvement:           {results['improvement']:+.4f}")
            print(f"  Percentage Improvement:          {improvement_pct:+.2f}%")
            print(f"  Paired t-test p-value:           {results['t_pvalue']:.4f}")
            print(f"  Wilcoxon p-value:                {results['wilcoxon_pvalue']:.4f}")
            print(f"  Statistically Significant:       {'Yes' if results['significant'] else 'No'}")
        
        # Compare Stage II (PDE) vs Stage III (baseline continuation) for both test sets
        comparison_results = {}
        
        # In-distribution comparison
        print(f"\n{'='*70}")
        print(f"COMPARING STAGE II (PDE) vs STAGE III (BASELINE) FOR {config.name} - IN-DISTRIBUTION")
        print(f"{'='*70}")
        
        in_dist_comparison = compare_models_statistically(
            pde_test_metrics['in_dist'],
            stage3_in_dist_metrics,
            alpha=0.05
        )
        comparison_results['in_dist'] = in_dist_comparison
        
        # Print in-distribution comparison summary
        print("\nStatistical Comparison Results - In-Distribution (α = 0.05):")
        print("-" * 70)
        for metric_name, results in in_dist_comparison.items():
            metric_display = metric_name.replace('_', ' ').title()
            print(f"\n{metric_display}:")
            print(f"  Stage II (PDE) Mean:        {results['baseline_mean']:.4f} ± {results['baseline_std']:.4f}")
            print(f"  Stage III (Baseline) Mean: {results['pde_mean']:.4f} ± {results['pde_std']:.4f}")
            print(f"  Improvement:                {results['improvement']:+.4f}")
            print(f"  Paired t-test p-value:     {results['t_pvalue']:.4f}")
            print(f"  Wilcoxon p-value:          {results['wilcoxon_pvalue']:.4f}")
            print(f"  Statistically Significant: {'Yes' if results['significant'] else 'No'}")
        
        # Out-of-distribution comparison
        print(f"\n{'='*70}")
        print(f"COMPARING STAGE II (PDE) vs STAGE III (BASELINE) FOR {config.name} - OUT-OF-DISTRIBUTION")
        print(f"{'='*70}")
        
        out_dist_comparison = compare_models_statistically(
            pde_test_metrics['out_dist'],
            stage3_out_dist_metrics,
            alpha=0.05
        )
        comparison_results['out_dist'] = out_dist_comparison
        
        # Print out-of-distribution comparison summary
        print("\nStatistical Comparison Results - Out-of-Distribution (α = 0.05):")
        print("-" * 70)
        for metric_name, results in out_dist_comparison.items():
            metric_display = metric_name.replace('_', ' ').title()
            print(f"\n{metric_display}:")
            print(f"  Stage II (PDE) Mean:        {results['baseline_mean']:.4f} ± {results['baseline_std']:.4f}")
            print(f"  Stage III (Baseline) Mean: {results['pde_mean']:.4f} ± {results['pde_std']:.4f}")
            print(f"  Improvement:                {results['improvement']:+.4f}")
            print(f"  Paired t-test p-value:     {results['t_pvalue']:.4f}")
            print(f"  Wilcoxon p-value:          {results['wilcoxon_pvalue']:.4f}")
            print(f"  Statistically Significant: {'Yes' if results['significant'] else 'No'}")
        
        # Save Stage II vs Stage I comparison results to CSV
        # In-distribution
        stage2_vs_stage1_in_dist_csv = variant_output_dir / f"{config.name.replace(' ', '_').lower()}_stage1_vs_stage2_comparison_in_dist.csv"
        stage2_vs_stage1_in_dist_data = []
        for metric_name, results in stage2_vs_stage1_in_dist.items():
            baseline_mean = results['baseline_mean']
            improvement_abs = results['improvement']
            improvement_pct = (improvement_abs / baseline_mean) * 100 if baseline_mean > 0 else 0.0
            stage2_vs_stage1_in_dist_data.append({
                'metric': metric_name,
                'stage1_mean': results['baseline_mean'],
                'stage1_std': results['baseline_std'],
                'stage2_mean': results['pde_mean'],
                'stage2_std': results['pde_std'],
                'improvement': results['improvement'],
                'improvement_pct': improvement_pct,
                't_pvalue': results['t_pvalue'],
                'wilcoxon_pvalue': results['wilcoxon_pvalue'],
                'significant': results['significant']
            })
        stage2_vs_stage1_in_dist_df = pd.DataFrame(stage2_vs_stage1_in_dist_data)
        stage2_vs_stage1_in_dist_df.to_csv(stage2_vs_stage1_in_dist_csv, index=False)
        print(f"\nStage II vs Stage I comparison (in-dist) saved to: {stage2_vs_stage1_in_dist_csv}")
        
        # Out-of-distribution
        stage2_vs_stage1_out_dist_csv = variant_output_dir / f"{config.name.replace(' ', '_').lower()}_stage1_vs_stage2_comparison_out_dist.csv"
        stage2_vs_stage1_out_dist_data = []
        for metric_name, results in stage2_vs_stage1_out_dist.items():
            baseline_mean = results['baseline_mean']
            improvement_abs = results['improvement']
            if metric_name == 'hausdorff_distances':
                improvement_pct = (-improvement_abs / baseline_mean) * 100 if baseline_mean > 0 else 0.0
            else:
                improvement_pct = (improvement_abs / baseline_mean) * 100 if baseline_mean > 0 else 0.0
            stage2_vs_stage1_out_dist_data.append({
                'metric': metric_name,
                'stage1_mean': results['baseline_mean'],
                'stage1_std': results['baseline_std'],
                'stage2_mean': results['pde_mean'],
                'stage2_std': results['pde_std'],
                'improvement': results['improvement'],
                'improvement_pct': improvement_pct,
                't_pvalue': results['t_pvalue'],
                'wilcoxon_pvalue': results['wilcoxon_pvalue'],
                'significant': results['significant']
            })
        stage2_vs_stage1_out_dist_df = pd.DataFrame(stage2_vs_stage1_out_dist_data)
        stage2_vs_stage1_out_dist_df.to_csv(stage2_vs_stage1_out_dist_csv, index=False)
        print(f"Stage II vs Stage I comparison (out-dist) saved to: {stage2_vs_stage1_out_dist_csv}")
        
        # Save Stage III vs Stage I comparison results to CSV
        # In-distribution
        stage3_vs_stage1_in_dist_csv = variant_output_dir / f"{config.name.replace(' ', '_').lower()}_stage1_vs_stage3_comparison_in_dist.csv"
        stage3_vs_stage1_in_dist_data = []
        for metric_name, results in stage3_vs_stage1_in_dist.items():
            baseline_mean = results['baseline_mean']
            improvement_abs = results['improvement']
            improvement_pct = (improvement_abs / baseline_mean) * 100 if baseline_mean > 0 else 0.0
            stage3_vs_stage1_in_dist_data.append({
                'metric': metric_name,
                'stage1_mean': results['baseline_mean'],
                'stage1_std': results['baseline_std'],
                'stage3_mean': results['pde_mean'],
                'stage3_std': results['pde_std'],
                'improvement': results['improvement'],
                'improvement_pct': improvement_pct,
                't_pvalue': results['t_pvalue'],
                'wilcoxon_pvalue': results['wilcoxon_pvalue'],
                'significant': results['significant']
            })
        stage3_vs_stage1_in_dist_df = pd.DataFrame(stage3_vs_stage1_in_dist_data)
        stage3_vs_stage1_in_dist_df.to_csv(stage3_vs_stage1_in_dist_csv, index=False)
        print(f"Stage III vs Stage I comparison (in-dist) saved to: {stage3_vs_stage1_in_dist_csv}")
        
        # Out-of-distribution
        stage3_vs_stage1_out_dist_csv = variant_output_dir / f"{config.name.replace(' ', '_').lower()}_stage1_vs_stage3_comparison_out_dist.csv"
        stage3_vs_stage1_out_dist_data = []
        for metric_name, results in stage3_vs_stage1_out_dist.items():
            baseline_mean = results['baseline_mean']
            improvement_abs = results['improvement']
            if metric_name == 'hausdorff_distances':
                improvement_pct = (-improvement_abs / baseline_mean) * 100 if baseline_mean > 0 else 0.0
            else:
                improvement_pct = (improvement_abs / baseline_mean) * 100 if baseline_mean > 0 else 0.0
            stage3_vs_stage1_out_dist_data.append({
                'metric': metric_name,
                'stage1_mean': results['baseline_mean'],
                'stage1_std': results['baseline_std'],
                'stage3_mean': results['pde_mean'],
                'stage3_std': results['pde_std'],
                'improvement': results['improvement'],
                'improvement_pct': improvement_pct,
                't_pvalue': results['t_pvalue'],
                'wilcoxon_pvalue': results['wilcoxon_pvalue'],
                'significant': results['significant']
            })
        stage3_vs_stage1_out_dist_df = pd.DataFrame(stage3_vs_stage1_out_dist_data)
        stage3_vs_stage1_out_dist_df.to_csv(stage3_vs_stage1_out_dist_csv, index=False)
        print(f"Stage III vs Stage I comparison (out-dist) saved to: {stage3_vs_stage1_out_dist_csv}")
        
        # Save comparison results to CSV for both test sets
        # In-distribution comparison CSV
        in_dist_comparison_csv = variant_output_dir / f"{config.name.replace(' ', '_').lower()}_stage2_vs_stage3_comparison_in_dist.csv"
        in_dist_comparison_data = []
        for metric_name, results in in_dist_comparison.items():
            in_dist_comparison_data.append({
                'metric': metric_name,
                'stage2_mean': results['baseline_mean'],
                'stage2_std': results['baseline_std'],
                'stage3_mean': results['pde_mean'],
                'stage3_std': results['pde_std'],
                'improvement': results['improvement'],
                't_pvalue': results['t_pvalue'],
                'wilcoxon_pvalue': results['wilcoxon_pvalue'],
                'significant': results['significant']
            })
        in_dist_comparison_df = pd.DataFrame(in_dist_comparison_data)
        in_dist_comparison_df.to_csv(in_dist_comparison_csv, index=False)
        print(f"\nIn-distribution comparison results saved to: {in_dist_comparison_csv}")
        
        # Out-of-distribution comparison CSV
        out_dist_comparison_csv = variant_output_dir / f"{config.name.replace(' ', '_').lower()}_stage2_vs_stage3_comparison_out_dist.csv"
        out_dist_comparison_data = []
        for metric_name, results in out_dist_comparison.items():
            out_dist_comparison_data.append({
                'metric': metric_name,
                'stage2_mean': results['baseline_mean'],
                'stage2_std': results['baseline_std'],
                'stage3_mean': results['pde_mean'],
                'stage3_std': results['pde_std'],
                'improvement': results['improvement'],
                't_pvalue': results['t_pvalue'],
                'wilcoxon_pvalue': results['wilcoxon_pvalue'],
                'significant': results['significant']
            })
        out_dist_comparison_df = pd.DataFrame(out_dist_comparison_data)
        out_dist_comparison_df.to_csv(out_dist_comparison_csv, index=False)
        print(f"Out-of-distribution comparison results saved to: {out_dist_comparison_csv}")
        
        # Save final model for three-stage (Stage III model)
        model_path = variant_output_dir / f"{config.name.replace(' ', '_').lower()}_after_stage3.pth"
        torch.save(model.state_dict(), model_path)
        print(f"Saved final model (after Stage III) to: {model_path}")
        
        # Return results for three-stage training
        # pde_model_path should be defined from Stage II evaluation above
        pde_path = pde_model_path if 'pde_model_path' in locals() else None
        return {
            'config': config.to_dict(),
            'model_path': str(model_path),
            'pde_model_path': str(pde_path) if pde_path else None,
            'baseline_model_path': str(baseline_model_path) if baseline_model_path else None,
            'baseline_in_dist_metrics': {k: v.tolist() if isinstance(v, np.ndarray) else v 
                                        for k, v in baseline_test_metrics['in_dist'].items()},
            'baseline_out_dist_metrics': {k: v.tolist() if isinstance(v, np.ndarray) else v 
                                         for k, v in baseline_test_metrics['out_dist'].items()},
            'pde_in_dist_metrics': {k: v.tolist() if isinstance(v, np.ndarray) else v 
                                   for k, v in pde_test_metrics['in_dist'].items()},
            'pde_out_dist_metrics': {k: v.tolist() if isinstance(v, np.ndarray) else v 
                                    for k, v in pde_test_metrics['out_dist'].items()},
            'stage3_in_dist_metrics': {k: v.tolist() if isinstance(v, np.ndarray) else v 
                                      for k, v in stage3_in_dist_metrics.items()},
            'stage3_out_dist_metrics': {k: v.tolist() if isinstance(v, np.ndarray) else v 
                                       for k, v in stage3_out_dist_metrics.items()},
            'metrics': {k: v.tolist() if isinstance(v, np.ndarray) else v 
                       for k, v in stage3_in_dist_metrics.items()},  # Default to in-dist Stage III metrics
            'in_dist_metrics': {k: v.tolist() if isinstance(v, np.ndarray) else v 
                               for k, v in stage3_in_dist_metrics.items()},
            'out_dist_metrics': {k: v.tolist() if isinstance(v, np.ndarray) else v 
                                for k, v in stage3_out_dist_metrics.items()},
            'stage_comparison': {
                'stage1_vs_stage2': {
                    'in_dist': {k: {
                        'stage1_mean': float(v['baseline_mean']),
                        'stage1_std': float(v['baseline_std']),
                        'stage2_mean': float(v['pde_mean']),
                        'stage2_std': float(v['pde_std']),
                        'improvement': float(v['improvement']),
                        'improvement_pct': float((v['improvement'] / v['baseline_mean']) * 100) if v['baseline_mean'] > 0 else 0.0,
                        't_pvalue': float(v['t_pvalue']),
                        'wilcoxon_pvalue': float(v['wilcoxon_pvalue']),
                        'significant': bool(v['significant'])
                    } for k, v in stage2_vs_stage1_in_dist.items()},
                    'out_dist': {k: {
                        'stage1_mean': float(v['baseline_mean']),
                        'stage1_std': float(v['baseline_std']),
                        'stage2_mean': float(v['pde_mean']),
                        'stage2_std': float(v['pde_std']),
                        'improvement': float(v['improvement']),
                        'improvement_pct': float((-v['improvement'] / v['baseline_mean']) * 100) if k == 'hausdorff_distances' and v['baseline_mean'] > 0 else float((v['improvement'] / v['baseline_mean']) * 100) if v['baseline_mean'] > 0 else 0.0,
                        't_pvalue': float(v['t_pvalue']),
                        'wilcoxon_pvalue': float(v['wilcoxon_pvalue']),
                        'significant': bool(v['significant'])
                    } for k, v in stage2_vs_stage1_out_dist.items()}
                },
                'stage1_vs_stage3': {
                    'in_dist': {k: {
                        'stage1_mean': float(v['baseline_mean']),
                        'stage1_std': float(v['baseline_std']),
                        'stage3_mean': float(v['pde_mean']),
                        'stage3_std': float(v['pde_std']),
                        'improvement': float(v['improvement']),
                        'improvement_pct': float((v['improvement'] / v['baseline_mean']) * 100) if v['baseline_mean'] > 0 else 0.0,
                        't_pvalue': float(v['t_pvalue']),
                        'wilcoxon_pvalue': float(v['wilcoxon_pvalue']),
                        'significant': bool(v['significant'])
                    } for k, v in stage3_vs_stage1_in_dist.items()},
                    'out_dist': {k: {
                        'stage1_mean': float(v['baseline_mean']),
                        'stage1_std': float(v['baseline_std']),
                        'stage3_mean': float(v['pde_mean']),
                        'stage3_std': float(v['pde_std']),
                        'improvement': float(v['improvement']),
                        'improvement_pct': float((-v['improvement'] / v['baseline_mean']) * 100) if k == 'hausdorff_distances' and v['baseline_mean'] > 0 else float((v['improvement'] / v['baseline_mean']) * 100) if v['baseline_mean'] > 0 else 0.0,
                        't_pvalue': float(v['t_pvalue']),
                        'wilcoxon_pvalue': float(v['wilcoxon_pvalue']),
                        'significant': bool(v['significant'])
                    } for k, v in stage3_vs_stage1_out_dist.items()}
                },
                'stage2_vs_stage3': {
                    'in_dist': {k: {
                        'stage2_mean': float(v['baseline_mean']),
                        'stage2_std': float(v['baseline_std']),
                        'stage3_mean': float(v['pde_mean']),
                        'stage3_std': float(v['pde_std']),
                        'improvement': float(v['improvement']),
                        't_pvalue': float(v['t_pvalue']),
                        'wilcoxon_pvalue': float(v['wilcoxon_pvalue']),
                        'significant': bool(v['significant'])
                    } for k, v in comparison_results['in_dist'].items()},
                    'out_dist': {k: {
                        'stage2_mean': float(v['baseline_mean']),
                        'stage2_std': float(v['baseline_std']),
                        'stage3_mean': float(v['pde_mean']),
                        'stage3_std': float(v['pde_std']),
                        'improvement': float(v['improvement']),
                        't_pvalue': float(v['t_pvalue']),
                        'wilcoxon_pvalue': float(v['wilcoxon_pvalue']),
                        'significant': bool(v['significant'])
                    } for k, v in comparison_results['out_dist'].items()}
                }
            }
        }
    
    # Save final model
    # For two-stage with PDE (e.g., A2), we already saved baseline and PDE models above
    # For three-stage, we already evaluated and compared Stage II vs Stage III above
    # This final save is for single-stage configurations only
    if not (config.use_two_stage and config.use_pde and not config.use_three_stage) and not config.use_three_stage:
        model_filename = f"{config.name.replace(' ', '_').lower()}_{config.seed}.pth"
        model_path = variant_output_dir / model_filename
        torch.save(model.state_dict(), model_path)
        print(f"Model saved to: {model_path}")
        
        # Evaluate on both test sets for single-stage configurations
        print(f"\nEvaluating on test sets...")
        
        try:
            from .evaluate import evaluate_model
        except ImportError:
            from src.evaluate import evaluate_model
        
        # Evaluate on in-distribution test set
        print(f"\n  In-distribution test set:")
        in_dist_test_dataset = CellSegmentationDataset(in_dist_test_dir, in_dist_test_json)
        in_dist_test_loader = DataLoader(
            in_dist_test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=2,
            pin_memory=True if torch.cuda.is_available() else False
        )
        in_dist_test_metrics = evaluate_model(
            model,
            in_dist_test_loader,
            device,
            threshold=0.5
        )
        
        # Evaluate on out-of-distribution test set
        print(f"\n  Out-of-distribution test set:")
        out_dist_test_dataset = CellSegmentationDataset(out_dist_test_dir, out_dist_test_json)
        out_dist_test_loader = DataLoader(
            out_dist_test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=2,
            pin_memory=True if torch.cuda.is_available() else False
        )
        out_dist_test_metrics = evaluate_model(
            model,
            out_dist_test_loader,
            device,
            threshold=0.5
        )
        
        return {
            'config': config.to_dict(),
            'model_path': str(model_path),
            'in_dist_metrics': {k: v.tolist() if isinstance(v, np.ndarray) else v 
                               for k, v in in_dist_test_metrics.items()},
            'out_dist_metrics': {k: v.tolist() if isinstance(v, np.ndarray) else v 
                                for k, v in out_dist_test_metrics.items()},
            'metrics': {k: v.tolist() if isinstance(v, np.ndarray) else v 
                       for k, v in in_dist_test_metrics.items()}  # Default to in-dist for backward compatibility
        }
    else:
        # For two-stage with PDE (A2), we already evaluated both models above
        # Use the PDE model path as the final model path
        model_path = variant_output_dir / f"{config.name.replace(' ', '_').lower()}_after_pde_stage2.pth"
        
        # Verify that all required variables are set for A2 comparison
        if baseline_test_metrics is None or pde_test_metrics is None or comparison_results is None:
            raise ValueError(
                f"Stage comparison variables not set for {config.name}. "
                "This should only happen for two-stage with PDE configurations (A2)."
            )
        
        return {
            'config': config.to_dict(),
            'model_path': str(model_path),
            'baseline_model_path': str(baseline_model_path) if baseline_model_path else None,
            'pde_model_path': str(pde_model_path) if pde_model_path else None,
            'baseline_in_dist_metrics': {k: v.tolist() if isinstance(v, np.ndarray) else v 
                                        for k, v in baseline_test_metrics['in_dist'].items()},
            'baseline_out_dist_metrics': {k: v.tolist() if isinstance(v, np.ndarray) else v 
                                          for k, v in baseline_test_metrics['out_dist'].items()},
            'pde_in_dist_metrics': {k: v.tolist() if isinstance(v, np.ndarray) else v 
                                   for k, v in pde_test_metrics['in_dist'].items()},
            'pde_out_dist_metrics': {k: v.tolist() if isinstance(v, np.ndarray) else v 
                                    for k, v in pde_test_metrics['out_dist'].items()},
            'metrics': {k: v.tolist() if isinstance(v, np.ndarray) else v 
                       for k, v in pde_test_metrics['in_dist'].items()},  # Default to in-dist PDE metrics
            'in_dist_metrics': {k: v.tolist() if isinstance(v, np.ndarray) else v 
                               for k, v in pde_test_metrics['in_dist'].items()},
            'out_dist_metrics': {k: v.tolist() if isinstance(v, np.ndarray) else v 
                                for k, v in pde_test_metrics['out_dist'].items()},
            'stage_comparison': {
                'in_dist': {k: {
                    'baseline_mean': float(v['baseline_mean']),
                    'baseline_std': float(v['baseline_std']),
                    'pde_mean': float(v['pde_mean']),
                    'pde_std': float(v['pde_std']),
                    'improvement': float(v['improvement']),
                    't_pvalue': float(v['t_pvalue']),
                    'wilcoxon_pvalue': float(v['wilcoxon_pvalue']),
                    'significant': bool(v['significant'])
                } for k, v in comparison_results['in_dist'].items()},
                'out_dist': {k: {
                    'baseline_mean': float(v['baseline_mean']),
                    'baseline_std': float(v['baseline_std']),
                    'pde_mean': float(v['pde_mean']),
                    'pde_std': float(v['pde_std']),
                    'improvement': float(v['improvement']),
                    't_pvalue': float(v['t_pvalue']),
                    'wilcoxon_pvalue': float(v['wilcoxon_pvalue']),
                    'significant': bool(v['significant'])
                } for k, v in comparison_results['out_dist'].items()}
            }
        }


def run_ablation_study(
    ablation_name: str,
    variants: List[AblationConfig],
    train_dir: Path,
    train_json: Path,
    val_dir: Path,
    val_json: Path,
    in_dist_test_dir: Path,
    in_dist_test_json: Path,
    out_dist_test_dir: Path,
    out_dist_test_json: Path,
    device: torch.device,
    batch_size: int = 8,
    learning_rate: float = 1e-4,
    stage1_epochs: int = 50,
    stage2_epochs: int = 50,
    early_stopping_patience: int = 10,
    output_dir: Optional[Path] = None
) -> Dict:
    """
    Run a complete ablation study with multiple variants (single run per variant).
    
    Args:
        ablation_name: Name of the ablation study (e.g., "A1", "A2")
        variants: List of ablation configurations to test
        train_dir: Training images directory
        train_json: Training annotations JSON
        val_dir: Validation images directory
        val_json: Validation annotations JSON
        in_dist_test_dir: In-distribution test images directory
        in_dist_test_json: In-distribution test annotations JSON
        out_dist_test_dir: Out-of-distribution test images directory
        out_dist_test_json: Out-of-distribution test annotations JSON
        device: Device to run on
        batch_size: Batch size for training
        learning_rate: Learning rate
        stage1_epochs: Max epochs for stage 1
        stage2_epochs: Max epochs for stage 2
        early_stopping_patience: Early stopping patience
        output_dir: Directory to save results
        
    Returns:
        Dictionary containing all results and summary statistics
    """
    # Create ablation-specific folder in output/ablation/{ablation_name}/
    results_output_dir = Path(__file__).parent.parent / 'output' / 'ablation'
    results_output_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    ablation_folder = results_output_dir / f"{ablation_name}_{timestamp}"
    ablation_folder.mkdir(parents=True, exist_ok=True)
    
    print("=" * 70)
    print(f"ABLATION STUDY: {ablation_name}")
    print("=" * 70)
    print(f"Output folder: {ablation_folder}")
    print(f"Number of variants: {len(variants)}")
    print(f"Total experiments: {len(variants)}")
    
    all_results = []
    
    # Run each variant once
    for variant in variants:
        # Use seed from config (default is 42)
        result = run_ablation_variant(
            variant,
            train_dir,
            train_json,
            val_dir,
            val_json,
            in_dist_test_dir,
            in_dist_test_json,
            out_dist_test_dir,
            out_dist_test_json,
            device,
            batch_size=batch_size,
            learning_rate=learning_rate,
            stage1_epochs=stage1_epochs,
            stage2_epochs=stage2_epochs,
            early_stopping_patience=early_stopping_patience,
            output_dir=None,  # Not used anymore
            ablation_folder=ablation_folder
        )
        
        all_results.append(result)
    
    # Aggregate results by variant for both test sets
    aggregated_results = {}
    aggregated_results_in_dist = {}
    aggregated_results_out_dist = {}
    
    for variant in variants:
        variant_name = variant.name
        variant_runs = [r for r in all_results if r['config']['name'] == variant_name]
        
        # Aggregate metrics across runs for in-distribution
        aggregated_metrics_in_dist = {
            'dice_scores': [],
            'iou_scores': [],
            'boundary_f1_scores': [],
            'hausdorff_distances': []
        }
        
        # Aggregate metrics across runs for out-of-distribution
        aggregated_metrics_out_dist = {
            'dice_scores': [],
            'iou_scores': [],
            'boundary_f1_scores': [],
            'hausdorff_distances': []
        }
        
        for run in variant_runs:
            # Aggregate in-distribution metrics
            if 'in_dist_metrics' in run:
                for metric_name in aggregated_metrics_in_dist.keys():
                    if metric_name in run['in_dist_metrics']:
                        aggregated_metrics_in_dist[metric_name].extend(run['in_dist_metrics'][metric_name])
            
            # Aggregate out-of-distribution metrics
            if 'out_dist_metrics' in run:
                for metric_name in aggregated_metrics_out_dist.keys():
                    if metric_name in run['out_dist_metrics']:
                        aggregated_metrics_out_dist[metric_name].extend(run['out_dist_metrics'][metric_name])
        
        # Convert to numpy arrays and compute statistics for in-distribution
        aggregated_results_in_dist[variant_name] = {}
        for metric_name, values in aggregated_metrics_in_dist.items():
            if values:
                values_array = np.array(values)
                aggregated_results_in_dist[variant_name][metric_name] = {
                    'mean': float(np.mean(values_array)),
                    'std': 0.0,  # No std for single run
                    'count': len(values_array),
                    'values': values_array.tolist()
                }
        
        # Convert to numpy arrays and compute statistics for out-of-distribution
        aggregated_results_out_dist[variant_name] = {}
        for metric_name, values in aggregated_metrics_out_dist.items():
            if values:
                values_array = np.array(values)
                aggregated_results_out_dist[variant_name][metric_name] = {
                    'mean': float(np.mean(values_array)),
                    'std': 0.0,  # No std for single run
                    'count': len(values_array),
                    'values': values_array.tolist()
                }
        
        # For backward compatibility, use in-distribution as default
        aggregated_results[variant_name] = aggregated_results_in_dist[variant_name]
    
    # Save results (JSON and CSV go to ablation folder)
    results_json = ablation_folder / f"ablation_{ablation_name}_{timestamp}.json"
    with open(results_json, 'w') as f:
        json.dump({
            'ablation_name': ablation_name,
            'variants': [v.to_dict() for v in variants],
            'num_runs': 1,
            'results': all_results,
            'aggregated_results': aggregated_results,  # Backward compatibility (in-dist)
            'aggregated_results_in_dist': aggregated_results_in_dist,
            'aggregated_results_out_dist': aggregated_results_out_dist
        }, f, indent=2)
    
    # Create summary DataFrames for both test sets
    # In-distribution summary
    summary_data_in_dist = []
    for variant_name, metrics in aggregated_results_in_dist.items():
        for metric_name, stats_dict in metrics.items():
            summary_data_in_dist.append({
                'variant': variant_name,
                'metric': metric_name,
                'mean': stats_dict['mean'],
                'std': stats_dict['std'],
                'count': stats_dict['count']
            })
    
    summary_df_in_dist = pd.DataFrame(summary_data_in_dist)
    summary_csv_in_dist = ablation_folder / f"ablation_{ablation_name}_{timestamp}_summary_in_dist.csv"
    summary_df_in_dist.to_csv(summary_csv_in_dist, index=False)
    
    # Out-of-distribution summary
    summary_data_out_dist = []
    for variant_name, metrics in aggregated_results_out_dist.items():
        for metric_name, stats_dict in metrics.items():
            summary_data_out_dist.append({
                'variant': variant_name,
                'metric': metric_name,
                'mean': stats_dict['mean'],
                'std': stats_dict['std'],
                'count': stats_dict['count']
            })
    
    summary_df_out_dist = pd.DataFrame(summary_data_out_dist)
    summary_csv_out_dist = ablation_folder / f"ablation_{ablation_name}_{timestamp}_summary_out_dist.csv"
    summary_df_out_dist.to_csv(summary_csv_out_dist, index=False)
    
    # Backward compatibility: also save the old format (in-dist)
    summary_df = summary_df_in_dist.copy()
    summary_csv = ablation_folder / f"ablation_{ablation_name}_{timestamp}_summary.csv"
    summary_df.to_csv(summary_csv, index=False)
    
    print(f"\n{'='*70}")
    print("ABLATION STUDY COMPLETE")
    print(f"{'='*70}")
    print(f"All files saved to: {ablation_folder}")
    print(f"  - Results JSON: {results_json.name}")
    print(f"  - Summary CSV (in-dist): {summary_csv_in_dist.name}")
    print(f"  - Summary CSV (out-dist): {summary_csv_out_dist.name}")
    print(f"  - Summary CSV (legacy): {summary_csv.name}")
    print(f"  - Model checkpoints: {len(variants)} files")
    print(f"  - Training metrics: CSV files for each variant and stage")
    
    # Print summary for both test sets
    print("\nSummary Statistics - In-Distribution:")
    print("-" * 70)
    for variant_name, metrics in aggregated_results_in_dist.items():
        print(f"\n{variant_name}:")
        for metric_name, stats_dict in metrics.items():
            print(f"  {metric_name}: {stats_dict['mean']:.4f}")
    
    print("\nSummary Statistics - Out-of-Distribution:")
    print("-" * 70)
    for variant_name, metrics in aggregated_results_out_dist.items():
        print(f"\n{variant_name}:")
        for metric_name, stats_dict in metrics.items():
            print(f"  {metric_name}: {stats_dict['mean']:.4f}")
    
    return {
        'ablation_name': ablation_name,
        'results_json': str(results_json),
        'summary_csv': str(summary_csv),
        'aggregated_results': aggregated_results
    }


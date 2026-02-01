import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Optional, List
import numpy as np


def plot_training_curves(
    csv_path: Path,
    output_path: Optional[Path] = None,
    show_plot: bool = False
):
    """
    Plot training curves from a CSV metrics file.
    
    Creates multiple subplots showing:
    - Training and validation loss
    - Validation Dice score
    - Loss components (Dice, BCE, PDE if available)
    
    Args:
        csv_path: Path to CSV file with training metrics
        output_path: Path to save the plot (if None, saves next to CSV)
        show_plot: Whether to display the plot interactively
    """
    # Load data
    df = pd.read_csv(csv_path)
    
    # Determine output path
    if output_path is None:
        output_path = csv_path.parent / f"{csv_path.stem}_curves.png"
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f'Training Curves: {csv_path.stem}', fontsize=16, fontweight='bold')
    
    # 1. Total Loss (Train vs Val)
    ax1 = axes[0, 0]
    ax1.plot(df['epoch'], df['train_loss'], label='Train Loss', linewidth=2, color='#2E86AB')
    ax1.plot(df['epoch'], df['val_loss'], label='Val Loss', linewidth=2, color='#A23B72')
    ax1.set_xlabel('Epoch', fontsize=11)
    ax1.set_ylabel('Loss', fontsize=11)
    ax1.set_title('Total Loss', fontsize=12, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. Validation Dice Score
    ax2 = axes[0, 1]
    ax2.plot(df['epoch'], df['val_dice_score'], label='Val Dice Score', 
             linewidth=2, color='#06A77D', marker='o', markersize=4)
    ax2.set_xlabel('Epoch', fontsize=11)
    ax2.set_ylabel('Dice Score', fontsize=11)
    ax2.set_title('Validation Dice Score', fontsize=12, fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim([0, 1])
    
    # 3. Training Loss Components
    ax3 = axes[1, 0]
    ax3.plot(df['epoch'], df['train_dice_loss'], label='Dice Loss', 
             linewidth=2, linestyle='--', alpha=0.8)
    ax3.plot(df['epoch'], df['train_bce_loss'], label='BCE Loss', 
             linewidth=2, linestyle='--', alpha=0.8)
    if df['train_pde_loss'].sum() > 0:  # Only plot if PDE loss exists
        ax3.plot(df['epoch'], df['train_pde_loss'], label='PDE Loss', 
                 linewidth=2, linestyle='--', alpha=0.8, color='#F18F01')
    ax3.set_xlabel('Epoch', fontsize=11)
    ax3.set_ylabel('Loss', fontsize=11)
    ax3.set_title('Training Loss Components', fontsize=12, fontweight='bold')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # 4. Validation Loss Components
    ax4 = axes[1, 1]
    ax4.plot(df['epoch'], df['val_dice_loss'], label='Dice Loss', 
             linewidth=2, linestyle='--', alpha=0.8)
    ax4.plot(df['epoch'], df['val_bce_loss'], label='BCE Loss', 
             linewidth=2, linestyle='--', alpha=0.8)
    if df['val_pde_loss'].sum() > 0:  # Only plot if PDE loss exists
        ax4.plot(df['epoch'], df['val_pde_loss'], label='PDE Loss', 
                 linewidth=2, linestyle='--', alpha=0.8, color='#F18F01')
    ax4.set_xlabel('Epoch', fontsize=11)
    ax4.set_ylabel('Loss', fontsize=11)
    ax4.set_title('Validation Loss Components', fontsize=12, fontweight='bold')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Training curves saved to: {output_path}")
    
    if show_plot:
        plt.show()
    else:
        plt.close()


def plot_combined_stage_loss(
    csv_path_stage1: Path,
    csv_path_stage2: Path,
    output_path: Optional[Path] = None,
    show_plot: bool = False
):
    """
    Plot combined total loss for both Stage 1 and Stage 2 with stage transition marker.
    
    Creates a single plot showing:
    - Training total loss for Stage 1 and Stage 2
    - Validation total loss for Stage 1 and Stage 2
    - Clear visual marker for stage transition
    
    Args:
        csv_path_stage1: Path to Stage I metrics CSV
        csv_path_stage2: Path to Stage II metrics CSV
        output_path: Path to save the plot (if None, saves to output directory)
        show_plot: Whether to display the plot interactively
    """
    # Load data
    df1 = pd.read_csv(csv_path_stage1)
    df2 = pd.read_csv(csv_path_stage2)
    
    # Determine output path
    if output_path is None:
        output_dir = csv_path_stage1.parent
        timestamp = csv_path_stage1.stem.split('_')[-1] if '_' in csv_path_stage1.stem else 'combined'
        output_path = output_dir / f"combined_loss_{timestamp}.png"
    
    # Get stage transition point
    stage1_max_epoch = df1['epoch'].max()
    stage2_start_epoch = stage1_max_epoch
    
    # Adjust Stage 2 epochs to continue from Stage 1
    df2_adjusted = df2.copy()
    df2_adjusted['epoch'] = df2_adjusted['epoch'] + stage1_max_epoch
    
    # Create figure
    fig, ax = plt.subplots(figsize=(14, 8))
    
    # Plot Stage 1
    ax.plot(df1['epoch'], df1['train_loss'], 
            label='Train Loss (Stage 1)', 
            linewidth=2.5, 
            color='#2E86AB',
            alpha=0.9)
    ax.plot(df1['epoch'], df1['val_loss'], 
            label='Val Loss (Stage 1)', 
            linewidth=2.5, 
            color='#A23B72',
            alpha=0.9)
    
    # Plot Stage 2
    ax.plot(df2_adjusted['epoch'], df2_adjusted['train_loss'], 
            label='Train Loss (Stage 2)', 
            linewidth=2.5, 
            color='#06A77D',
            alpha=0.9)
    ax.plot(df2_adjusted['epoch'], df2_adjusted['val_loss'], 
            label='Val Loss (Stage 2)', 
            linewidth=2.5, 
            color='#F18F01',
            alpha=0.9)
    
    # Add vertical line for stage transition
    ax.axvline(x=stage1_max_epoch, 
               color='red', 
               linestyle='--', 
               linewidth=2, 
               alpha=0.7,
               label='Stage Transition')
    
    # Add text annotation for stage transition
    ax.text(stage1_max_epoch, ax.get_ylim()[1] * 0.95, 
            'Stage 1 → Stage 2', 
            rotation=90, 
            verticalalignment='top',
            horizontalalignment='right',
            fontsize=11,
            fontweight='bold',
            color='red',
            bbox=dict(boxstyle='round,pad=0.5', facecolor='white', edgecolor='red', alpha=0.8))
    
    # Styling
    ax.set_xlabel('Epoch', fontsize=13, fontweight='bold')
    ax.set_ylabel('Total Loss', fontsize=13, fontweight='bold')
    ax.set_title('Combined Training: Total Loss (Stage 1 + Stage 2)', 
                 fontsize=15, fontweight='bold', pad=20)
    ax.legend(loc='best', fontsize=11, framealpha=0.9)
    ax.grid(True, alpha=0.3, linestyle='--')
    
    # Add stage labels on x-axis
    stage1_mid = stage1_max_epoch / 2
    stage2_mid = stage1_max_epoch + (df2_adjusted['epoch'].max() - stage1_max_epoch) / 2
    
    ax.text(stage1_mid, ax.get_ylim()[0] - (ax.get_ylim()[1] - ax.get_ylim()[0]) * 0.05,
            'Stage 1\n(Baseline)', 
            horizontalalignment='center',
            fontsize=10,
            fontweight='bold',
            color='#2E86AB',
            bbox=dict(boxstyle='round,pad=0.5', facecolor='#E8F4F8', edgecolor='#2E86AB', alpha=0.7))
    
    ax.text(stage2_mid, ax.get_ylim()[0] - (ax.get_ylim()[1] - ax.get_ylim()[0]) * 0.05,
            'Stage 2\n(PDE-Constrained)', 
            horizontalalignment='center',
            fontsize=10,
            fontweight='bold',
            color='#06A77D',
            bbox=dict(boxstyle='round,pad=0.5', facecolor='#E8F8F0', edgecolor='#06A77D', alpha=0.7))
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Combined loss plot saved to: {output_path}")
    
    if show_plot:
        plt.show()
    else:
        plt.close()


def plot_two_stage_comparison(
    csv_path_stage1: Path,
    csv_path_stage2: Path,
    output_path: Optional[Path] = None,
    show_plot: bool = False
):
    """
    Plot comparison between Stage I (baseline) and Stage II (PDE-constrained) training.
    
    Args:
        csv_path_stage1: Path to Stage I metrics CSV
        csv_path_stage2: Path to Stage II metrics CSV
        output_path: Path to save the plot
        show_plot: Whether to display the plot interactively
    """
    # Load data
    df1 = pd.read_csv(csv_path_stage1)
    df2 = pd.read_csv(csv_path_stage2)
    
    # Adjust Stage II epochs to continue from Stage I
    max_epoch_stage1 = df1['epoch'].max()
    df2_adjusted = df2.copy()
    df2_adjusted['epoch'] = df2_adjusted['epoch'] + max_epoch_stage1
    
    # Determine output path
    if output_path is None:
        output_path = csv_path_stage1.parent / "two_stage_comparison.png"
    
    # Create figure
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Two-Stage Training Comparison', fontsize=16, fontweight='bold')
    
    # 1. Total Loss Comparison
    ax1 = axes[0, 0]
    ax1.plot(df1['epoch'], df1['val_loss'], label='Stage I (Baseline)', 
             linewidth=2, color='#2E86AB', linestyle='-')
    ax1.plot(df2_adjusted['epoch'], df2_adjusted['val_loss'], 
             label='Stage II (PDE-constrained)', linewidth=2, color='#A23B72', linestyle='-')
    ax1.axvline(x=max_epoch_stage1, color='gray', linestyle='--', alpha=0.5, label='Stage Transition')
    ax1.set_xlabel('Epoch', fontsize=11)
    ax1.set_ylabel('Validation Loss', fontsize=11)
    ax1.set_title('Validation Loss', fontsize=12, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. Dice Score Comparison
    ax2 = axes[0, 1]
    ax2.plot(df1['epoch'], df1['val_dice_score'], label='Stage I (Baseline)', 
             linewidth=2, color='#2E86AB', marker='o', markersize=4)
    ax2.plot(df2_adjusted['epoch'], df2_adjusted['val_dice_score'], 
             label='Stage II (PDE-constrained)', linewidth=2, color='#A23B72', 
             marker='s', markersize=4)
    ax2.axvline(x=max_epoch_stage1, color='gray', linestyle='--', alpha=0.5, label='Stage Transition')
    ax2.set_xlabel('Epoch', fontsize=11)
    ax2.set_ylabel('Validation Dice Score', fontsize=11)
    ax2.set_title('Validation Dice Score', fontsize=12, fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim([0, 1])
    
    # 3. PDE Loss (Stage II only)
    ax3 = axes[1, 0]
    if df2['val_pde_loss'].sum() > 0:
        ax3.plot(df2_adjusted['epoch'], df2_adjusted['val_pde_loss'], 
                 label='PDE Loss', linewidth=2, color='#F18F01')
        ax3.axvline(x=max_epoch_stage1, color='gray', linestyle='--', alpha=0.5, label='Stage Transition')
        ax3.set_xlabel('Epoch', fontsize=11)
        ax3.set_ylabel('PDE Loss', fontsize=11)
        ax3.set_title('PDE Regularization Loss (Stage II)', fontsize=12, fontweight='bold')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
    else:
        ax3.text(0.5, 0.5, 'No PDE Loss Data', ha='center', va='center', 
                transform=ax3.transAxes, fontsize=12)
        ax3.set_title('PDE Regularization Loss', fontsize=12, fontweight='bold')
    
    # 4. Improvement Analysis
    ax4 = axes[1, 1]
    best_dice_stage1 = df1['val_dice_score'].max()
    best_dice_stage2 = df2['val_dice_score'].max()
    improvement = best_dice_stage2 - best_dice_stage1
    
    bars = ax4.bar(['Stage I\n(Baseline)', 'Stage II\n(PDE-constrained)'], 
                   [best_dice_stage1, best_dice_stage2],
                   color=['#2E86AB', '#A23B72'], alpha=0.7, edgecolor='black', linewidth=1.5)
    ax4.set_ylabel('Best Validation Dice Score', fontsize=11)
    ax4.set_title(f'Best Performance Comparison\n(Improvement: {improvement:+.4f})', 
                  fontsize=12, fontweight='bold')
    ax4.set_ylim([0, 1])
    ax4.grid(True, alpha=0.3, axis='y')
    
    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        ax4.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.4f}',
                ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Two-stage comparison plot saved to: {output_path}")
    
    if show_plot:
        plt.show()
    else:
        plt.close()


def plot_all_metrics(
    csv_path: Path,
    output_path: Optional[Path] = None,
    show_plot: bool = False
):
    """
    Create a comprehensive plot with all metrics in separate subplots.
    
    Args:
        csv_path: Path to CSV file with training metrics
        output_path: Path to save the plot
        show_plot: Whether to display the plot interactively
    """
    # Load data
    df = pd.read_csv(csv_path)
    
    # Determine output path
    if output_path is None:
        output_path = csv_path.parent / f"{csv_path.stem}_all_metrics.png"
    
    # Count number of subplots needed
    num_plots = 6  # loss, dice_score, train components, val components, and potentially PDE
    has_pde = df['train_pde_loss'].sum() > 0 or df['val_pde_loss'].sum() > 0
    
    if has_pde:
        fig, axes = plt.subplots(3, 2, figsize=(14, 12))
    else:
        fig, axes = plt.subplots(3, 2, figsize=(14, 12))
    
    fig.suptitle(f'All Training Metrics: {csv_path.stem}', fontsize=16, fontweight='bold')
    axes = axes.flatten()
    
    plot_idx = 0
    
    # 1. Total Loss
    ax = axes[plot_idx]
    ax.plot(df['epoch'], df['train_loss'], label='Train', linewidth=2)
    ax.plot(df['epoch'], df['val_loss'], label='Val', linewidth=2)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.set_title('Total Loss')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plot_idx += 1
    
    # 2. Dice Score
    ax = axes[plot_idx]
    ax.plot(df['epoch'], df['val_dice_score'], label='Val Dice Score', 
            linewidth=2, color='green', marker='o', markersize=3)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Dice Score')
    ax.set_title('Validation Dice Score')
    ax.set_ylim([0, 1])
    ax.legend()
    ax.grid(True, alpha=0.3)
    plot_idx += 1
    
    # 3. Training Loss Components
    ax = axes[plot_idx]
    ax.plot(df['epoch'], df['train_dice_loss'], label='Dice', linewidth=2, linestyle='--')
    ax.plot(df['epoch'], df['train_bce_loss'], label='BCE', linewidth=2, linestyle='--')
    if has_pde:
        ax.plot(df['epoch'], df['train_pde_loss'], label='PDE', linewidth=2, linestyle='--', color='orange')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.set_title('Training Loss Components')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plot_idx += 1
    
    # 4. Validation Loss Components
    ax = axes[plot_idx]
    ax.plot(df['epoch'], df['val_dice_loss'], label='Dice', linewidth=2, linestyle='--')
    ax.plot(df['epoch'], df['val_bce_loss'], label='BCE', linewidth=2, linestyle='--')
    if has_pde:
        ax.plot(df['epoch'], df['val_pde_loss'], label='PDE', linewidth=2, linestyle='--', color='orange')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.set_title('Validation Loss Components')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plot_idx += 1
    
    # 5. Training vs Validation Dice Loss
    ax = axes[plot_idx]
    ax.plot(df['epoch'], df['train_dice_loss'], label='Train Dice Loss', linewidth=2)
    ax.plot(df['epoch'], df['val_dice_loss'], label='Val Dice Loss', linewidth=2)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Dice Loss')
    ax.set_title('Dice Loss: Train vs Val')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plot_idx += 1
    
    # 6. Training vs Validation BCE Loss
    ax = axes[plot_idx]
    ax.plot(df['epoch'], df['train_bce_loss'], label='Train BCE Loss', linewidth=2)
    ax.plot(df['epoch'], df['val_bce_loss'], label='Val BCE Loss', linewidth=2)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('BCE Loss')
    ax.set_title('BCE Loss: Train vs Val')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plot_idx += 1
    
    # Hide unused subplots
    for i in range(plot_idx, len(axes)):
        axes[i].axis('off')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"All metrics plot saved to: {output_path}")
    
    if show_plot:
        plt.show()
    else:
        plt.close()


def plot_training_results(
    csv_path_stage1: Optional[Path] = None,
    csv_path_stage2: Optional[Path] = None,
    output_dir: Optional[Path] = None,
    show_plots: bool = False
):
    """
    Main function to plot training results.
    
    Automatically detects whether it's a two-stage or single-stage training
    and creates appropriate plots.
    
    Args:
        csv_path_stage1: Path to Stage I metrics CSV (or single-stage CSV)
        csv_path_stage2: Path to Stage II metrics CSV (optional)
        output_dir: Directory to save plots (defaults to CSV directory)
        show_plots: Whether to display plots interactively
    """
    if csv_path_stage1 is None:
        print("No CSV file provided for plotting.")
        return
    
    csv_path_stage1 = Path(csv_path_stage1)
    
    if not csv_path_stage1.exists():
        print(f"CSV file not found: {csv_path_stage1}")
        return
    
    # Determine output directory
    if output_dir is None:
        output_dir = csv_path_stage1.parent
    else:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
    
    # Plot Stage I (or single-stage) metrics
    plot_training_curves(
        csv_path_stage1,
        output_path=output_dir / f"{csv_path_stage1.stem}_curves.png",
        show_plot=show_plots
    )
    
    plot_all_metrics(
        csv_path_stage1,
        output_path=output_dir / f"{csv_path_stage1.stem}_all_metrics.png",
        show_plot=show_plots
    )
    
    # If Stage II exists, plot it and create comparison
    if csv_path_stage2 is not None:
        csv_path_stage2 = Path(csv_path_stage2)
        
        if not csv_path_stage2.exists():
            print(f"Stage II CSV file not found: {csv_path_stage2}")
        else:
            # Plot Stage II metrics
            plot_training_curves(
                csv_path_stage2,
                output_path=output_dir / f"{csv_path_stage2.stem}_curves.png",
                show_plot=show_plots
            )
            
            plot_all_metrics(
                csv_path_stage2,
                output_path=output_dir / f"{csv_path_stage2.stem}_all_metrics.png",
                show_plot=show_plots
            )
            
            # Create combined loss plot
            plot_combined_stage_loss(
                csv_path_stage1,
                csv_path_stage2,
                output_path=output_dir / "combined_loss_stage1_stage2.png",
                show_plot=show_plots
            )
            
            # Create two-stage comparison
            plot_two_stage_comparison(
                csv_path_stage1,
                csv_path_stage2,
                output_path=output_dir / "two_stage_comparison.png",
                show_plot=show_plots
            )
        csv_path_stage2 = Path(csv_path_stage2)
        if csv_path_stage2.exists():
            # Plot Stage II metrics
            plot_training_curves(
                csv_path_stage2,
                output_path=output_dir / f"{csv_path_stage2.stem}_curves.png",
                show_plot=show_plots
            )
            
            plot_all_metrics(
                csv_path_stage2,
                output_path=output_dir / f"{csv_path_stage2.stem}_all_metrics.png",
                show_plot=show_plots
            )
            
            # Plot comparison
            plot_two_stage_comparison(
                csv_path_stage1,
                csv_path_stage2,
                output_path=output_dir / "two_stage_comparison.png",
                show_plot=show_plots
            )
    
    print(f"\nAll plots saved to: {output_dir}")


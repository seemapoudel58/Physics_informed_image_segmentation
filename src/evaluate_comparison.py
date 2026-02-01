import torch
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional, Any
import json
from datetime import datetime

# Handle both relative and absolute imports
try:
    from .unet import UNet
    from .evaluate import (
        evaluate_on_test_set,
        compare_models_statistically,
        format_metric_report,
        compute_statistics
    )
except ImportError:
    # Fallback for direct execution
    import sys
    from pathlib import Path as PathLib
    sys.path.insert(0, str(PathLib(__file__).parent.parent))
    from src.unet import UNet
    from src.evaluate import (
        evaluate_on_test_set,
        compare_models_statistically,
        format_metric_report,
        compute_statistics
    )


def make_json_serializable(obj: Any) -> Any:
    """
    Convert numpy types and other non-JSON-serializable types to Python native types.
    
    Args:
        obj: Object to convert
        
    Returns:
        JSON-serializable version of the object
    """
    if isinstance(obj, (np.integer, np.int64, np.int32)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float64, np.float32)):
        return float(obj)
    elif isinstance(obj, np.bool_):
        return bool(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {key: make_json_serializable(value) for key, value in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [make_json_serializable(item) for item in obj]
    elif isinstance(obj, (bool, int, float, str)) or obj is None:
        return obj
    else:
        # Try to convert to string as last resort
        return str(obj)


def load_model(model_path: Path, device: torch.device) -> UNet:
    """
    Load a trained UNet model from checkpoint.
    
    Args:
        model_path: Path to model checkpoint
        device: Device to load model on
        
    Returns:
        Loaded UNet model
    """
    model = UNet(in_channels=1, out_channels=1, base_channels=64)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model = model.to(device)
    model.eval()
    return model


def evaluate_and_compare(
    baseline_model_path: Path,
    pde_model_path: Path,
    test_dir: Path,
    test_json: Path,
    device: torch.device,
    batch_size: int = 8,
    threshold: float = 0.5,
    output_dir: Optional[Path] = None
) -> Dict:
    """
    Evaluate both baseline and PDE-constrained models and perform statistical comparison.
    
    Args:
        baseline_model_path: Path to baseline model checkpoint
        pde_model_path: Path to PDE-constrained model checkpoint
        test_dir: Directory containing test images
        test_json: Path to test annotations JSON file
        device: Device to run evaluation on
        batch_size: Batch size for evaluation (default: 8)
        threshold: Threshold for binarizing predictions (default: 0.5)
        output_dir: Directory to save evaluation results (default: output/)
        
    Returns:
        Dictionary containing evaluation results and statistical comparisons
    """
    if output_dir is None:
        output_dir = Path(__file__).parent.parent / 'output'
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)
    
    print("=" * 70)
    print("MODEL EVALUATION AND STATISTICAL COMPARISON")
    print("=" * 70)
    
    # Load models
    print("\nLoading models...")
    baseline_model = load_model(baseline_model_path, device)
    pde_model = load_model(pde_model_path, device)
    
    # Evaluate baseline model
    baseline_metrics = evaluate_on_test_set(
        baseline_model,
        test_dir,
        test_json,
        device,
        batch_size=batch_size,
        threshold=threshold,
        model_name="Baseline (Unconstrained)"
    )
    
    # Evaluate PDE-constrained model
    pde_metrics = evaluate_on_test_set(
        pde_model,
        test_dir,
        test_json,
        device,
        batch_size=batch_size,
        threshold=threshold,
        model_name="PDE-Constrained"
    )
    
    # Statistical comparison
    print("\n" + "=" * 70)
    print("STATISTICAL COMPARISON")
    print("=" * 70)
    
    comparison_results = compare_models_statistically(
        baseline_metrics,
        pde_metrics,
        alpha=0.05
    )
    
    # Print comparison results
    print("\nStatistical Test Results (α = 0.05):")
    print("-" * 70)
    
    for metric_name, results in comparison_results.items():
        print(f"\n{metric_name.replace('_', ' ').title()}:")
        print(f"  Baseline Mean: {results['baseline_mean']:.4f}")
        print(f"  PDE Mean:      {results['pde_mean']:.4f}")
        print(f"  Improvement:   {results['improvement']:+.4f}")
        print(f"  Paired t-test:")
        print(f"    t-statistic: {results['t_statistic']:.4f}")
        print(f"    p-value:     {results['t_pvalue']:.4f}")
        print(f"  Wilcoxon signed-rank test:")
        print(f"    statistic:   {results['wilcoxon_statistic']:.4f}")
        print(f"    p-value:     {results['wilcoxon_pvalue']:.4f}")
        print(f"  Significant:  {'Yes' if results['significant'] else 'No'}")
    
    # Save results to CSV
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Save per-image metrics
    results_df = pd.DataFrame({
        'image_id': range(len(baseline_metrics['dice_scores'])),
        'baseline_dice': baseline_metrics['dice_scores'],
        'pde_dice': pde_metrics['dice_scores'],
        'baseline_iou': baseline_metrics['iou_scores'],
        'pde_iou': pde_metrics['iou_scores'],
        'baseline_boundary_f1': baseline_metrics['boundary_f1_scores'],
        'pde_boundary_f1': pde_metrics['boundary_f1_scores'],
        'baseline_hausdorff': baseline_metrics['hausdorff_distances'],
        'pde_hausdorff': pde_metrics['hausdorff_distances']
    })
    
    results_csv = output_dir / f"evaluation_results_{timestamp}.csv"
    results_df.to_csv(results_csv, index=False)
    print(f"\nPer-image metrics saved to: {results_csv}")
    
    # Save summary statistics
    summary_data = {}
    for metric_name in baseline_metrics.keys():
        baseline_stats = compute_statistics(baseline_metrics[metric_name])
        pde_stats = compute_statistics(pde_metrics[metric_name])
        comparison = comparison_results[metric_name]
        
        summary_data[metric_name] = {
            'baseline_mean': baseline_stats['mean'],
            'baseline_std': baseline_stats['std'],
            'pde_mean': pde_stats['mean'],
            'pde_std': pde_stats['std'],
            'improvement': comparison['improvement'],
            't_pvalue': comparison['t_pvalue'],
            'wilcoxon_pvalue': comparison['wilcoxon_pvalue'],
            'significant': comparison['significant']
        }
    
    summary_df = pd.DataFrame(summary_data).T
    summary_csv = output_dir / f"evaluation_summary_{timestamp}.csv"
    summary_df.to_csv(summary_csv)
    print(f"Summary statistics saved to: {summary_csv}")
    
    # Save detailed comparison results as JSON
    comparison_json = output_dir / f"statistical_comparison_{timestamp}.json"
    # Convert numpy types to Python native types for JSON serialization
    serializable_results = make_json_serializable(comparison_results)
    with open(comparison_json, 'w') as f:
        json.dump(serializable_results, f, indent=2)
    print(f"Statistical comparison saved to: {comparison_json}")
    
    return {
        'baseline_metrics': baseline_metrics,
        'pde_metrics': pde_metrics,
        'comparison_results': comparison_results,
        'results_csv': results_csv,
        'summary_csv': summary_csv,
        'comparison_json': comparison_json
    }


def run_repeated_evaluations(
    baseline_model_paths: List[Path],
    pde_model_paths: List[Path],
    test_dir: Path,
    test_json: Path,
    device: torch.device,
    batch_size: int = 8,
    threshold: float = 0.5,
    output_dir: Optional[Path] = None
) -> Dict:
    """
    Run evaluation on multiple model checkpoints (repeated experiments).
    
    Args:
        baseline_model_paths: List of paths to baseline model checkpoints
        pde_model_paths: List of paths to PDE-constrained model checkpoints
        test_dir: Directory containing test images
        test_json: Path to test annotations JSON file
        device: Device to run evaluation on
        batch_size: Batch size for evaluation (default: 8)
        threshold: Threshold for binarizing predictions (default: 0.5)
        output_dir: Directory to save evaluation results
        
    Returns:
        Dictionary containing aggregated results across all runs
    """
    if output_dir is None:
        output_dir = Path(__file__).parent.parent / 'output'
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)
    
    print("=" * 70)
    print("REPEATED EXPERIMENTS EVALUATION")
    print("=" * 70)
    print(f"Number of runs: {len(baseline_model_paths)}")
    
    all_baseline_metrics = {
        'dice_scores': [],
        'iou_scores': [],
        'boundary_f1_scores': [],
        'hausdorff_distances': []
    }
    
    all_pde_metrics = {
        'dice_scores': [],
        'iou_scores': [],
        'boundary_f1_scores': [],
        'hausdorff_distances': []
    }
    
    # Evaluate each run
    for run_idx, (baseline_path, pde_path) in enumerate(zip(baseline_model_paths, pde_model_paths)):
        print(f"\n{'='*70}")
        print(f"Run {run_idx + 1}/{len(baseline_model_paths)}")
        print(f"{'='*70}")
        
        baseline_model = load_model(baseline_path, device)
        pde_model = load_model(pde_path, device)
        
        baseline_metrics = evaluate_on_test_set(
            baseline_model,
            test_dir,
            test_json,
            device,
            batch_size=batch_size,
            threshold=threshold,
            model_name=f"Baseline Run {run_idx + 1}"
        )
        
        pde_metrics = evaluate_on_test_set(
            pde_model,
            test_dir,
            test_json,
            device,
            batch_size=batch_size,
            threshold=threshold,
            model_name=f"PDE-Constrained Run {run_idx + 1}"
        )
        
        # Aggregate metrics
        for key in all_baseline_metrics.keys():
            all_baseline_metrics[key].extend(baseline_metrics[key])
            all_pde_metrics[key].extend(pde_metrics[key])
    
    # Convert to numpy arrays
    for key in all_baseline_metrics.keys():
        all_baseline_metrics[key] = np.array(all_baseline_metrics[key])
        all_pde_metrics[key] = np.array(all_pde_metrics[key])
    
    # Print aggregated results
    print("\n" + "=" * 70)
    print("AGGREGATED RESULTS (All Runs Combined)")
    print("=" * 70)
    
    baseline_report = format_metric_report(
        all_baseline_metrics,
        model_name="Baseline (All Runs)"
    )
    print(baseline_report)
    
    pde_report = format_metric_report(
        all_pde_metrics,
        model_name="PDE-Constrained (All Runs)"
    )
    print(pde_report)
    
    # Statistical comparison on aggregated data
    comparison_results = compare_models_statistically(
        all_baseline_metrics,
        all_pde_metrics,
        alpha=0.05
    )
    
    print("\n" + "=" * 70)
    print("STATISTICAL COMPARISON (Aggregated)")
    print("=" * 70)
    
    for metric_name, results in comparison_results.items():
        print(f"\n{metric_name.replace('_', ' ').title()}:")
        print(f"  Baseline: {results['baseline_mean']:.4f} ± {results.get('baseline_std', 0):.4f}")
        print(f"  PDE:      {results['pde_mean']:.4f} ± {results.get('pde_std', 0):.4f}")
        print(f"  Improvement: {results['improvement']:+.4f}")
        print(f"  Significant: {'Yes' if results['significant'] else 'No'} (p={results['t_pvalue']:.4f})")
    
    # Save aggregated results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    aggregated_df = pd.DataFrame({
        'metric': [],
        'model': [],
        'mean': [],
        'std': [],
        'count': []
    })
    
    for metric_name in all_baseline_metrics.keys():
        baseline_stats = compute_statistics(all_baseline_metrics[metric_name])
        pde_stats = compute_statistics(all_pde_metrics[metric_name])
        
        aggregated_df = pd.concat([
            aggregated_df,
            pd.DataFrame([{
                'metric': metric_name,
                'model': 'baseline',
                'mean': baseline_stats['mean'],
                'std': baseline_stats['std'],
                'count': baseline_stats['count']
            }]),
            pd.DataFrame([{
                'metric': metric_name,
                'model': 'pde',
                'mean': pde_stats['mean'],
                'std': pde_stats['std'],
                'count': pde_stats['count']
            }])
        ], ignore_index=True)
    
    aggregated_csv = output_dir / f"aggregated_results_{timestamp}.csv"
    aggregated_df.to_csv(aggregated_csv, index=False)
    print(f"\nAggregated results saved to: {aggregated_csv}")
    
    return {
        'baseline_metrics': all_baseline_metrics,
        'pde_metrics': all_pde_metrics,
        'comparison_results': comparison_results,
        'aggregated_csv': aggregated_csv
    }


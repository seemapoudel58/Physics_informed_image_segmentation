"""
Command-line script for evaluating segmentation models.

Usage:
    python evaluate.py --baseline models/unet_baseline.pth --pde models/unet_pde_regularized.pth
    python evaluate.py --baseline models/unet_baseline.pth --pde models/unet_pde_regularized.pth --repeated
"""

import argparse
from pathlib import Path
import torch
from glob import glob

from src.evaluate_comparison import evaluate_and_compare, run_repeated_evaluations


def main():
    parser = argparse.ArgumentParser(
        description='Evaluate and compare segmentation models'
    )
    parser.add_argument(
        '--baseline',
        type=str,
        required=True,
        help='Path to baseline model checkpoint (or pattern for repeated experiments)'
    )
    parser.add_argument(
        '--pde',
        type=str,
        required=True,
        help='Path to PDE-constrained model checkpoint (or pattern for repeated experiments)'
    )
    parser.add_argument(
        '--test-dir',
        type=str,
        default='images/testing',
        help='Directory containing test images (default: images/testing)'
    )
    parser.add_argument(
        '--test-json',
        type=str,
        default='images/annotation/testing_annotation.json',
        help='Path to test annotations JSON (default: images/annotation/testing_annotation.json)'
    )
    parser.add_argument(
        '--batch-size',
        type=int,
        default=8,
        help='Batch size for evaluation (default: 8)'
    )
    parser.add_argument(
        '--threshold',
        type=float,
        default=0.5,
        help='Threshold for binarizing predictions (default: 0.5)'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='output',
        help='Directory to save evaluation results (default: output)'
    )
    parser.add_argument(
        '--repeated',
        action='store_true',
        help='Run repeated experiments evaluation (baseline and pde should be glob patterns)'
    )
    
    args = parser.parse_args()
    
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Convert paths
    test_dir = Path(args.test_dir)
    test_json = Path(args.test_json)
    output_dir = Path(args.output_dir)
    
    if args.repeated:
        # Find all matching model files
        baseline_paths = sorted(glob(args.baseline))
        pde_paths = sorted(glob(args.pde))
        
        if len(baseline_paths) == 0:
            print(f"Error: No baseline models found matching pattern: {args.baseline}")
            return
        
        if len(pde_paths) == 0:
            print(f"Error: No PDE models found matching pattern: {args.pde}")
            return
        
        if len(baseline_paths) != len(pde_paths):
            print(f"Warning: Number of baseline models ({len(baseline_paths)}) != "
                  f"number of PDE models ({len(pde_paths)})")
        
        print(f"\nFound {len(baseline_paths)} baseline models")
        print(f"Found {len(pde_paths)} PDE-constrained models")
        
        baseline_paths = [Path(p) for p in baseline_paths]
        pde_paths = [Path(p) for p in pde_paths]
        
        # Run repeated evaluations
        results = run_repeated_evaluations(
            baseline_model_paths=baseline_paths,
            pde_model_paths=pde_paths,
            test_dir=test_dir,
            test_json=test_json,
            device=device,
            batch_size=args.batch_size,
            threshold=args.threshold,
            output_dir=output_dir
        )
    else:
        # Single evaluation
        baseline_path = Path(args.baseline)
        pde_path = Path(args.pde)
        
        if not baseline_path.exists():
            print(f"Error: Baseline model not found: {baseline_path}")
            return
        
        if not pde_path.exists():
            print(f"Error: PDE model not found: {pde_path}")
            return
        
        # Run evaluation and comparison
        results = evaluate_and_compare(
            baseline_model_path=baseline_path,
            pde_model_path=pde_path,
            test_dir=test_dir,
            test_json=test_json,
            device=device,
            batch_size=args.batch_size,
            threshold=args.threshold,
            output_dir=output_dir
        )
    
    print("\n" + "=" * 70)
    print("EVALUATION COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()


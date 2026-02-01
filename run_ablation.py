"""
Script to run ablation studies for PDE-constrained cell segmentation.

This script defines and runs ablation studies:
- R1: Influence of PDE Constraints (100% Data) - Component Ablation
- R2: Low Sample Regime Analysis - Varying data fractions (10%, 25%, 50%, 75%, 100%)
- R3: Influence of PDE Constraints (10% Data) - Component Ablation with 10% data
- S1: Reaction Threshold Sensitivity Analysis
- S2: Diffusion Coefficient Sensitivity Analysis
- S3: Interface Width Sensitivity Analysis
"""

import argparse
from pathlib import Path
import torch

from src.ablation import (
    AblationConfig,
    run_ablation_study
)


def define_ablation_r1() -> list:
    """
    R1: Influence of PDE Constraints (100% Data)
    
    Tests the contribution of each PDE constraint component:
    1. Baseline: No PDE constraints (Dice + BCE only)
    2. RD Only: Reaction-Diffusion PDE only
    3. Phase-Field Only: Phase-field energy only
    4. RD + Phase-Field: Both constraints combined
    
    Fixed Parameters (optimal):
    - D = 5.0 (diffusion coefficient)
    - a = 0.5 (reaction threshold)
    - ε = 0.05 (interface width)
    - λ_RD = 1e-4 (when enabled)
    - λ_PF = 1e-4 (when enabled)
    - Two-stage training
    - 100% of training data
    """
    return [
        AblationConfig(
            name="R1.0 Baseline",
            description="Baseline UNet (Dice + BCE only, no PDE constraints)",
            use_pde=False,
            pde_weight=0.0,
            phase_field_weight=0.0,
            use_two_stage=False
        ),
        AblationConfig(
            name="R1.1 RD Only",
            description="Reaction-Diffusion PDE only (λ_RD=1e-4, λ_PF=0.0)",
            use_pde=True,
            pde_weight=1e-4,
            phase_field_weight=0.0,
            diffusion_coeff=5.0,
            reaction_threshold=0.5,
            use_two_stage=True
        ),
        AblationConfig(
            name="R1.2 Phase-Field Only",
            description="Phase-field energy only (λ_RD=0.0, λ_PF=1e-4)",
            use_pde=True,
            pde_weight=0.0,
            phase_field_weight=1e-4,
            epsilon=0.05,
            diffusion_coeff=5.0,
            reaction_threshold=0.5,
            use_two_stage=True
        ),
        AblationConfig(
            name="R1.3 RD + Phase-Field",
            description="Reaction-Diffusion + Phase-Field (λ_RD=1e-4, λ_PF=1e-4)",
            use_pde=True,
            pde_weight=1e-4,
            phase_field_weight=1e-4,
            diffusion_coeff=5.0,
            reaction_threshold=0.5,
            epsilon=0.05,
            use_two_stage=True
        )
    ]


def define_ablation_r2() -> list:
    """
    R2: Low Sample Regime Analysis
    
    Tests the full PDE-constrained model (RD + Phase-Field) with varying
    amounts of training data to evaluate sample complexity and data efficiency.
    
    Fixed Parameters (optimal):
    - Full model: RD + Phase-Field
    - λ_RD = 1e-4, λ_PF = 1e-4
    - D = 5.0, a = 0.5, ε = 0.05
    - Two-stage training
    
    Variants:
    - 10%, 25%, 50%, 75%, 100% of training data
    """
    fractions = [0.1, 0.25, 0.5, 0.75, 1.0]
    return [
        AblationConfig(
            name=f"R2.{i} {int(fraction*100)}% Data",
            description=f"Full model (RD + Phase-Field) with {int(fraction*100)}% training data",
            use_pde=True,
            pde_weight=1e-4,
            phase_field_weight=1e-4,
            diffusion_coeff=5.0,
            reaction_threshold=0.5,
            epsilon=0.05,
            train_fraction=fraction,
            use_two_stage=True
        )
        for i, fraction in enumerate(fractions)
    ]


def define_ablation_s1() -> list:
    """
    S1: Reaction Threshold Sensitivity Analysis
    
    Tests different reaction threshold (a) values in the reaction term
    f(u) = u(1-u)(u-a). The threshold controls the bistability of the
    reaction-diffusion equation and affects boundary sharpness.
    
    Fixed Parameters (optimal):
    - Full model: RD + Phase-Field
    - λ_RD = 1e-4, λ_PF = 1e-4
    - D = 5.0, ε = 0.05
    - Two-stage training
    - 10% of training data
    
    Variants:
    - a ∈ {0.3, 0.4, 0.5, 0.6, 0.7}
    - Lower a: Favors lower values, softer boundaries
    - Higher a: Favors higher values, sharper boundaries
    - a = 0.5: Standard symmetric threshold
    """
    thresholds = [0.3, 0.4, 0.5, 0.6, 0.7]
    return [
        AblationConfig(
            name=f"S1.{i} a={a:.1f}",
            description=f"Full model (RD + Phase-Field) with reaction threshold a={a}",
            use_pde=True,
            pde_weight=1e-4,
            phase_field_weight=1e-4,
            diffusion_coeff=5.0,
            reaction_threshold=a,
            epsilon=0.05,
            train_fraction=0.1,
            use_two_stage=True
        )
        for i, a in enumerate(thresholds)
    ]


def define_ablation_s2() -> list:
    """
    S2: Diffusion Coefficient Sensitivity Analysis
    
    Tests different diffusion coefficient (D) values to find optimal
    smoothing strength. All variants use reaction-diffusion PDE with
    fixed pde_weight=1e-3.
    
    Fixed Parameters:
    - Reaction-Diffusion only (no Phase-Field)
    - pde_weight = 1e-3
    - 10% of training data
    - Two-stage training
    
    Variants:
    - D ∈ {0.5, 1.0, 2.0, 5.0, 10.0, 100.0}
    """
    return [
        AblationConfig(
            name=f"S2.{i} D={d:.1f}" if d < 10 else f"S2.{i} D={d:.0f}",
            description=f"Reaction-diffusion with diffusion coefficient D={d}",
            use_pde=True,
            pde_weight=1e-3,
            diffusion_coeff=d,
            phase_field_weight=0.0,
            train_fraction=0.1,
            use_two_stage=True
        )
        for i, d in enumerate([0.5, 1.0, 2.0, 5.0, 10.0, 100.0])
    ]


def define_ablation_s3() -> list:
    """
    S3: Interface Width Sensitivity Analysis
    
    Tests different epsilon values for phase-field interface width parameter.
    Smaller epsilon = sharper interfaces, larger epsilon = smoother interfaces.
    
    Fixed Parameters:
    - Full model: RD + Phase-Field
    - λ_RD = 1e-4 (reaction-diffusion weight)
    - λ_PF = 1e-4 (phase-field weight)
    - D = 5.0 (diffusion coefficient)
    - a = 0.5 (reaction threshold)
    - 10% of training data
    - Two-stage training
    
    Variants:
    - ε ∈ {0.001, 0.01, 0.05, 0.1, 0.2}
    """
    return [
        AblationConfig(
            name=f"S3.{i} ε={eps:.3f}" if eps < 0.01 else f"S3.{i} ε={eps:.2f}",
            description=f"Reaction-diffusion + phase-field (ε={eps}, λ_RD=1e-4, λ_PF=1e-4, D=5.0)",
            use_pde=True,
            pde_weight=1e-4,
            phase_field_weight=1e-4,
            diffusion_coeff=5.0,
            reaction_threshold=0.5,
            epsilon=eps,
            train_fraction=0.1,
            use_two_stage=True
        )
        for i, eps in enumerate([0.001, 0.01, 0.05, 0.1, 0.2])
    ]


def define_ablation_r3() -> list:
    """
    R3: Influence of PDE Constraints (10% Data)
    
    Tests the contribution of each PDE constraint component with 10% of training data
    to evaluate the effectiveness of PDE constraints in low-data regimes.
    
    Tests the contribution of each PDE constraint component:
    1. Baseline: No PDE constraints (Dice + BCE only)
    2. RD Only: Reaction-Diffusion PDE only
    3. Phase-Field Only: Phase-field energy only
    4. RD + Phase-Field: Both constraints combined
    
    Fixed Parameters:
    - D = 5.0 (diffusion coefficient)
    - a = 0.5 (reaction threshold)
    - ε = 0.05 (interface width)
    - λ_RD = 1e-4 (when enabled)
    - λ_PF = 1e-4 (when enabled)
    - Two-stage training
    - train_fraction = 0.1 (10% of training data)
    """
    return [
        AblationConfig(
            name="R3.0 Baseline",
            description="Baseline UNet (Dice + BCE only, no PDE constraints) with 10% data",
            use_pde=False,
            pde_weight=0.0,
            phase_field_weight=0.0,
            train_fraction=0.1,
            use_two_stage=False
        ),
        AblationConfig(
            name="R3.1 RD Only",
            description="Reaction-Diffusion PDE only (λ_RD=1e-4, λ_PF=0.0) with 10% data",
            use_pde=True,
            pde_weight=1e-4,
            phase_field_weight=0.0,
            diffusion_coeff=5.0,
            reaction_threshold=0.5,
            train_fraction=0.1,
            use_two_stage=True
        ),
        AblationConfig(
            name="R3.2 Phase-Field Only",
            description="Phase-field energy only (λ_RD=0.0, λ_PF=1e-4) with 10% data",
            use_pde=True,
            pde_weight=0.0,
            phase_field_weight=1e-4,
            epsilon=0.05,
            diffusion_coeff=5.0,
            reaction_threshold=0.5,
            train_fraction=0.1,
            use_two_stage=True
        ),
        AblationConfig(
            name="R3.3 RD + Phase-Field",
            description="Reaction-Diffusion + Phase-Field (λ_RD=1e-4, λ_PF=1e-4) with 10% data",
            use_pde=True,
            pde_weight=1e-4,
            phase_field_weight=1e-4,
            diffusion_coeff=5.0,
            reaction_threshold=0.5,
            epsilon=0.05,
            train_fraction=0.1,
            use_two_stage=True
        )
    ]


def main():
    parser = argparse.ArgumentParser(
        description='Run ablation studies for PDE-constrained cell segmentation'
    )
    parser.add_argument(
        '--ablation',
        type=str,
        required=True,
        choices=['R1', 'R2', 'R3', 'S1', 'S2', 'S3', 'all'],
        help='Which ablation study to run. "all" runs all ablation studies.'
    )
    parser.add_argument(
        '--train-dir',
        type=str,
        default='images/training',
        help='Training images directory'
    )
    parser.add_argument(
        '--train-json',
        type=str,
        default='images/annotation/training_annotation.json',
        help='Training annotations JSON'
    )
    parser.add_argument(
        '--val-dir',
        type=str,
        default='images/validation',
        help='Validation images directory'
    )
    parser.add_argument(
        '--val-json',
        type=str,
        default='images/annotation/validation_annotation.json',
        help='Validation annotations JSON'
    )
    parser.add_argument(
        '--test-dir',
        type=str,
        default='images/testing',
        help='[DEPRECATED] Use --in-dist-test-dir and --out-dist-test-dir instead'
    )
    parser.add_argument(
        '--test-json',
        type=str,
        default='images/annotation/testing_annotation.json',
        help='[DEPRECATED] Use --in-dist-test-json and --out-dist-test-json instead'
    )
    parser.add_argument(
        '--in-dist-test-dir',
        type=str,
        default='images/in_dist_testing',
        help='In-distribution test images directory (default: images/in_dist_testing)'
    )
    parser.add_argument(
        '--in-dist-test-json',
        type=str,
        default='images/annotation/in_dist_testing_annotation.json',
        help='In-distribution test annotations JSON (default: images/annotation/in_dist_testing_annotation.json)'
    )
    parser.add_argument(
        '--out-dist-test-dir',
        type=str,
        default='images/out_dist_testing',
        help='Out-of-distribution test images directory (default: images/out_dist_testing)'
    )
    parser.add_argument(
        '--out-dist-test-json',
        type=str,
        default='images/annotation/out_dist_testing_annotation.json',
        help='Out-of-distribution test annotations JSON (default: images/annotation/out_dist_testing_annotation.json)'
    )
    parser.add_argument(
        '--batch-size',
        type=int,
        default=8,
        help='Batch size for training (default: 8)'
    )
    parser.add_argument(
        '--learning-rate',
        type=float,
        default=1e-4,
        help='Learning rate (default: 1e-4)'
    )
    parser.add_argument(
        '--stage1-epochs',
        type=int,
        default=50,
        help='Max epochs for stage 1 (default: 50)'
    )
    parser.add_argument(
        '--stage2-epochs',
        type=int,
        default=50,
        help='Max epochs for stage 2 (default: 50)'
    )
    parser.add_argument(
        '--early-stopping-patience',
        type=int,
        default=10,
        help='Early stopping patience (default: 10)'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default=None,
        help='[DEPRECATED] Not used anymore. All files are saved in output/ablation/{ablation_name}_{timestamp}/'
    )
    
    args = parser.parse_args()
    
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Convert paths and resolve to absolute paths
    train_dir = Path(args.train_dir).resolve()
    train_json = Path(args.train_json).resolve()
    val_dir = Path(args.val_dir).resolve()
    val_json = Path(args.val_json).resolve()
    
    # In-distribution and out-of-distribution test sets
    in_dist_test_dir = Path(args.in_dist_test_dir).resolve()
    in_dist_test_json = Path(args.in_dist_test_json).resolve()
    out_dist_test_dir = Path(args.out_dist_test_dir).resolve()
    out_dist_test_json = Path(args.out_dist_test_json).resolve()
    
    # Legacy support: if old test_dir/test_json are provided, use them as in-dist
    if args.test_dir != 'images/testing' or args.test_json != 'images/annotation/testing_annotation.json':
        print("Warning: --test-dir and --test-json are deprecated. Using them as in-distribution test set.")
        in_dist_test_dir = Path(args.test_dir).resolve()
        in_dist_test_json = Path(args.test_json).resolve()
    
    output_dir = Path(args.output_dir).resolve() if args.output_dir is not None else None
    
    # Define ablation studies
    ablation_studies = {
        'R1': define_ablation_r1(),  # Influence of PDE Constraints (100% data)
        'R2': define_ablation_r2(),  # Low Sample Regime Analysis
        'R3': define_ablation_r3(),  # Influence of PDE Constraints (10% data)
        'S1': define_ablation_s1(),  # Reaction Threshold Sensitivity Analysis
        'S2': define_ablation_s2(),  # Diffusion Coefficient Sensitivity Analysis
        'S3': define_ablation_s3()   # Interface Width Sensitivity Analysis
    }
    
    # Run selected ablation(s)
    if args.ablation == 'all':
        studies_to_run = ['R1', 'R2', 'R3', 'S1', 'S2', 'S3']
    else:
        studies_to_run = [args.ablation]
    
    for ablation_name in studies_to_run:
        if ablation_name not in ablation_studies:
            print(f"Warning: Ablation {ablation_name} not defined, skipping...")
            continue
        
        variants = ablation_studies[ablation_name]
        
        print(f"\n{'='*70}")
        print(f"Starting Ablation Study: {ablation_name}")
        print(f"{'='*70}")
        
        results = run_ablation_study(
            ablation_name=ablation_name,
            variants=variants,
            train_dir=train_dir,
            train_json=train_json,
            val_dir=val_dir,
            val_json=val_json,
            in_dist_test_dir=in_dist_test_dir,
            in_dist_test_json=in_dist_test_json,
            out_dist_test_dir=out_dist_test_dir,
            out_dist_test_json=out_dist_test_json,
            device=device,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            stage1_epochs=args.stage1_epochs,
            stage2_epochs=args.stage2_epochs,
            early_stopping_patience=args.early_stopping_patience,
            output_dir=output_dir
        )
        
        print(f"\nAblation {ablation_name} complete!")
        print(f"Results: {results['results_json']}")
        print(f"Summary: {results['summary_csv']}")
    
    print("\n" + "=" * 70)
    print("ALL ABLATION STUDIES COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()


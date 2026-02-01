from src.train import train
import argparse


def main():
    parser = argparse.ArgumentParser(
        description='Train PDE-constrained cell segmentation model'
    )
    parser.add_argument(
        '--single-stage',
        action='store_true',
        help='Use single-stage training (PDE from start) instead of two-stage'
    )
    parser.add_argument(
        '--pde-weight',
        type=float,
        default=1e-4,
        help='Weight for PDE regularization λ_RD (default: 1e-4, optimal)'
    )
    parser.add_argument(
        '--diffusion-coeff',
        type=float,
        default=5.0,
        help='Diffusion coefficient D for PDE (default: 5.0, optimal)'
    )
    parser.add_argument(
        '--reaction-threshold',
        type=float,
        default=0.5,
        help='Reaction term threshold a for PDE (default: 0.5, optimal)'
    )
    parser.add_argument(
        '--phase-field-weight',
        type=float,
        default=1e-4,
        help='Weight for phase-field energy λ_PF (default: 1e-4, optimal)'
    )
    parser.add_argument(
        '--epsilon',
        type=float,
        default=0.05,
        help='Interface width parameter ε for phase-field energy (default: 0.05, optimal)'
    )
    parser.add_argument(
        '--batch-size',
        type=int,
        default=8,
        help='Batch size for training (default: 8, recommended: 8-16)'
    )
    parser.add_argument(
        '--learning-rate',
        type=float,
        default=1e-4,
        help='Learning rate for AdamW optimizer (default: 1e-4)'
    )
    parser.add_argument(
        '--stage1-epochs',
        type=int,
        default=50,
        help='Maximum epochs for Stage I (baseline training) (default: 50)'
    )
    parser.add_argument(
        '--stage2-epochs',
        type=int,
        default=50,
        help='Maximum epochs for Stage II (PDE fine-tuning) (default: 50)'
    )
    parser.add_argument(
        '--early-stopping-patience',
        type=int,
        default=5,
        help='Patience for early stopping (default: 10)'
    )
    parser.add_argument(
        '--train-fraction',
        type=float,
        default=None,
        help='Fraction of training data to use (e.g., 0.1 for 10%%, 0.25 for 25%%)'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed for reproducibility (default: 42)'
    )
    args = parser.parse_args()
    
    train(
        use_two_stage=not args.single_stage,
        pde_weight=args.pde_weight,
        diffusion_coeff=args.diffusion_coeff,
        reaction_threshold=args.reaction_threshold,
        phase_field_weight=args.phase_field_weight,
        epsilon=args.epsilon,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        stage1_epochs=args.stage1_epochs,
        stage2_epochs=args.stage2_epochs,
        early_stopping_patience=args.early_stopping_patience,
        train_fraction=args.train_fraction,
        seed=args.seed
    )


if __name__ == "__main__":
    main()
import torch
import torch.nn as nn

from .pde import PDERegularization


class DiceBCELoss(nn.Module):
    """
    Combined Dice Loss + Binary Cross-Entropy Loss.
    
    This loss function is used for training the baseline UNet.
    It does NOT include any PDE terms, curvature penalties, or 
    phase-field terms.
    
    Dice Loss: Measures overlap between prediction and ground truth
    BCE Loss: Standard binary classification loss
    
    Args:
        dice_weight: Weight for Dice loss component (default: 0.5)
        bce_weight: Weight for BCE loss component (default: 0.5)
        smooth: Smoothing factor for Dice loss (default: 1e-6)
    """
    
    def __init__(
        self,
        dice_weight: float = 0.5,
        bce_weight: float = 0.5,
        smooth: float = 1e-6
    ):
        super(DiceBCELoss, self).__init__()
        self.dice_weight = dice_weight
        self.bce_weight = bce_weight
        self.smooth = smooth
        self.bce = nn.BCELoss()
    
    def forward(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute combined Dice + BCE loss.
        
        Args:
            predictions: Model output tensor (B, 1, H, W) with values in (0,1)
            targets: Ground truth binary mask (B, 1, H, W) with values in {0,1}
            
        Returns:
            Combined loss value
        """
        # Flatten tensors for Dice calculation
        predictions_flat = predictions.view(-1)
        targets_flat = targets.view(-1)
        
        # Dice Loss
        intersection = (predictions_flat * targets_flat).sum()
        dice = (2.0 * intersection + self.smooth) / (
            predictions_flat.sum() + targets_flat.sum() + self.smooth
        )
        dice_loss = 1 - dice
        
        # Binary Cross-Entropy Loss
        bce_loss = self.bce(predictions, targets)
        
        # Combined loss
        total_loss = self.dice_weight * dice_loss + self.bce_weight * bce_loss
        
        return total_loss


class DiceBCEPDELoss(nn.Module):
    """
    Combined Dice Loss + Binary Cross-Entropy Loss + PDE Regularization.
    
    Args:
        dice_weight: Weight for Dice loss component (default: 0.5)
        bce_weight: Weight for BCE loss component (default: 0.5)
        pde_weight: Weight for reaction-diffusion PDE regularization λ_RD (default: 1e-3)
        phase_field_weight: Weight for phase-field energy λ_PF (default: 0.0)
        smooth: Smoothing factor for Dice loss (default: 1e-6)
        diffusion_coeff: Diffusion coefficient D > 0 for PDE (default: 1.0)
        reaction_threshold: Reaction term threshold a ∈ (0,1) for PDE (default: 0.5)
        epsilon: Interface width parameter for phase-field energy (default: 0.05)
    """
    
    def __init__(
        self,
        dice_weight: float = 0.5,
        bce_weight: float = 0.5,
        pde_weight: float = 1e-3,
        phase_field_weight: float = 0.0,
        smooth: float = 1e-6,
        diffusion_coeff: float = 1.0,
        reaction_threshold: float = 0.5,
        epsilon: float = 0.05
    ):
        super(DiceBCEPDELoss, self).__init__()
        self.dice_weight = dice_weight
        self.bce_weight = bce_weight
        self.pde_weight = pde_weight
        self.phase_field_weight = phase_field_weight
        self.smooth = smooth
        self.epsilon = epsilon
        
        # Initialize PDE regularization module
        self.pde_regularization = PDERegularization(
            diffusion_coeff=diffusion_coeff,
            reaction_threshold=reaction_threshold
        )
        
        # Binary cross-entropy loss
        self.bce = nn.BCELoss()
    
    def forward(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute combined Dice + BCE + PDE losses.
        
        Args:
            predictions: Model output tensor (B, 1, H, W) with values in (0,1)
            targets: Ground truth binary mask (B, 1, H, W) with values in {0,1}
            
        Returns:
            Combined loss value
        """
        # Flatten tensors for Dice calculation
        predictions_flat = predictions.view(-1)
        targets_flat = targets.view(-1)
        
        # Dice Loss
        intersection = (predictions_flat * targets_flat).sum()
        dice = (2.0 * intersection + self.smooth) / (
            predictions_flat.sum() + targets_flat.sum() + self.smooth
        )
        dice_loss = 1 - dice
        
        # Binary Cross-Entropy Loss
        bce_loss = self.bce(predictions, targets)
        
        # Initialize total loss with data fidelity terms
        total_loss = (
            self.dice_weight * dice_loss +
            self.bce_weight * bce_loss
        )
        
        # Reaction-Diffusion PDE Regularization Loss
        if self.pde_weight > 0:
            pde_loss = self.pde_regularization.compute_loss(predictions)
            total_loss = total_loss + self.pde_weight * pde_loss
        
        # Phase-Field Interface Energy Loss
        if self.phase_field_weight > 0:
            phase_field_loss = self.pde_regularization.compute_phase_field_loss(
                predictions, 
                epsilon=self.epsilon
            )
            total_loss = total_loss + self.phase_field_weight * phase_field_loss
        
        return total_loss
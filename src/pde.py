import torch
import torch.nn as nn
import torch.nn.functional as F


class PDERegularization(nn.Module):    
    def __init__(
        self,
        diffusion_coeff: float = 1.0,
        reaction_threshold: float = 0.5
    ):
        super(PDERegularization, self).__init__()
        
        if diffusion_coeff <= 0:
            raise ValueError("diffusion_coeff must be positive")
        if not (0 < reaction_threshold < 1):
            raise ValueError("reaction_threshold must be in (0,1)")
        
        self.diffusion_coeff = diffusion_coeff
        self.reaction_threshold = reaction_threshold
        
        # Create Laplacian kernel for 5-point stencil
        # ∇²u ≈ u_{i+1,j} + u_{i-1,j} + u_{i,j+1} + u_{i,j-1} - 4u_{i,j}
        laplacian_kernel = torch.tensor([
            [0.0, 1.0, 0.0],
            [1.0, -4.0, 1.0],
            [0.0, 1.0, 0.0]
        ], dtype=torch.float32)
        
        # Create gradient kernels for phase-field energy
        # Central differences: ∂u/∂x ≈ (u[i,j+1] - u[i,j-1]) / 2
        grad_x_kernel = torch.tensor([
            [0.0, 0.0, 0.0],
            [-0.5, 0.0, 0.5],
            [0.0, 0.0, 0.0]
        ], dtype=torch.float32)
        
        grad_y_kernel = torch.tensor([
            [0.0, -0.5, 0.0],
            [0.0, 0.0, 0.0],
            [0.0, 0.5, 0.0]
        ], dtype=torch.float32)
        
        # Register as buffers (not parameters, but part of model state)
        self.register_buffer('laplacian_kernel', laplacian_kernel.unsqueeze(0).unsqueeze(0))
        self.register_buffer('grad_x_kernel', grad_x_kernel.unsqueeze(0).unsqueeze(0))
        self.register_buffer('grad_y_kernel', grad_y_kernel.unsqueeze(0).unsqueeze(0))
    
    def compute_laplacian(
        self,
        u: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute Laplacian using 5-point stencil with Neumann boundary conditions.
        
        Neumann boundary conditions are enforced via symmetric (mirror) padding,
        which prevents artificial flux across image boundaries.
        
        Args:
            u: Input tensor of shape (B, 1, H, W) with values in (0,1)
            
        Returns:
            Laplacian tensor of shape (B, 1, H, W)
        """
        # Apply symmetric padding for Neumann boundary conditions
        # mode='reflect' implements mirror padding: [a, b, c] -> [b, a, b, c, c, b]
        u_padded = F.pad(u, (1, 1, 1, 1), mode='reflect')
        
        # Apply Laplacian kernel via convolution
        # The kernel is (1, 1, 3, 3), so we need to expand u_padded to (B, 1, H+2, W+2)
        # Ensure kernel is on the same device as input
        laplacian_kernel = self.laplacian_kernel.to(u.device)
        laplacian = F.conv2d(
            u_padded,
            laplacian_kernel,
            padding=0  # Already padded
        )
        
        return laplacian
    
    def reaction_term(
        self,
        u: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute nonlinear reaction term f(u) = u(1-u)(u-a).
        
        This form:
        - Encourages bistability near 0 and 1
        - Suppresses intermediate noise
        - Is smooth and fully differentiable
        
        Args:
            u: Input tensor of shape (B, 1, H, W) with values in (0,1)
            
        Returns:
            Reaction term tensor of shape (B, 1, H, W)
        """
        return u * (1.0 - u) * (u - self.reaction_threshold)
    
    def compute_residual(
        self,
        u: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute PDE residual r(x) = D∇²u + f(u).
        
        This residual measures the local deviation of the predicted segmentation
        field from the PDE equilibrium.
        
        Args:
            u: Input tensor of shape (B, 1, H, W) with values in (0,1)
            
        Returns:
            Residual tensor of shape (B, 1, H, W)
        """
        laplacian = self.compute_laplacian(u)
        reaction = self.reaction_term(u)
        
        residual = self.diffusion_coeff * laplacian + reaction
        
        return residual
    
    def compute_loss(
        self,
        u: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute L2 PDE residual penalty.
        This yields smooth gradients and admits a statistical interpretation
        under a Gaussian noise assumption on the residual.
        
        Args:
            u: Input tensor of shape (B, 1, H, W) with values in (0,1)
            
        Returns:
            Scalar loss value
        """
        residual = self.compute_residual(u)
        
        # Compute mean squared residual over all spatial locations
        # |Ω| = B * H * W (total number of spatial locations)
        pde_loss = torch.mean(residual ** 2)
        
        return pde_loss
    
    def compute_gradient_magnitude(
        self,
        u: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute gradient magnitude squared |∇u|² using finite differences.
        
        Uses first-order finite differences with Neumann boundary conditions
        (symmetric padding) to approximate spatial gradients.
        
        Args:
            u: Input tensor of shape (B, 1, H, W) with values in (0,1)
            
        Returns:
            Gradient magnitude squared tensor of shape (B, 1, H, W)
        """
        # Apply symmetric padding for Neumann boundary conditions
        u_padded = F.pad(u, (1, 1, 1, 1), mode='reflect')
        
        # Compute gradients via convolution using central differences
        # ∂u/∂x ≈ (u[i,j+1] - u[i,j-1]) / 2
        # ∂u/∂y ≈ (u[i+1,j] - u[i-1,j]) / 2
        # Ensure kernels are on the same device as input
        grad_x_kernel = self.grad_x_kernel.to(u.device)
        grad_y_kernel = self.grad_y_kernel.to(u.device)
        grad_x = F.conv2d(u_padded, grad_x_kernel, padding=0)
        grad_y = F.conv2d(u_padded, grad_y_kernel, padding=0)
        
        # Gradient magnitude squared: |∇u|² = (∂u/∂x)² + (∂u/∂y)²
        grad_mag_sq = grad_x ** 2 + grad_y ** 2
        
        return grad_mag_sq
    
    def compute_phase_field_loss(
        self,
        u: torch.Tensor,
        epsilon: float = 0.05
    ) -> torch.Tensor:
        """
        Compute phase-field interface energy loss.
        
        This enforces sharp yet smooth interfaces via variational energy
        minimization. The gradient term penalizes interface width, while
        the double-well potential enforces bistability at 0 and 1.
        
        Args:
            u: Input tensor of shape (B, 1, H, W) with values in (0,1)
            epsilon: Interface width parameter (default: 0.05)
            
        Returns:
            Scalar loss value
        """
        if epsilon <= 0:
            raise ValueError("epsilon must be positive")
        
        # Gradient energy term: (ε/2)|∇u|²
        grad_mag_sq = self.compute_gradient_magnitude(u)
        gradient_energy = (epsilon / 2.0) * grad_mag_sq
        
        # Double-well potential term: (1/ε)u²(1-u)²
        double_well = (1.0 / epsilon) * (u ** 2) * ((1.0 - u) ** 2)
        
        # Combine terms and compute spatial mean
        phase_field_loss = torch.mean(gradient_energy + double_well)
        
        return phase_field_loss


def create_pde_regularization(
    diffusion_coeff: float = 1.0,
    reaction_threshold: float = 0.5
) -> PDERegularization:
    """
    Factory function to create a PDE regularization module.
    
    Args:
        diffusion_coeff: Diffusion coefficient D > 0 (default: 1.0)
        reaction_threshold: Reaction term threshold a ∈ (0,1) (default: 0.5)
        
    Returns:
        PDERegularization instance
    """
    return PDERegularization(
        diffusion_coeff=diffusion_coeff,
        reaction_threshold=reaction_threshold
    )


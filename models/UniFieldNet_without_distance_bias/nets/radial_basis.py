import torch
import torch.nn as nn
import numpy as np
import math

class RadialBasis(nn.Module):
    """
    Radial Basis Function (RBF) specifically for Spherical Bessel Basis.
    Extracted from GemNet/OCP for standalone use.
    """
    def __init__(self, num_radial, cutoff, rbf={'name': 'spherical_bessel'}):
        super().__init__()
        self.num_radial = num_radial
        self.cutoff = cutoff
        self.rbf_name = rbf.get('name', 'spherical_bessel')

        if self.rbf_name == 'spherical_bessel':
            # Initialize frequencies at canonical roots of spherical Bessel functions
            # roots of j_0(x) are n*pi
            self.inv_cutoff = 1 / cutoff
            self.prefactor = math.sqrt(2 / (cutoff**3))
            
            # The frequencies are n * pi
            # We register them as a buffer so they are saved with the model but not updated by optimizer
            self.register_buffer(
                "frequencies",
                math.pi * torch.arange(1, num_radial + 1, dtype=torch.float32)
            )

    def forward(self, d):
        """
        Args:
            d: Tensor containing distances of shape [..., N]
        Returns:
            rbf: Tensor of shape [..., N, num_radial]
        """
        # d shape: [num_edges]
        # output shape: [num_edges, num_radial]
        
        if self.rbf_name == 'spherical_bessel':
            d_scaled = d * self.inv_cutoff
            
            # d_scaled unsqueezed to [num_edges, 1]
            # frequencies unsqueezed to [1, num_radial]
            # Result is [num_edges, num_radial]
            d_scaled = d_scaled.unsqueeze(-1)
            freq = self.frequencies.unsqueeze(0)
            
            # Calculate spherical bessel functions: j_0(x) = sin(x) / x
            # Here arguments are n * pi * d / cutoff
            arg = d_scaled * freq
            
            # Handle numerical instability at d=0 (though d usually > 0 in graphs)
            # sin(0)/0 -> 1, but we can just compute strictly.
            # In molecular graphs, d is typically > 0.
            
            rbf = self.prefactor * torch.sin(arg) / d_scaled
            
            return rbf
        else:
            raise ValueError(f"Unknown RBF name: {self.rbf_name}")
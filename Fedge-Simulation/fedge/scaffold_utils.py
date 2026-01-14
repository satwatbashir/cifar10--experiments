"""
SCAFFOLD (Stochastic Controlled Averaging for Federated Learning) Implementation

This module implements SCAFFOLD control variates to correct client and server drift
in non-IID federated learning scenarios, addressing the fundamental cause of 
performance collapse after round 30+.

Reference: Karimireddy et al. "SCAFFOLD: Stochastic Controlled Averaging for Federated Learning"
"""

import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple
import copy
import logging

logger = logging.getLogger(__name__)


class SCAFFOLDControlVariates:
    """
    Manages SCAFFOLD control variates for drift correction in federated learning.
    
    Key Concepts:
    - c_i: Client control variate (captures local drift)
    - c_server: Server control variate (captures global drift)  
    - Corrected gradient = local_grad - c_i + c_server
    """
    
    def __init__(self, model: nn.Module):
        """Initialize control variates to zero for all model parameters."""
        self.client_control = self._init_control_variate(model)
        self.server_control = self._init_control_variate(model)
        
    def _init_control_variate(self, model: nn.Module) -> Dict[str, torch.Tensor]:
        """Initialize control variate as zero tensors matching model parameters."""
        control = {}
        for name, param in model.named_parameters():
            if param.requires_grad:
                control[name] = torch.zeros_like(param.data)
        return control
        
    def get_client_control(self) -> Dict[str, torch.Tensor]:
        """Get client control variate c_i."""
        return self.client_control
        
    def get_server_control(self) -> Dict[str, torch.Tensor]:
        """Get server control variate c_server."""
        return self.server_control
        
    def update_client_control(self,
                            local_model: nn.Module,
                            global_model: nn.Module,
                            learning_rate: float,
                            local_epochs: int,
                            clip_value: float = 1.0,
                            scaling_factor: float = 0.1) -> None:
        """
        Update client control variate after local training.

        v9 Fix: Use scaling factor instead of division by (K*lr) to prevent
        amplification issues that caused collapse in v7/v8.

        Original formula: c_i^{new} = c_i - c_server + (1/K*lr) * (global - local)
        v9 formula: c_i^{new} = c_i - c_server + scaling_factor * (global - local)

        Args:
            local_model: Model after local training
            global_model: Global model before local training
            learning_rate: Learning rate used in training
            local_epochs: Number of local epochs
            clip_value: Clipping bound for control variates (v9: tightened to 1.0)
            scaling_factor: Scaling factor for model diff (v9: replaces 1/(K*lr))
        """
        with torch.no_grad():
            for name, local_param in local_model.named_parameters():
                if name in self.client_control:
                    global_param = dict(global_model.named_parameters())[name]

                    # Compute model difference
                    model_diff = global_param.data - local_param.data

                    # v9 fix: Use scaling factor instead of division by (K*lr)
                    # Old: model_diff / (local_epochs * learning_rate) = 20x amplification
                    # New: scaling_factor * model_diff = controlled magnitude
                    new_control = (
                        self.client_control[name]
                        - self.server_control[name]
                        + scaling_factor * model_diff
                    )

                    # v9 fix: Tighter clipping (1.0 instead of 10.0)
                    self.client_control[name] = torch.clamp(new_control, min=-clip_value, max=clip_value)

        logger.debug(f"Updated client control variate for {len(self.client_control)} parameters")
        
    def update_server_control(self, 
                            client_controls: List[Dict[str, torch.Tensor]],
                            client_weights: List[float]) -> None:
        """
        Update server control variate by aggregating client controls.
        
        Formula: c_server^{new} = sum(w_i * c_i) where w_i are client weights
        """
        if not client_controls:
            return
            
        # Weighted average of client control variates
        with torch.no_grad():
            for param_name in self.server_control:
                weighted_sum = torch.zeros_like(self.server_control[param_name])
                
                for client_control, weight in zip(client_controls, client_weights):
                    if param_name in client_control:
                        weighted_sum += weight * client_control[param_name]
                        
                self.server_control[param_name] = weighted_sum
                
        logger.debug(f"Updated server control variate from {len(client_controls)} clients")
        
    def apply_scaffold_correction(self,
                                model: nn.Module,
                                learning_rate: float,
                                current_round: int = 0,
                                warmup_rounds: int = 10,
                                correction_clip: float = 0.1) -> None:
        """
        Apply SCAFFOLD correction to model gradients during training.

        v9 Fixes:
        1. Clip corrections before applying (prevent gradient explosion)
        2. Warmup scaling (gradual activation over warmup_rounds)

        This should be called during the training loop to correct gradients:
        corrected_grad = original_grad - c_i + c_server

        Args:
            model: The model being trained
            learning_rate: Current learning rate
            current_round: Current training round (for warmup)
            warmup_rounds: Number of rounds for gradual SCAFFOLD activation
            correction_clip: Max magnitude of corrections (v9: tight bound)
        """
        # v9 fix: Gradual warmup to prevent sudden activation collapse
        warmup_factor = min(1.0, current_round / max(1, warmup_rounds))

        with torch.no_grad():
            for name, param in model.named_parameters():
                if param.grad is not None and name in self.client_control:
                    # Apply correction: grad = grad - c_i + c_server
                    raw_correction = -self.client_control[name] + self.server_control[name]

                    # v9 fix: Clip correction magnitude (prevents gradient explosion)
                    clipped_correction = torch.clamp(raw_correction, min=-correction_clip, max=correction_clip)

                    # v9 fix: Apply warmup scaling
                    scaled_correction = warmup_factor * clipped_correction

                    param.grad.data += scaled_correction

        logger.debug(f"Applied SCAFFOLD gradient correction (warmup={warmup_factor:.2f})")


def create_scaffold_manager(model: nn.Module) -> SCAFFOLDControlVariates:
    """Factory function to create SCAFFOLD control variate manager."""
    return SCAFFOLDControlVariates(model)


def aggregate_scaffold_controls(client_controls: List[Dict[str, torch.Tensor]], 
                              client_weights: List[float]) -> Dict[str, torch.Tensor]:
    """
    Aggregate client control variates into server control variate.
    
    Args:
        client_controls: List of client control variates
        client_weights: Weights for aggregation (typically based on data size)
        
    Returns:
        Aggregated server control variate
    """
    if not client_controls:
        return {}
        
    # Initialize with zeros
    server_control = {}
    for param_name in client_controls[0]:
        server_control[param_name] = torch.zeros_like(client_controls[0][param_name])
        
    # Weighted aggregation
    total_weight = sum(client_weights)
    for client_control, weight in zip(client_controls, client_weights):
        normalized_weight = weight / total_weight
        for param_name in server_control:
            if param_name in client_control:
                server_control[param_name] += normalized_weight * client_control[param_name]
                
    return server_control


def scaffold_enabled_from_config(config: Dict) -> bool:
    """Check if SCAFFOLD is enabled in configuration."""
    return config.get("scaffold_enabled", False)

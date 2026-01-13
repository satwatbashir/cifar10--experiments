"""Server-level SCAFFOLD for cross-server knowledge sharing.

This module implements server-level control variates to correct for server drift
in hierarchical federated learning with non-IID data.

Key Concepts:
- c_server_i: Server i's control variate (captures server drift toward global)
- c_global: Global control variate (weighted avg of all c_server_i)
- Correction: theta_corrected = theta_cluster - lr * (c_server - c_global)

This enables knowledge sharing across isolated servers through gradient direction
information, not model averaging.
"""
from __future__ import annotations

import numpy as np
from typing import Dict, List, Optional
import logging

logger = logging.getLogger(__name__)


class ServerSCAFFOLD:
    """Manages server-level control variates for cross-server drift correction.

    This class implements SCAFFOLD Option II at the server level, enabling
    knowledge sharing between servers while allowing each server to maintain
    its own specialized model.
    """

    def __init__(self, num_servers: int = 3):
        """Initialize server-level SCAFFOLD.

        Args:
            num_servers: Number of servers in the hierarchy
        """
        self.num_servers = num_servers
        self.c_server: Dict[int, List[np.ndarray]] = {}
        self.c_global: Optional[List[np.ndarray]] = None
        self.server_samples: Dict[int, int] = {}

    def update_server_control(
        self,
        server_id: int,
        theta_server: List[np.ndarray],
        theta_cluster: List[np.ndarray],
        n_samples: int,
        K: int = 1,
        eta: float = 1.0,
        clip_value: float = 10.0
    ) -> None:
        """Update c_server_i after server sends model to cloud.

        SCAFFOLD Option II formula:
        c_server_i_new = c_server_i_old - c_global + (1/(K*eta)) * (theta_cluster - theta_server_i)

        Args:
            server_id: Server ID
            theta_server: Server's aggregated model weights
            theta_cluster: Cluster model from cloud
            n_samples: Number of samples this server trained on
            K: Number of local rounds (server_rounds_per_global)
            eta: Learning rate for control variate update
            clip_value: Clipping bound for control variates
        """
        # Initialize if first time
        if server_id not in self.c_server:
            self.c_server[server_id] = [np.zeros_like(w) for w in theta_server]
            logger.debug(f"Initialized c_server_{server_id}")

        c_old = self.c_server[server_id]
        c_global = self.c_global or [np.zeros_like(w) for w in theta_server]

        # SCAFFOLD Option II update with clipping
        c_new = []
        for c_o, c_g, w_cluster, w_server in zip(c_old, c_global, theta_cluster, theta_server):
            # c_server_new = c_server_old - c_global + (1/(K*eta)) * (theta_cluster - theta_server)
            raw = c_o - c_g + (1.0 / (K * eta)) * (w_cluster - w_server)

            # Clip to prevent explosion
            clipped = np.clip(raw, -clip_value, clip_value)
            c_new.append(clipped)

        self.c_server[server_id] = c_new
        self.server_samples[server_id] = n_samples

        # Log control variate statistics
        c_norm = np.sqrt(sum(np.sum(c**2) for c in c_new))
        logger.debug(f"Updated c_server_{server_id}, norm={c_norm:.4f}, samples={n_samples}")

    def update_global_control(self) -> None:
        """Update c_global as weighted average of c_server_i.

        c_global = sum((n_samples[i] / total_samples) * c_server[i] for i in servers)
        """
        if not self.c_server:
            logger.warning("No c_server values to aggregate")
            return

        total_samples = sum(self.server_samples.values())
        if total_samples == 0:
            logger.warning("Total samples is 0, skipping c_global update")
            return

        first_server = next(iter(self.c_server.values()))
        n_layers = len(first_server)

        # Initialize c_global to zeros
        self.c_global = [np.zeros_like(first_server[i]) for i in range(n_layers)]

        # Weighted sum
        for sid, c_s in self.c_server.items():
            weight = self.server_samples[sid] / total_samples
            for i in range(n_layers):
                self.c_global[i] += weight * c_s[i]

        # Log global control variate statistics
        c_global_norm = np.sqrt(sum(np.sum(c**2) for c in self.c_global))
        logger.info(f"Updated c_global from {len(self.c_server)} servers, norm={c_global_norm:.4f}")

    def apply_correction(
        self,
        server_id: int,
        theta_cluster: List[np.ndarray],
        correction_lr: float = 0.1
    ) -> List[np.ndarray]:
        """Apply SCAFFOLD correction to cluster model for server_id.

        Correction formula:
        theta_corrected = theta_cluster - correction_lr * (c_server - c_global)

        This pulls the server's model toward the global optimum while preserving
        its specialization.

        Args:
            server_id: Server ID to get corrected model for
            theta_cluster: Cluster model weights
            correction_lr: Learning rate for correction

        Returns:
            Corrected model weights
        """
        if server_id not in self.c_server or self.c_global is None:
            logger.debug(f"No correction for server {server_id} (not initialized)")
            return theta_cluster

        c_server = self.c_server[server_id]

        theta_corrected = []
        correction_magnitude = 0.0

        for w_c, c_s, c_g in zip(theta_cluster, c_server, self.c_global):
            # Correction: theta - lr * (c_server - c_global)
            correction = correction_lr * (c_s - c_g)
            corrected = w_c - correction
            theta_corrected.append(corrected)

            correction_magnitude += np.sum(correction**2)

        correction_magnitude = np.sqrt(correction_magnitude)
        logger.debug(f"Applied correction to server {server_id}, magnitude={correction_magnitude:.4f}")

        return theta_corrected

    def get_server_divergence(self) -> Dict[int, float]:
        """Get divergence of each server from global control variate.

        Returns:
            Dict mapping server_id to divergence (L2 norm of c_server - c_global)
        """
        if self.c_global is None:
            return {}

        divergences = {}
        for sid, c_s in self.c_server.items():
            diff_norm = np.sqrt(sum(
                np.sum((c_s[i] - self.c_global[i])**2)
                for i in range(len(c_s))
            ))
            divergences[sid] = float(diff_norm)

        return divergences

    def reset(self) -> None:
        """Reset all control variates (for testing or new experiments)."""
        self.c_server.clear()
        self.c_global = None
        self.server_samples.clear()
        logger.info("Reset ServerSCAFFOLD control variates")

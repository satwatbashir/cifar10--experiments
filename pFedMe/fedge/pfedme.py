import numpy as np
import torch
from typing import Any, Dict, List, Tuple, Optional, Union

from flwr.common import (
    FitIns, FitRes, NDArrays, Parameters, 
    ndarrays_to_parameters, parameters_to_ndarrays,
    Scalar, EvaluateIns, EvaluateRes
)
from flwr.server.strategy import FedAvg
from flwr.server.client_manager import ClientManager
from flwr.server.client_proxy import ClientProxy

class pFedMe(FedAvg):
    """
    Authentic pFedMe strategy implementing bi-level optimization.

    Algorithm:
    1. Clients receive global model w.
    2. Clients perform K inner steps optimizing θ with Moreau envelope.
    3. Clients perform R outer steps updating w toward θ.
    4. Server aggregates w with optional β-mixing.
    """

    def __init__(
        self,
        lamda: float = 15.0,
        inner_steps: int = 5,
        outer_steps: int = 1,
        inner_lr: float = 0.01,
        outer_lr: float = 0.01,
        beta: float = 1.0,  # Server mixing parameter
        **kwargs: Any,
    ):
        super().__init__(**kwargs)
        self.lamda = lamda
        self.inner_steps = inner_steps
        self.outer_steps = outer_steps
        self.inner_lr = inner_lr
        self.outer_lr = outer_lr
        self.beta = beta
        
        # Store previous global model for β-mixing
        self.previous_global_parameters: Optional[NDArrays] = None

    def configure_fit(
        self,
        server_round: int,
        parameters: Parameters,
        client_manager: ClientManager,
    ) -> List[Tuple[ClientProxy, FitIns]]:
        # Cache previous global parameters for β-mixing (except round 1)
        if server_round > 1 and parameters is not None:
            self.previous_global_parameters = parameters_to_ndarrays(parameters)
        
        # Use FedAvg to select clients and get FitIns
        fit_ins = super().configure_fit(server_round, parameters, client_manager)
        customized: List[Tuple[ClientProxy, FitIns]] = []
        for client, ins in fit_ins:
            # Inject pFedMe hyperparameters into config
            config = {
                **ins.config, 
                "lamda": self.lamda,
                "inner_steps": self.inner_steps,
                "outer_steps": self.outer_steps,
                "inner_lr": self.inner_lr,
                "outer_lr": self.outer_lr,
            }
            customized.append((client, FitIns(parameters=ins.parameters, config=config)))
        return customized

    def configure_evaluate(
        self,
        server_round: int,
        parameters: Parameters,
        client_manager: ClientManager,
    ) -> List[Tuple[ClientProxy, EvaluateIns]]:
        """Configure evaluation with support for both global and personalized evaluation."""
        # Get standard evaluation configuration from parent
        eval_ins = super().configure_evaluate(server_round, parameters, client_manager)
        
        # Create two evaluation passes: global and personalized
        customized: List[Tuple[ClientProxy, EvaluateIns]] = []
        
        for client, ins in eval_ins:
            # First pass: Global model evaluation (standard)
            global_config = {**ins.config, "personalized_eval": False}
            customized.append((client, EvaluateIns(parameters=ins.parameters, config=global_config)))
            
            # Second pass: Personalized model evaluation (if theta exists)
            personalized_config = {**ins.config, "personalized_eval": True}
            customized.append((client, EvaluateIns(parameters=ins.parameters, config=personalized_config)))
        
        return customized
    
    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, FitRes]],
        failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]],
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        """Aggregate client updates with pFedMe-specific β-mixing and invoke metrics aggregator."""
        if not results:
            return None, {}

        # 1) Sample-weighted average of client weights
        weights_results = [
            (parameters_to_ndarrays(fit_res.parameters), fit_res.num_examples)
            for _, fit_res in results
        ]
        weighted_avg = self._weighted_average(weights_results)

        # 2) β-mixing with previous global
        if self.previous_global_parameters is not None:
            mixed_parameters = self._beta_mixing(
                weighted_avg, self.previous_global_parameters, self.beta
            )
        else:
            mixed_parameters = weighted_avg

        # 3) Call configured fit-metrics aggregator to populate server cache
        metrics_aggregated: Dict[str, Scalar] = {}
        agg_fn = getattr(self, "fit_metrics_aggregation_fn", None)
        if agg_fn is not None:
            try:
                # Newer style: (server_round, results, failures)
                metrics_aggregated = agg_fn(server_round, results, failures)
            except TypeError:
                # Classic style: list of (num_examples, metrics)
                metrics_list = [(fit_res.num_examples, fit_res.metrics) for _, fit_res in results]
                metrics_aggregated = agg_fn(metrics_list)

        # 4) Update previous_global and return
        self.previous_global_parameters = mixed_parameters
        return ndarrays_to_parameters(mixed_parameters), metrics_aggregated
    
    def _weighted_average(self, weights_and_sizes: List[Tuple[NDArrays, int]]) -> NDArrays:
        """Compute sample-weighted average of client parameters."""
        if not weights_and_sizes:
            raise ValueError("Cannot average empty list")
        
        # Calculate total samples
        total_samples = sum(num_examples for _, num_examples in weights_and_sizes)
        if total_samples == 0:
            raise ValueError("Total samples cannot be zero")
        
        # Initialize with zeros using first client's structure
        first_weights, _ = weights_and_sizes[0]
        avg_weights = [np.zeros_like(w) for w in first_weights]
        
        # Weighted sum of all client parameters
        for client_weights, num_examples in weights_and_sizes:
            weight = num_examples / total_samples
            for i, w in enumerate(client_weights):
                avg_weights[i] += w * weight
            
        return avg_weights
    
    def _beta_mixing(
        self, 
        new_params: NDArrays, 
        old_params: NDArrays, 
        beta: float
    ) -> NDArrays:
        """Apply β-mixing: w^{t+1} = (1-β)*w^t + β*new_avg."""
        mixed_params = []
        for new_p, old_p in zip(new_params, old_params):
            mixed_p = (1.0 - beta) * old_p + beta * new_p
            mixed_params.append(mixed_p)
        return mixed_params

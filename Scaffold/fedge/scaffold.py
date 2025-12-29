import numpy as np
import torch
import time
from typing import Any, Dict, List, Tuple
from copy import deepcopy

from flwr.common import NDArrays, Parameters, FitIns, ndarrays_to_parameters, parameters_to_ndarrays
from flwr.common.typing import FitRes, EvaluateRes, Scalar
from flwr.server.strategy import FedAvg
from flwr.server.client_manager import ClientManager
from flwr.server.client_proxy import ClientProxy


class Scaffold(FedAvg):
    """
    SCAFFOLD strategy for Flower.

    1. Server keeps a global control variate c_global.
    2. Each client keeps its own c_local.
    3. Clients solve local problem with gradient correction ∇F + c_global - c_local.
    4. Clients return two diffs: y_delta = W_local - W_global, and c_delta = c_local_new - c_local_old.
    5. Server updates:
         W_global ← W_global + η * average(y_delta)
         c_global ← c_global + average(c_delta)
    """

    def __init__(
        self,
        *,
        fraction_fit: float = 1.0,
        fraction_evaluate: float = 1.0,
        min_fit_clients: int = 2,
        min_evaluate_clients: int = 2,
        min_available_clients: int = 2,
        global_lr: float = 1.0,
        **kwargs: Any,
    ):
        super().__init__(
            fraction_fit=fraction_fit,
            fraction_evaluate=fraction_evaluate,
            min_fit_clients=min_fit_clients,
            min_evaluate_clients=min_evaluate_clients,
            min_available_clients=min_available_clients,
            **kwargs,
        )
        self.global_lr = global_lr
        self.c_global: NDArrays = []
        self.c_locals: Dict[str, NDArrays] = {}
        self._round_t0 = None  # wall-clock per round
        self._initialized = False
        # Store latest global parameters for update
        self._latest_global_parameters: Parameters | None = None

    def _init_control(self, parameters: Parameters) -> None:
        """Initialize c_global (zeros) on first round."""
        nd = parameters_to_ndarrays(parameters)
        self.c_global = [np.zeros(w.shape, dtype=np.float32) for w in nd]
        self._initialized = True

    def configure_fit(
        self,
        server_round: int,
        parameters: Parameters,
        client_manager: ClientManager,
    ) -> List[Tuple[ClientProxy, FitIns]]:
        # 0) Save the global parameters for later in aggregate_fit
        self._latest_global_parameters = parameters

        # 1) initialize c_global on round 0
        if not self._initialized:
            self._init_control(parameters)

        # 2) get the default FedAvg FitIns
        fit_ins = super().configure_fit(server_round, parameters, client_manager)
        custom: List[Tuple[ClientProxy, FitIns]] = []
        m = len(self.c_global)

        for client, ins in fit_ins:
            cid = client.cid
            if cid not in self.c_locals:
                self.c_locals[cid] = deepcopy(self.c_global)

            # pack global + c_global + c_local into one Parameters
            global_nd = parameters_to_ndarrays(ins.parameters)
            # Avoid unnecessary deep copies; Flower serialization will detach arrays
            to_send = global_nd + list(self.c_global) + list(self.c_locals[cid])
            send_params = ndarrays_to_parameters(to_send)

            cfg = {**ins.config, "return_diff": True, "n_layers": m}
            custom.append((client, FitIns(parameters=send_params, config=cfg)))

        self._round_t0 = time.perf_counter()
        return custom


    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, FitRes]],
        failures: List[BaseException],
    ) -> Tuple[Parameters, Dict[str, Scalar]]:
        # Buffers for the two kinds of deltas
        y_deltas: List[List[torch.Tensor]] = []
        c_deltas: List[List[torch.Tensor]] = []
        weights: List[int] = []

        # Per-client tracking removed (CSV writing handled in server)

        m = len(self.c_global)

        # 1) Unpack each client's FitRes
        for client, fit_res in results:
            # Convert the returned Parameters → list of np.ndarray
            returned_params: Parameters = fit_res.parameters
            nd_list = parameters_to_ndarrays(returned_params)
            num_examples = fit_res.num_examples
            # metrics = fit_res.metrics  # if you need them

            # First m arrays are y_delta, next m are c_delta
            y_np = nd_list[:m]
            c_np = nd_list[m : 2 * m]

            y_deltas.append([torch.tensor(x) for x in y_np])
            c_deltas.append([torch.tensor(x) for x in c_np])
            weights.append(num_examples)

            # Client metrics collection removed (no local CSV writing)

            # Update this client's c_local with float32 operations
            cid = client.cid
            self.c_locals[cid] = [
                cl.astype(np.float32, copy=False) + cd.numpy().astype(np.float32)
                for cl, cd in zip(self.c_locals[cid], c_deltas[-1])
            ]

        # 2) Compute weighted average of y_deltas → global update
        total_weight = float(sum(weights))
        norm_weights = [w / total_weight for w in weights]

        # Fetch the previous global as a list
        prev_global_nd = parameters_to_ndarrays(self._latest_global_parameters)
        W = [torch.tensor(x) for x in prev_global_nd]

        for idx, parts in enumerate(zip(*y_deltas)):
            stacked = torch.stack(list(parts), dim=0)  # shape: [num_clients, …]
            # build a broadcastable weight tensor
            w = torch.tensor(norm_weights, device=stacked.device)
            for _ in range(stacked.ndim - 1):
                w = w.unsqueeze(1)  # now shape [num_clients,1,1,…,1]
            avg_delta = (stacked * w).sum(dim=0)
            if W[idx].is_floating_point():
                W[idx] = W[idx] + self.global_lr * avg_delta
            # else: leave integer/bool buffers unchanged

        new_global = ndarrays_to_parameters([w.numpy() for w in W])

        # 3) Update c_global = c_global + avg(c_deltas)
    
        for idx, parts in enumerate(zip(*c_deltas)):
            stacked = torch.stack(list(parts), dim=0)
            w = torch.tensor(norm_weights, device=stacked.device)
            for _ in range(stacked.ndim - 1):
                w = w.unsqueeze(1)
            avg_c = (stacked * w).sum(dim=0)
            self.c_global[idx] = self.c_global[idx].astype(np.float32, copy=False) \
                                 + avg_c.cpu().numpy().astype(np.float32)

        # Wall clock for this round
        wall_clock_sec = None
        if self._round_t0 is not None:
            wall_clock_sec = float(time.perf_counter() - self._round_t0)

        # CSV writing is handled by the server layer to ensure per-seed routing
        # and schema parity with FedProx. Strategy returns aggregated metrics only.

        # Let FedAvg compute and return aggregated metrics (no CSV here)
        _, agg_metrics = super().aggregate_fit(server_round, results, failures)
        return new_global, dict(agg_metrics)

    def aggregate_evaluate(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, EvaluateRes]],
        failures: List[BaseException],
    ) -> Tuple[float, Dict[str, Scalar]]:
        """Aggregate evaluation results and write per-client test metrics to clients.csv."""
        # CSV writing is handled by the server layer; just delegate to parent aggregation
        return super().aggregate_evaluate(server_round, results, failures)

# client_app.py — SCAFFOLD client (FedProx-parity plumbing)
import numpy as np
import torch
import torch.nn as nn
from typing import Any, Dict, List, Union

from flwr.client import ClientApp, NumPyClient
from flwr.common import Parameters, NDArrays, parameters_to_ndarrays

from fedge.task import Net, load_data, set_weights, test, get_weights, set_global_seed
import time


class FlowerClient(NumPyClient):
    def __init__(
        self,
        net: nn.Module,
        trainloader: torch.utils.data.DataLoader,
        valloader: torch.utils.data.DataLoader,
        local_epochs: int,
        local_lr: float = 1e-2,
        client_id: int = 0,
        base_seed: int = 0,
    ):
        super().__init__()
        self.net = net
        self.trainloader = trainloader
        self.valloader = valloader
        self.local_epochs = local_epochs
        self.local_lr = local_lr
        self.client_id = client_id
        self.base_seed = base_seed

        # Device and seeding
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.net.to(self.device)
        set_global_seed(self.base_seed + self.client_id)

        # Loss function
        self.criterion = nn.CrossEntropyLoss()

    def _unwrap_parameters(self, parameters: Union[List[np.ndarray], Parameters]) -> List[np.ndarray]:
        if isinstance(parameters, list):
            return parameters
        return parameters_to_ndarrays(parameters)

    def fit(self, parameters: NDArrays, config: Dict[str, Any]) -> tuple[List[np.ndarray], int, Dict[str, Any]]:
        """SCAFFOLD local training returning [y_delta, c_delta] concatenated."""
        t_start = time.perf_counter()

        # Server may send packed [W, c_global, c_local]
        return_diff = config.get("return_diff", False)
        n_layers = config.get("n_layers", None)
        if return_diff and n_layers is not None:
            all_nd = self._unwrap_parameters(parameters)
            m = int(n_layers)
            global_nd = all_nd[:m]
            c_global = all_nd[m : 2 * m]
            c_local = all_nd[2 * m : 3 * m]
        else:
            global_nd = self._unwrap_parameters(parameters)
            c_global = [np.zeros_like(x) for x in global_nd]
            c_local = [np.zeros_like(x) for x in global_nd]

        # Load global weights
        set_weights(self.net, global_nd)

        # Name->array maps for control variates (state_dict order)
        sd_keys = list(self.net.state_dict().keys())
        m = len(sd_keys)
        assert len(c_global) == m and len(c_local) == m, "control variates must match state_dict size"
        c_global_map = {k: v for k, v in zip(sd_keys, c_global)}
        c_local_map = {k: v for k, v in zip(sd_keys, c_local)}

        # Training with SCAFFOLD correction
        optimizer = torch.optim.SGD(self.net.parameters(), lr=self.local_lr)
        loss_sum = 0.0
        correct = 0
        total_samples = 0
        step_count = 0

        def scaffold_ctrl_term() -> torch.Tensor:
            terms = []
            for name, p in self.net.named_parameters():
                cg = torch.as_tensor(c_global_map[name], device=self.device, dtype=p.dtype)
                cl = torch.as_tensor(c_local_map[name], device=self.device, dtype=p.dtype)
                terms.append((cg.reshape(-1) - cl.reshape(-1)).dot(p.reshape(-1)))
            return sum(terms) if terms else torch.zeros((), device=self.device)

        self.net.train()
        for _ in range(self.local_epochs):
            for x, y in self.trainloader:
                x, y = x.to(self.device), y.to(self.device)
                y = y.long().view(-1)
                optimizer.zero_grad()
                logits = self.net(x)
                ce_loss = self.criterion(logits, y)
                loss = ce_loss + scaffold_ctrl_term()
                loss.backward()
                optimizer.step()
                step_count += 1

                # accumulate stats
                loss_sum += ce_loss.item() * y.size(0)
                correct += (logits.argmax(1) == y).sum().item()
                total_samples += y.size(0)

        # Deltas
        local_nd = get_weights(self.net)
        y_delta = [loc - glob for loc, glob in zip(local_nd, global_nd)]
        K = max(1, step_count)
        c_local_new = [cl - cg - (yd / (self.local_lr * K)) for cl, yd, cg in zip(c_local, y_delta, c_global)]
        c_delta = [new - old for new, old in zip(c_local_new, c_local)]

        out_nd = y_delta + c_delta

        # Cost metrics
        comp_time_sec = time.perf_counter() - t_start
        recv_arrays = list(global_nd) + list(c_global) + list(c_local)
        download_bytes = int(sum(arr.nbytes for arr in recv_arrays))
        upload_bytes = int(sum(arr.nbytes for arr in out_nd))

        train_loss_mean = loss_sum / max(total_samples, 1)
        train_accuracy_mean = correct / max(total_samples, 1)

        fit_metrics = {
            "comp_time_sec": comp_time_sec,
            "download_bytes": download_bytes,
            "upload_bytes": upload_bytes,
            "train_loss_mean": float(train_loss_mean),
            "train_accuracy_mean": float(train_accuracy_mean),
            "total_train_samples": int(total_samples),
            "num_inner_batches": int(step_count),
        }

        return out_nd, len(self.trainloader.dataset), fit_metrics

    def evaluate(self, parameters, config):
        t0 = time.perf_counter()
        nds = self._unwrap_parameters(parameters)
        set_weights(self.net, nds)
        eval_download_bytes = int(sum(arr.nbytes for arr in nds))
        loss, acc = test(self.net, self.valloader, self.device)
        metrics = {
            "cid": int(self.client_id),
            "test_loss": float(loss),
            "test_accuracy": float(acc),
            "download_bytes": eval_download_bytes,
            "upload_bytes": 0,
            "comp_time_sec": time.perf_counter() - t0,
            "client_id": int(self.client_id),
            "n_val_examples": len(self.valloader.dataset),
        }
        return float(loss), len(self.valloader.dataset), metrics


def client_fn(context):
    # Config
    dataset_flag = context.node_config.get("dataset_flag", "cifar10")
    base_seed = int(context.run_config.get("seed", 0))

    pid, num_parts = (
        context.node_config["partition-id"],
        context.node_config["num-partitions"],
    )

    trainloader, valloader, n_classes = load_data(
        dataset_flag,
        pid,
        num_parts,
        batch_size=context.run_config.get("batch_size", 32),
        alpha=context.run_config.get("dirichlet_alpha", 0.5),
        seed=base_seed,
    )

    sample, _ = next(iter(trainloader))
    _, c, h, w = sample.shape
    net = Net(in_ch=c, img_h=h, img_w=w, n_class=n_classes)

    return FlowerClient(
        net,
        trainloader,
        valloader,
        local_epochs=context.run_config["local-epochs"],
        client_id=pid,
        base_seed=base_seed,
    ).to_client()


app = ClientApp(client_fn)

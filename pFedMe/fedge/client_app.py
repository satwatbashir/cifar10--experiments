# client_app.py
import numpy as np
import torch
import torch.nn as nn
from typing import Any, Dict, List, Union

from flwr.client import ClientApp, NumPyClient
from flwr.common import Parameters, NDArrays, parameters_to_ndarrays

from fedge.task import (
    Net,
    get_weights,
    load_data_with_larger_test,
    set_weights,
    test,
    set_global_seed,
)
import copy
import time
from torch.amp import autocast, GradScaler


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
        self.net = net                      # global model copy (w)
        self.trainloader = trainloader
        self.valloader = valloader
        self.local_epochs = local_epochs
        self.local_lr = local_lr
        self.client_id = client_id
        self.base_seed = int(base_seed)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.net.to(self.device)

        # personalized model (theta)
        self.theta_net = copy.deepcopy(self.net)
        self.theta_net.to(self.device)

        # Loss
        self.criterion = nn.CrossEntropyLoss()

        # Reproducibility (tied to run seed)
        set_global_seed(self.base_seed + int(client_id))

        # Perf flags (optional)
        torch.backends.cudnn.benchmark = True

        # Initialized later when first round arrives
        self.theta_initialized = False

        # Runtime stats
        self.comp_time_sec = 0.0
        self.upload_bytes = 0
        self.download_bytes = 0
        self._last_train_acc = None
        self._last_train_loss = None

    def _unwrap_parameters(
        self, parameters: Union[List[np.ndarray], Parameters]
    ) -> List[np.ndarray]:
        """Handle either a raw list of NDArrays or a Parameters object."""
        if isinstance(parameters, list):
            return parameters
        return parameters_to_ndarrays(parameters)

    # ----------------------------- Fit (train) -----------------------------
    def fit(
        self, parameters: NDArrays, config: Dict[str, Any]
    ) -> tuple[List[np.ndarray], int, Dict[str, Any]]:
        """Authentic pFedMe bi-level optimization with inner/outer loops."""
        t_start = time.perf_counter()

        # Load incoming global weights into w
        global_nd = self._unwrap_parameters(parameters)
        set_weights(self.net, global_nd)  # w ← global
        set_weights(self.theta_net, global_nd)  # θ ← w (reset each round)
        self.theta_initialized = True

        # pFedMe hypers (from server run_config)
        lamda: float = float(config.get("lamda", 15.0))
        inner_steps: int = int(config.get("inner_steps", 5))
        outer_steps: int = int(config.get("outer_steps", 1))
        inner_lr: float = float(config.get("inner_lr", 0.01))
        outer_lr: float = float(config.get("outer_lr", 0.01))
        local_epochs: int = int(config.get("local-epochs", self.local_epochs))

        theta_optimizer = torch.optim.SGD(self.theta_net.parameters(), lr=inner_lr)
        amp_enabled = torch.cuda.is_available()
        scaler = GradScaler(enabled=amp_enabled)

        # Collect metrics
        all_losses: List[float] = []
        all_correct = 0
        all_total = 0

        # Bi-level optimization
        for _ in range(local_epochs):
            for _ in range(outer_steps):
                # Cache current w (detached) once per outer-step, not per batch
                with torch.no_grad():
                    w_params_detached = [p.detach() for p in self.net.parameters()]

                batch_count = 0
                for x, y in self.trainloader:
                    if batch_count >= inner_steps:
                        break

                    x = x.to(self.device, non_blocking=True)
                    y = y.to(self.device, non_blocking=True)
                    if y.ndim > 1:
                        y = y.squeeze()
                    y = y.long()

                    theta_optimizer.zero_grad(set_to_none=True)

                    # Mixed precision for forward and loss
                    with autocast(device_type="cuda", enabled=amp_enabled):
                        logits = self.theta_net(x)
                        loss = self.criterion(logits, y)

                        # Moreau envelope: (λ/2)*||θ - w||^2 against cached w
                        reg = torch.tensor(0.0, device=self.device)
                        for theta_param, w_param in zip(self.theta_net.parameters(), w_params_detached):
                            diff = theta_param - w_param
                            reg = reg + torch.sum(diff * diff)
                        loss = loss + (lamda / 2.0) * reg

                    # Guard
                    if not torch.isfinite(loss).all():
                        set_weights(self.theta_net, get_weights(self.net))  # θ ← w
                        break

                    scaler.scale(loss).backward()
                    # Unscale before clipping
                    scaler.unscale_(theta_optimizer)
                    torch.nn.utils.clip_grad_norm_(self.theta_net.parameters(), 5.0)
                    scaler.step(theta_optimizer)
                    scaler.update()

                    # Metrics
                    all_losses.append(loss.item())
                    _, preds = torch.max(logits, 1)
                    all_correct += (preds == y).sum().item()
                    all_total += y.size(0)
                    batch_count += 1

                # Outer update: w ← w - η_out * λ * (w - θ)
                with torch.no_grad():
                    for w_param, theta_param in zip(self.net.parameters(), self.theta_net.parameters()):
                        w_param.data = w_param.data - outer_lr * lamda * (w_param.data - theta_param.data)

        # Prepare results
        local_nd = get_weights(self.net)  # return updated w
        t_end = time.perf_counter()

        # Comms and time
        download_bytes = int(sum(arr.nbytes for arr in global_nd))
        upload_bytes = int(sum(arr.nbytes for arr in local_nd))
        self.comp_time_sec = t_end - t_start
        self.upload_bytes = upload_bytes
        self.download_bytes = download_bytes

        # Train metrics
        train_loss_mean = float(np.mean(all_losses)) if all_losses else 0.0
        train_accuracy_mean = float(all_correct / max(1, all_total)) if all_total > 0 else 0.0
        self._last_train_acc = train_accuracy_mean
        self._last_train_loss = train_loss_mean

        # Report MB (not bytes)
        fit_metrics = {
            "cid": int(self.client_id),
            "comp_time_sec": self.comp_time_sec,
            "download_MB": download_bytes / 1e6,
            "upload_MB": upload_bytes / 1e6,
            "train_loss_mean": train_loss_mean,
            "train_accuracy_mean": train_accuracy_mean,
            "inner_steps": int(inner_steps),
            "inner_batches": len(all_losses),
            "total_train_samples": int(all_total),
        }
        return local_nd, len(self.trainloader.dataset), fit_metrics

    # --------------------------- Evaluate (test) ---------------------------
    def evaluate(self, parameters, config) -> tuple[float, int, dict]:
        """Client-side evaluation supporting both global and personalized models."""
        t_eval_start = time.perf_counter()

        # Check if personalized evaluation is requested
        use_personalized = bool(config.get("personalized_eval", False)) and getattr(self, "theta_initialized", False)

        if use_personalized:
            # Evaluate the personalized model θ on the client's own shard
            net_to_eval = self.theta_net
            eval_download_bytes = 0  # No download needed for personalized model
        else:
            # Evaluate the server/global model on the client's shard (baseline/comparability)
            global_nd = self._unwrap_parameters(parameters)
            set_weights(self.net, global_nd)
            net_to_eval = self.net
            eval_download_bytes = int(sum(arr.nbytes for arr in global_nd))

        # Evaluate on this client's validation/test shard
        test_loss, test_acc = test(net_to_eval, self.valloader, self.device)

        comp_time = time.perf_counter() - t_eval_start

        # Metrics use names expected by server's aggregator
        metrics = {
            "test_loss": float(test_loss),
            "test_accuracy": float(test_acc),
            "eval_download_bytes": eval_download_bytes,
            "comp_time_sec": comp_time,
            "client_id": self.client_id,
            "n_val_examples": len(self.valloader.dataset),
            "personalized_eval": use_personalized,
        }

        # Add personalized-specific metrics when applicable
        if use_personalized:
            metrics["personalized_accuracy"] = float(test_acc)
            metrics["personalized_loss"] = float(test_loss)

        # Flower expects (loss, num_examples, metrics)
        return float(test_loss), len(self.valloader.dataset), metrics

# ----------------------------- Client factory -----------------------------
def client_fn(context):
    # 1) Data + model
    dataset_flag = context.node_config.get("dataset_flag", "cifar10")

    pid, num_parts = (
        context.node_config["partition-id"],
        context.node_config["num-partitions"],
    )

    # Run seed
    base_seed = int(context.run_config.get("seed", 0))

    trainloader, valloader, n_classes = load_data_with_larger_test(
        dataset_flag,
        pid,
        num_parts,
        batch_size=context.run_config.get("batch_size", 32),
        alpha=context.run_config["dirichlet_alpha"],
        seed=base_seed,
    )

    # Infer shape and build model
    sample, _ = next(iter(trainloader))
    _, c, h, w = sample.shape
    net = Net(in_ch=c, img_h=h, img_w=w, n_class=n_classes)

    # 2) Return NumPyClient
    return FlowerClient(
        net=net,
        trainloader=trainloader,
        valloader=valloader,
        local_epochs=context.run_config["local-epochs"],
        client_id=pid,  # For reproducibility
        base_seed=base_seed,
    ).to_client()


app = ClientApp(client_fn)

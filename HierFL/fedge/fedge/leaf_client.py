#!/usr/bin/env python3
import argparse
import json
import os
import sys
import time
import signal
import warnings
import logging
from pathlib import Path

import torch
import numpy as np
import grpc
from flwr.client import NumPyClient, start_client

# Import your task utilities:
from fedge.task import Net, load_data, set_weights, train, test, get_weights, set_global_seed

# ─── Logging setup ───────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.WARNING,
    format='[%(asctime)s] %(levelname)s - %(message)s',
)
logger = logging.getLogger(__name__)

warnings.filterwarnings("ignore", category=DeprecationWarning, module="flwr")
for name in ("flwr", "ece", "grpc"):
    logging.getLogger(name).setLevel(logging.ERROR)

# =============================================================================
# Flower NumPyClient implementation
# =============================================================================
class FlowerClient(NumPyClient):
    def __init__(self, net, trainloader, valloader, local_epochs):
        self.net = net
        self.trainloader = trainloader
        self.valloader = valloader
        self.local_epochs = local_epochs
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.net.to(self.device)

    def get_properties(self, config):
        from flwr.common import Properties
        cid = os.environ.get("CLIENT_ID", "")
        return Properties(other={"client_id": cid})

    def get_parameters(self, config):
        # Return the model parameters as numpy arrays
        return [val.cpu().numpy() for _, val in self.net.state_dict().items()]

    def fit(self, parameters, config):
        """Fit model for local_epochs full passes over dataset (matches FedProx)."""
        cid = os.environ.get("CLIENT_ID", "")
        # Communication: bytes received
        try:
            bytes_down = sum(arr.nbytes for arr in parameters)
        except Exception:
            bytes_down = 0
        start_time = time.perf_counter()
        # Set model weights
        set_weights(self.net, parameters)
        set_end = time.perf_counter()

        # Local training loop for `local_epochs` full passes over dataset
        optimizer = torch.optim.SGD(self.net.parameters(), lr=0.01)
        criterion = torch.nn.CrossEntropyLoss()
        self.net.train()
        all_losses = []
        total_correct = 0
        total_seen = 0
        num_batches = 0

        for _ in range(self.local_epochs):
            for x, y in self.trainloader:
                x, y = x.to(self.device), y.to(self.device)
                if y.ndim > 1:
                    y = y.squeeze()
                y = y.long()
                optimizer.zero_grad()
                logits = self.net(x)
                loss = criterion(logits, y)
                loss.backward()
                optimizer.step()
                all_losses.append(loss.item())
                _, preds = torch.max(logits, 1)
                total_correct += (preds == y).sum().item()
                total_seen += y.size(0)
                num_batches += 1

        train_end = time.perf_counter()
        # Get updated weights and compute bytes sent
        weights = get_weights(self.net)
        try:
            bytes_up = sum(arr.nbytes for arr in weights)
        except Exception:
            bytes_up = 0
        end_time = time.perf_counter()
        # Compute times
        comp_time_sec = train_end - set_end
        # Compute training metrics
        train_loss_mean = float(np.mean(all_losses)) if all_losses else 0.0
        train_accuracy_mean = float(total_correct / max(1, total_seen)) if total_seen > 0 else 0.0
        # Collect metrics using baseline key names
        metrics = {
            "comp_time_sec": comp_time_sec,
            "download_bytes": int(bytes_down),
            "upload_bytes": int(bytes_up),
            "train_loss_mean": train_loss_mean,
            "train_accuracy_mean": train_accuracy_mean,
            "num_inner_batches": int(num_batches),
            "total_train_samples": int(len(self.trainloader.dataset)),
            "client_id": cid,
        }
        return weights, len(self.trainloader.dataset), metrics

    def evaluate(self, parameters, config):
        """Evaluate global model on this client's own shard only (baseline)."""
        set_weights(self.net, parameters)
        loss, accuracy = test(self.net, self.valloader, self.device)
        cid = os.environ.get("CLIENT_ID", "")
        metrics = {"test_loss": float(loss), "test_accuracy": float(accuracy), "client_id": cid}
        return float(loss), len(self.valloader.dataset), metrics

# =============================================================================
# Signal handler for graceful shutdown
# =============================================================================
def handle_signal(sig, frame):
    client_id = os.environ.get("CLIENT_ID", "leaf_client")
    logger.info(f"[{client_id}] Received signal {sig}, shutting down gracefully...")
    sys.exit(0)

# =============================================================================
# Main: parse arguments, load CIFAR-10 shard, start Flower client
# =============================================================================
def main():
    # 1) Catch SIGINT/SIGTERM so we can clean up nicely
    signal.signal(signal.SIGINT, handle_signal)
    signal.signal(signal.SIGTERM, handle_signal)

    parser = argparse.ArgumentParser()
    parser.add_argument("--partition_id", type=int, required=True)
    parser.add_argument("--num_partitions", type=int, required=True)
    # Dataset flag for CIFAR-10
    parser.add_argument(
        "--dataset_flag",
        type=str,
        choices=["cifar10"],
        required=True,
        help="Dataset to load (cifar10)",
    )
    parser.add_argument("--local_epochs", type=int, required=True)
    parser.add_argument("--server_addr", type=str, default=os.getenv("LEAF_ADDRESS", "127.0.0.1:6100"))
    parser.add_argument(
        "--max_retries", type=int, default=5, help="Max gRPC connection retries"
    )
    parser.add_argument(
        "--retry_delay", type=int, default=2, help="Seconds between retry attempts"
    )
    args = parser.parse_args()

    # 2) Build a human-readable CLIENT_ID for logging (e.g. "leaf_0_client_3")
    client_id = f"leaf_{os.environ.get('SERVER_ID','?')}_client_{args.partition_id}"
    os.environ["CLIENT_ID"] = client_id

    # 3) Determine indices from PARTITIONS_JSON (MANDATORY)
    parts_path = os.environ.get("PARTITIONS_JSON")
    assert parts_path is not None and Path(parts_path).exists(), "PARTITIONS_JSON is required and must exist"
    with open(parts_path, "r", encoding="utf-8") as fp:
        combined = json.load(fp)
    sid = int(os.environ.get("SERVER_ID", "0"))
    # Expect combined mapping with keys 'train' and 'test'
    assert "train" in combined and "test" in combined, "PARTITIONS_JSON must contain 'train' and 'test' keys"
    train_map = combined["train"]
    test_map = combined["test"]
    indices = train_map[str(sid)][str(args.partition_id)]
    indices_test = test_map[str(sid)][str(args.partition_id)]

    # 3a) Deterministic seeding per process and per client
    import toml
    project_root = Path(__file__).resolve().parent.parent
    cfg = toml.load(project_root / "pyproject.toml")
    base_seed = int(cfg["tool"]["flwr"]["app"]["config"].get("seed", 0))
    clients_per_server = args.num_partitions
    global_client_id = sid * clients_per_server + args.partition_id
    set_global_seed(base_seed + global_client_id)

    trainloader, valloader, n_classes = load_data(
        args.dataset_flag,
        args.partition_id,
        args.num_partitions,
        indices=indices,
        indices_test=indices_test,
    )

    # 4) Instantiate LeNet model
    net = Net(n_class=n_classes)

    # 5) Wrap in the FlowerClient
    client = FlowerClient(net, trainloader, valloader, args.local_epochs)

    # 6) Connect (with retry logic) to the leaf server’s Flower endpoint
    retries = 0
    while retries < args.max_retries:
        try:
            logger.info(f"[{client_id}] Connecting to server at {args.server_addr}")
            start_client(server_address=args.server_addr, client=client.to_client())
            logger.info(f"[{client_id}] Client session completed successfully")
            break
        except grpc.RpcError as e:
            if e.code() == grpc.StatusCode.UNAVAILABLE and retries < args.max_retries:
                retries += 1
                logger.warning(
                    f"[{client_id}] Connection failed (attempt {retries}/{args.max_retries}): {e.details()}"
                )
                logger.info(f"[{client_id}] Retrying in {args.retry_delay}s …")
                time.sleep(args.retry_delay)
            else:
                logger.error(f"[{client_id}] Unexpected gRPC error: {e.details()}")
                return
        except Exception as e:
            logger.error(f"[{client_id}] Unexpected error: {e}")
            return

if __name__ == "__main__":
    main()

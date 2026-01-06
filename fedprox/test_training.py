#!/usr/bin/env python3
"""
Standalone test script - bypasses Flower/Ray to test pure PyTorch training.
If this works, the issue is in Flower/Ray, not the training code.
"""
import sys
import os

# Add fedge to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
from fedge.task import ResNet18, load_data, test, set_global_seed

def main():
    print("=" * 60)
    print("STANDALONE PYTORCH TEST (No Flower/Ray)")
    print("=" * 60)

    # Set seed for reproducibility
    set_global_seed(42)

    # Check device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Load data for 1 client (partition 0 out of 3)
    print("\n[1] Loading data...")
    trainloader, testloader, n_classes = load_data(
        "cifar10",
        partition_id=0,
        num_partitions=3,  # Only 3 clients for quick test
        batch_size=64,
        alpha=0.5,
        seed=42
    )
    print(f"    Train batches: {len(trainloader)}")
    print(f"    Test batches: {len(testloader)}")
    print(f"    Classes: {n_classes}")

    # Create model
    print("\n[2] Creating ResNet18 model...")
    net = ResNet18(num_classes=n_classes).to(device)
    params = sum(p.numel() for p in net.parameters())
    print(f"    Parameters: {params:,} (~{params * 4 / 1e6:.1f} MB)")

    # Train 1 epoch (or just 20 batches for quick test)
    print("\n[3] Training (20 batches)...")
    net.train()
    optimizer = torch.optim.SGD(net.parameters(), lr=0.01)
    criterion = torch.nn.CrossEntropyLoss()

    for i, (x, y) in enumerate(trainloader):
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        out = net(x)
        loss = criterion(out, y)
        loss.backward()
        optimizer.step()

        if i % 5 == 0:
            print(f"    Batch {i:3d}, Loss: {loss.item():.4f}")

        if i >= 20:  # Only 20 batches for quick test
            break

    # Evaluate
    print("\n[4] Evaluating...")
    loss, acc = test(net, testloader, device)
    print(f"    Test Loss: {loss:.4f}")
    print(f"    Test Accuracy: {acc:.4f} ({acc*100:.1f}%)")

    print("\n" + "=" * 60)
    print("SUCCESS! Training loop works without Flower/Ray.")
    print("If Flower hangs, the issue is in Ray/Flower, not the model.")
    print("=" * 60)

if __name__ == "__main__":
    main()

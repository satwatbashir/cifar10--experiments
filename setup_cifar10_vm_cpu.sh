#!/bin/bash
# CIFAR10 Fedge - Complete VM Setup Script (CPU-only version)
# Run on a fresh Ubuntu 22.04 GCP VM WITHOUT GPU

set -e

echo "=== CIFAR10 Fedge VM Setup (CPU-only) ==="
echo ""

# 1. System dependencies
echo "[1/6] Installing system dependencies..."
sudo apt-get update
sudo apt-get install -y python3-pip python3-venv git unzip wget

# 2. Create virtual environment
echo "[2/6] Creating virtual environment..."
cd ~
python3 -m venv .venv
source ~/.venv/bin/activate

# 3. Install PyTorch (CPU-only version)
echo "[3/6] Installing PyTorch (CPU-only)..."
pip install --upgrade pip
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

# 4. Install Flower and dependencies
echo "[4/6] Installing Flower and dependencies..."
pip install -U "flwr[simulation]"
pip install flwr-datasets
pip install pandas numpy scipy matplotlib seaborn tqdm typing-extensions toml

# 5. Clone CIFAR10 repo
echo "[5/6] Cloning CIFAR10 repository..."
cd ~
if [ -d "cifar10" ]; then
    echo "cifar10 folder exists, pulling latest..."
    cd cifar10 && git pull && cd ~
else
    git clone https://github.com/satwatbashir/cifar10--experiments.git cifar10
fi

# 6. Install projects
echo "[6/6] Installing projects..."

# Install HierFL
echo "  - Installing HierFL..."
cd ~/cifar10/HierFL/fedge
pip install -e .

# Install FedProx
echo "  - Installing FedProx..."
cd ~/cifar10/fedprox
pip install -e .

# Verify
echo ""
echo "=== Verifying Installation ==="
python3 -c "import torch; print(f'PyTorch: {torch.__version__}')"
python3 -c "import torch; print(f'CUDA available: {torch.cuda.is_available()} (expected: False for CPU)')"
python3 -c "import flwr; print(f'Flower: {flwr.__version__}')"

echo ""
echo "=== Setup Complete ==="
echo ""
echo "To run HierFL (hierarchical):"
echo "  source ~/.venv/bin/activate"
echo "  cd ~/cifar10/HierFL/fedge"
echo "  python3 orchestrator.py"
echo ""
echo "To run FedProx (flat):"
echo "  source ~/.venv/bin/activate"
echo "  cd ~/cifar10/fedprox"
echo "  flwr run . --run-config num-server-rounds=200"
echo ""

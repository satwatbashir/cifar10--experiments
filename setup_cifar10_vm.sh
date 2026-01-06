#!/bin/bash
# CIFAR10 Fedge - Complete VM Setup Script
# Run on a fresh Ubuntu 22.04 GCP VM with NVIDIA T4 GPU

set -e

echo "=== CIFAR10 Fedge VM Setup ==="
echo ""

# 1. System dependencies
echo "[1/7] Installing system dependencies..."
sudo apt-get update
sudo apt-get install -y python3-pip python3-venv git unzip wget

# 2. Install NVIDIA drivers (if not already installed)
echo "[2/7] Checking/Installing NVIDIA drivers..."
if ! command -v nvidia-smi &> /dev/null; then
    echo "Installing NVIDIA drivers..."
    sudo apt-get install -y nvidia-driver-535
    echo ""
    echo "=== NVIDIA drivers installed. REBOOT REQUIRED! ==="
    echo "Run: sudo reboot"
    echo "After reboot, run this script again."
    exit 0
fi
echo "NVIDIA drivers already installed."
nvidia-smi

# 3. Create virtual environment
echo "[3/7] Creating virtual environment..."
cd ~
python3 -m venv .venv
source ~/.venv/bin/activate

# 4. Install PyTorch with CUDA
echo "[4/7] Installing PyTorch with CUDA..."
pip install --upgrade pip
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# 5. Install Flower and dependencies
echo "[5/7] Installing Flower and dependencies..."
pip install -U "flwr[simulation]"
pip install flwr-datasets
pip install pandas numpy scipy matplotlib seaborn tqdm typing-extensions toml

# 6. Clone CIFAR10 repo
echo "[6/7] Cloning CIFAR10 repository..."
cd ~
if [ -d "cifar10" ]; then
    echo "cifar10 folder exists, pulling latest..."
    cd cifar10 && git pull && cd ~
else
    git clone https://github.com/satwatbashir/cifar10--experiments.git cifar10
fi

# 7. Install project
echo "[7/7] Installing Fedge project..."
cd ~/cifar10/Fedge-Simulation/fedge
pip install -e .

# Verify
echo ""
echo "=== Verifying Installation ==="
python3 -c "import torch; print(f'PyTorch: {torch.__version__}')"
python3 -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
python3 -c "import flwr; print(f'Flower: {flwr.__version__}')"

echo ""
echo "=== Setup Complete ==="
echo ""
echo "To run experiments:"
echo "  source ~/.venv/bin/activate"
echo "  cd ~/cifar10/Fedge-Simulation/fedge"
echo "  SEED=42 python3 orchestrator.py"
echo ""

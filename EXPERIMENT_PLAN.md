# CIFAR-10 Federated Learning Experiments Plan

**Author:** Satwat Bashir
**Institution:** London South Bank University
**Date:** January 2026
**Purpose:** PhD Research - Hierarchical Federated Learning

---

## Model: ResNet-18 for CIFAR-10

**Model Selection Rationale:**
- ResNet-18 (~11.2M parameters) provides higher accuracy benchmarks than simple CNNs
- Widely used in FL research for CIFAR-10 (FedRAD, FedLC, etc.)
- CIFAR-10 adapted: 3x3 first conv (not 7x7), stride 1, no initial maxpool

**Literature Reference:** FedRAD (PMC), FedLC, FL-Simulator

---

## Methods to Compare

| # | Method | Description | Folder | Run Command |
|---|--------|-------------|--------|-------------|
| 1 | **FedProx** | Proximal regularization baseline | `fedprox/` | `cd fedprox && SEED=42 flwr run .` |
| 2 | **HierFL** | Hierarchical FL (3 servers x 5 clients) | `HierFL/fedge/` | `cd HierFL/fedge && SEED=42 python3 orchestrator.py` |
| 3 | **Fedge** | Hierarchical + SCAFFOLD + clustering | `fedge/` | `cd fedge && SEED=42 python3 orchestrator.py` |

---

## Experiment Plan (200 Rounds, 3 Seeds)

### Phase 1: Baselines

| Method | Rounds | Seeds | Status |
|--------|--------|-------|--------|
| FedProx | 200 | 3 (42, 43, 44) | To run |
| HierFL | 200 | 3 (42, 43, 44) | To run |

### Phase 2: Your Method (Fedge)

| Method | Rounds | Seeds | Status |
|--------|--------|-------|--------|
| Fedge | 200 | 3 (42, 43, 44) | To run after baselines |

**Total runs needed:** 9 runs (3 methods x 3 seeds)

---

## Priority Order

### Must Have (for publication):
1. **FedProx seed 42** - 200 rounds
2. **FedProx seed 43** - 200 rounds
3. **FedProx seed 44** - 200 rounds
4. **HierFL seed 42** - 200 rounds
5. **HierFL seed 43** - 200 rounds
6. **HierFL seed 44** - 200 rounds
7. **Fedge seed 42** - 200 rounds
8. **Fedge seed 43** - 200 rounds
9. **Fedge seed 44** - 200 rounds

### Why 3 Seeds?
- Sufficient for mean +/- std deviation
- Enables 95% confidence intervals
- Standard for ML conferences (NeurIPS, ICML, etc.)
- Some journals prefer 5 seeds, but 3 is acceptable

---

## Settings Summary

### Common Settings (All Methods) - ResNet-18

| Parameter | Value | Literature Reference |
|-----------|-------|---------------------|
| Dataset | CIFAR-10 (50K train, 10K test) | Standard |
| Classes | 10 | - |
| **Model** | **ResNet-18** | **~11.2M params** |
| **Batch Size** | **64** | NIID-Bench (ICDE 2022) |
| **Learning Rate** | **0.01** | NIID-Bench standard |
| **Local Epochs** | **5** | NIID-Bench standard |
| Optimizer | SGD | Standard |
| **Communication Rounds** | **200** | NIID-Bench reported accuracy |
| Fraction Fit | 1.0 (100% participation) | Required for SCAFFOLD |
| **Non-IID** | **Dirichlet alpha=0.5** | NIID-Bench standard |

### FedProx Specific

| Parameter | Value | Notes |
|-----------|-------|-------|
| Clients | 10 | NIID-Bench setting |
| Local Epochs | 5 | NIID-Bench standard |
| Batch Size | 64 | NIID-Bench standard |
| proximal_mu | 0.01 | Literature default |
| dirichlet_alpha | 0.5 | NIID-Bench standard |

### HierFL Specific

| Parameter | Value | Notes |
|-----------|-------|-------|
| Total Clients | 15 (3 servers x 5 clients) | Hierarchical |
| Leaf Servers | 3 | Cloud -> 3 Edge -> 15 Clients |
| Clients per Server | [5, 5, 5] | Balanced |
| Local Epochs | 5 | NIID-Bench standard |
| Batch Size | 64 | NIID-Bench standard |
| Server Rounds per Global | 1 | Sync after each local round |
| alpha_server | 0.5 | Non-IID across servers (heterogeneous regions) |
| alpha_client | 1000.0 | IID within each server (uniform split) |

### Fedge Specific (Your Method)

| Parameter | Value | Notes |
|-----------|-------|-------|
| Leaf Servers | 3 | Same as HierFL |
| Clients per Server | [5, 5, 5] | Same as HierFL |
| SCAFFOLD | Enabled | Server-side control variates |
| Cloud Clustering | Enabled | Dynamic clustering |
| prox_mu | 0.001 | Lower than FedProx |

---

## Expected Results (Based on NIID-Bench Literature)

### Target Accuracy Ranges (Dirichlet alpha=0.5, ResNet-18)

| Method | Expected Accuracy | Source |
|--------|-------------------|--------|
| FedAvg (baseline) | ~67% (simple-cnn) | NIID-Bench (ICDE 2022) |
| **FedProx (mu=0.01)** | **~66-68%** | NIID-Bench |
| SCAFFOLD | ~71% | NIID-Bench (full participation) |
| HierFL | TBD | Your experiments |
| Fedge | TBD | Your experiments |

*Note: NIID-Bench uses simple-cnn; ResNet-18 should achieve higher accuracy

### Key Benchmarks from Literature

| Paper | Setting | FedAvg | FedProx | Notes |
|-------|---------|--------|---------|-------|
| **NIID-Bench (ICDE 2022)** | 10 clients, 200 rounds, Dir 0.5 | **67.4%** | **66.4%** | simple-cnn, batch=64 |
| FedRAD (PMC) | 10 clients, 100 rounds, Dir 0.5 | 73.42% | 75.52% | ResNet-18, batch=128 |
| FedLC | 10 clients, 100 rounds | ~72% | ~74% | ResNet-18 |

### Comparison: Simple-CNN vs ResNet-18

| Model | Parameters | FedAvg (Dir 0.5) | FedProx (Dir 0.5) | Source |
|-------|------------|------------------|-------------------|--------|
| Simple-CNN | ~62K | 67.4% | 66.4% | NIID-Bench (200 rounds) |
| **ResNet-18** | **~11.2M** | **Higher expected** | **Higher expected** | **Larger model capacity** |

---

## Commands Summary

### FedProx (200 rounds, ResNet-18):
```bash
cd ~/cifar10/fedprox

# Seed 42
SEED=42 flwr run .
mkdir -p results_seed42 && mv metrics results_seed42/

# Seed 43
SEED=43 flwr run .
mkdir -p results_seed43 && mv metrics results_seed43/

# Seed 44
SEED=44 flwr run .
mkdir -p results_seed44 && mv metrics results_seed44/
```

### HierFL (200 rounds, ResNet-18):
```bash
cd ~/cifar10/HierFL/fedge

# Seed 42
rm -rf rounds runs signals metrics
SEED=42 python3 orchestrator.py
mkdir -p results_seed42 && mv metrics rounds runs results_seed42/

# Seed 43
rm -rf rounds runs signals metrics
SEED=43 python3 orchestrator.py
mkdir -p results_seed43 && mv metrics rounds runs results_seed43/

# Seed 44
rm -rf rounds runs signals metrics
SEED=44 python3 orchestrator.py
mkdir -p results_seed44 && mv metrics rounds runs results_seed44/
```

### Fedge (200 rounds, ResNet-18):
```bash
cd ~/cifar10/fedge

# Seed 42
rm -rf rounds runs signals metrics clusters
SEED=42 python3 orchestrator.py
mkdir -p results_seed42 && mv metrics rounds runs clusters results_seed42/

# (repeat for seeds 43, 44)
```

---

## Folder Structure for Results

```
CIFAR-10/
|-- fedprox/
|   |-- results_seed42/
|   |   +-- metrics/
|   |-- results_seed43/
|   +-- results_seed44/
|
|-- HierFL/fedge/
|   |-- results_seed42/
|   |   |-- metrics/
|   |   +-- rounds/
|   |-- results_seed43/
|   +-- results_seed44/
|
+-- fedge/
    |-- results_seed42/
    |-- results_seed43/
    +-- results_seed44/
```

---

## Metrics to Track

### Per Round
- `test_accuracy` - Global model test accuracy
- `test_loss` - Global model test loss
- `train_accuracy` - Training accuracy
- `train_loss` - Training loss
- `accuracy_gap` - train_acc - test_acc (overfitting indicator)

### Communication Metrics
- `bytes_uploaded` - Client -> Server bytes
- `bytes_downloaded` - Server -> Client bytes
- `total_communication` - Total bytes transferred

### Computation Metrics
- `round_time` - Time per round
- `training_time` - Client training time
- `aggregation_time` - Server aggregation time

---

## Configuration Files

| Method | Config File |
|--------|-------------|
| FedProx | `fedprox/pyproject.toml` |
| HierFL | `HierFL/fedge/pyproject.toml` |
| Fedge | `fedge/pyproject.toml` |

### Key Settings to Verify Before Running (ResNet-18)

```toml
# FedProx - fedprox/pyproject.toml
num-server-rounds = 200   # NIID-Bench: 200 rounds
fraction-fit = 1.0        # 100% participation
min_available_clients = 10
proximal_mu = 0.01        # Standard FedProx mu
local-epochs = 5          # NIID-Bench: 5 local epochs
batch_size = 64           # NIID-Bench: 64
learning_rate = 0.01      # NIID-Bench: 0.01
dirichlet_alpha = 0.5     # NIID-Bench standard
seed = 42

# HierFL - HierFL/fedge/pyproject.toml
global_rounds = 200       # NIID-Bench: 200 rounds
local-epochs = 5          # NIID-Bench: 5 local epochs
batch_size = 64           # NIID-Bench: 64
learning_rate = 0.01      # NIID-Bench: 0.01
alpha_server = 0.5        # Non-IID across servers
alpha_client = 1000.0     # IID within each server
num_servers = 3
clients_per_server = [5,5,5]
```

---

## Comparison with HHAR Experiments

| Aspect | HHAR | CIFAR-10 |
|--------|------|----------|
| Dataset | HAR sensor data | Image classification |
| Non-IID Source | Natural (user-based) | Synthetic (Dirichlet) |
| Clients | 9 | 10-15 |
| **Model** | 1D-CNN (6 channels) | **ResNet-18 (3 channels)** |
| **Model Params** | ~100K | **~11.2M** |
| Batch Size | 32-64 | **64** |
| Local Epochs | 5 | 5 |
| Learning Rate | 0.05 | **0.01** |
| **Rounds** | 100-200 | **200** |
| **Expected Accuracy** | 60-80% | **73-76%** |

---

## Reproducibility Checklist

Before each run:
- [ ] Set correct SEED environment variable
- [ ] Clear previous run artifacts (`rm -rf rounds runs signals metrics`)
- [ ] Verify pyproject.toml settings match plan
- [ ] Check GPU/CPU availability
- [ ] Ensure data folder exists (CIFAR-10 will auto-download)

After each run:
- [ ] Save results to `results_seedX/` folder
- [ ] Verify metrics files were generated
- [ ] Record any errors or warnings
- [ ] Update status in this document

---

## Results Template (Fill After Experiments)

### Final Results Table (ResNet-18, 200 rounds, Dir alpha=0.5)

| Method | Seed 42 | Seed 43 | Seed 44 | Mean +/- Std |
|--------|---------|---------|---------|--------------|
| FedProx | TBD | TBD | TBD | Expected: ~70%+ |
| HierFL | TBD | TBD | TBD | TBD |
| Fedge | TBD | TBD | TBD | TBD |

### Convergence Analysis

| Method | Rounds to 60% | Rounds to 70% | Final Accuracy |
|--------|---------------|---------------|----------------|
| FedProx | TBD | TBD | ~70%+ expected |
| HierFL | TBD | TBD | TBD |
| Fedge | TBD | TBD | TBD |

---

*Document prepared for PhD research at London South Bank University*
*Last updated: January 2026*
*Model: ResNet-18 (~11.2M params) | Settings from NIID-Bench (ICDE 2022)*

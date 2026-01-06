# CIFAR-10 Federated Learning Experiments Plan

**Author:** Satwat Bashir
**Institution:** London South Bank University
**Date:** January 2026
**Purpose:** PhD Research - Hierarchical Federated Learning

---

## Methods to Compare

| # | Method | Description | Folder | Run Command |
|---|--------|-------------|--------|-------------|
| 1 | **FedProx** | Proximal regularization baseline | `fedprox/` | `cd fedprox && SEED=42 flwr run .` |
| 2 | **HierFL** | Hierarchical FL (3 servers × 5 clients) | `HierFL/fedge/` | `cd HierFL/fedge && SEED=42 python3 orchestrator.py` |
| 3 | **Fedge** | Hierarchical + SCAFFOLD + clustering | `fedge/` | `cd fedge && SEED=42 python3 orchestrator.py` |

---

## Experiment Plan (150 Rounds, 3 Seeds)

### Phase 1: Baselines

| Method | Rounds | Seeds | Status |
|--------|--------|-------|--------|
| FedProx | 150 | 3 (42, 43, 44) | To run |
| HierFL | 150 | 3 (42, 43, 44) | To run |

### Phase 2: Your Method (Fedge)

| Method | Rounds | Seeds | Status |
|--------|--------|-------|--------|
| Fedge | 150 | 3 (42, 43, 44) | To run after baselines |

**Total runs needed:** 9 runs (3 methods × 3 seeds)

---

## Priority Order

### Must Have (for publication):
1. **FedProx seed 42** - 150 rounds
2. **FedProx seed 43** - 150 rounds
3. **FedProx seed 44** - 150 rounds
4. **HierFL seed 42** - 150 rounds
5. **HierFL seed 43** - 150 rounds
6. **HierFL seed 44** - 150 rounds
7. **Fedge seed 42** - 150 rounds
8. **Fedge seed 43** - 150 rounds
9. **Fedge seed 44** - 150 rounds

### Why 3 Seeds?
- Sufficient for mean ± std deviation
- Enables 95% confidence intervals
- Standard for ML conferences (NeurIPS, ICML, etc.)
- Some journals prefer 5 seeds, but 3 is acceptable

---

## Settings Summary

### Common Settings (All Methods)

| Parameter | Value | Literature Reference |
|-----------|-------|---------------------|
| Dataset | CIFAR-10 (50K train, 10K test) | Standard |
| Classes | 10 | - |
| Model | LeNet-5 / Simple CNN | ~62K params |
| **Batch Size** | **64** | NIID-Bench, MOON |
| **Learning Rate** | **0.01** | NIID-Bench standard |
| **Local Epochs** | **5** | NIID-Bench (FedProx paper: 5-20) |
| Optimizer | SGD | Standard |
| Communication Rounds | 150 | Literature: 50-200 |
| Fraction Fit | 1.0 (100% participation) | Required for SCAFFOLD |
| Non-IID | Dirichlet α=0.5 | NIID-Bench standard |

### FedProx Specific

| Parameter | Value | Notes |
|-----------|-------|-------|
| Clients | 10 | Matches NIID-Bench |
| **Local Epochs** | **5** | NIID-Bench standard |
| **Batch Size** | **64** | Literature standard |
| proximal_mu | 0.01 | Literature default |
| dirichlet_alpha | 0.5 | Standard benchmark |

### HierFL Specific

| Parameter | Value | Notes |
|-----------|-------|-------|
| Total Clients | 15 (3 servers × 5 clients) | Hierarchical |
| Leaf Servers | 3 | Cloud → 3 Edge → 15 Clients |
| Clients per Server | [5, 5, 5] | Balanced |
| **Local Epochs** | **5** | NIID-Bench standard |
| **Batch Size** | **64** | Literature standard |
| Server Rounds per Global | 1 | Sync after each local round |
| alpha_server | 0.5 | Outer Dirichlet split |
| alpha_client | 0.3 | Inner Dirichlet split (more non-IID) |

### Fedge Specific (Your Method)

| Parameter | Value | Notes |
|-----------|-------|-------|
| Leaf Servers | 3 | Same as HierFL |
| Clients per Server | [5, 5, 5] | Same as HierFL |
| SCAFFOLD | Enabled | Server-side control variates |
| Cloud Clustering | Enabled | Dynamic clustering |
| prox_mu | 0.001 | Lower than FedProx |

---

## Expected Results (Based on Literature)

### Target Accuracy Ranges (Dirichlet α=0.5)

| Method | Expected Accuracy | Source |
|--------|-------------------|--------|
| FedAvg (baseline) | 65-70% | NIID-Bench, MOON |
| FedProx (μ=0.01) | 68-72% | MOON paper |
| SCAFFOLD | 70-76% | ICML 2020 (full participation) |
| MOON | 70-74% | CVPR 2021 |
| HierFL | TBD | Your experiments |
| Fedge | TBD | Your experiments |

### Key Benchmarks from Literature

| Paper | Setting | FedAvg | FedProx | SCAFFOLD |
|-------|---------|--------|---------|----------|
| NIID-Bench | 10 clients, 50 rounds, Dir 0.5 | 65.91% | ~66% | ~70% |
| MOON | 10 clients, 100 rounds, Dir 0.5 | 65.8% | 68.5% | - |
| Flower Baselines | 100 rounds | - | 68.52% | - |

---

## Commands Summary

### FedProx (150 rounds):
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

### HierFL (150 rounds):
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

### Fedge (150 rounds):
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
├── fedprox/
│   ├── results_seed42/
│   │   └── metrics/
│   ├── results_seed43/
│   └── results_seed44/
│
├── HierFL/fedge/
│   ├── results_seed42/
│   │   ├── metrics/
│   │   └── rounds/
│   ├── results_seed43/
│   └── results_seed44/
│
└── fedge/
    ├── results_seed42/
    ├── results_seed43/
    └── results_seed44/
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
- `bytes_uploaded` - Client → Server bytes
- `bytes_downloaded` - Server → Client bytes
- `total_communication` - Total bytes transferred

### Computation Metrics
- `round_time` - Time per round
- `training_time` - Client training time
- `aggregation_time` - Server aggregation time

---

## Estimated Run Times

| Method | Time per Seed | Total (3 seeds) |
|--------|---------------|-----------------|
| FedProx (150 rounds) | ~4-6 hrs | ~12-18 hrs |
| HierFL (150 rounds) | ~8-12 hrs | ~24-36 hrs |
| Fedge (150 rounds) | ~8-12 hrs | ~24-36 hrs |

**Total for all experiments:** ~60-90 hours

---

## Configuration Files

| Method | Config File |
|--------|-------------|
| FedProx | `fedprox/pyproject.toml` |
| HierFL | `HierFL/fedge/pyproject.toml` |
| Fedge | `fedge/pyproject.toml` |

### Key Settings to Verify Before Running

```toml
# FedProx - fedprox/pyproject.toml
num-server-rounds = 150
local-epochs = 5          # CORRECTED: was 1
batch_size = 64           # CORRECTED: was 32
learning_rate = 0.01
proximal_mu = 0.01
dirichlet_alpha = 0.5

# HierFL - HierFL/fedge/pyproject.toml
global_rounds = 150
local-epochs = 5          # CORRECTED: added
batch_size = 64           # CORRECTED: was 32
learning_rate = 0.01      # CORRECTED: added
alpha_server = 0.5
alpha_client = 0.3
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
| Model | 1D-CNN (6 channels) | LeNet-5 (3 channels) |
| **Batch Size** | 32-64 | **64** |
| **Local Epochs** | 5 | **5** |
| **Learning Rate** | 0.05 | **0.01** |
| Rounds | 100-200 | 150 |
| Expected Accuracy | 60-80% | 65-75% |

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

### Final Results Table

| Method | Seed 42 | Seed 43 | Seed 44 | Mean ± Std |
|--------|---------|---------|---------|------------|
| FedProx | TBD | TBD | TBD | TBD |
| HierFL | TBD | TBD | TBD | TBD |
| Fedge | TBD | TBD | TBD | TBD |

### Convergence Analysis

| Method | Rounds to 60% | Rounds to 65% | Final Accuracy |
|--------|---------------|---------------|----------------|
| FedProx | TBD | TBD | TBD |
| HierFL | TBD | TBD | TBD |
| Fedge | TBD | TBD | TBD |

---

*Document prepared for PhD research at London South Bank University*
*Last updated: January 2026*

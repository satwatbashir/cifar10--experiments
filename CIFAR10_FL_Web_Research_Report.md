# CIFAR-10 Federated Learning: Web Research Report

**Author:** Claude Code (AI Assistant)
**Date:** January 2026
**Purpose:** Comprehensive analysis of CIFAR-10 FL accuracy achievements from recent literature

---

## Executive Summary

Based on web research of recent publications (2024-2025), this report summarizes which methods, settings, and configurations achieve the **highest accuracies** on CIFAR-10 in federated learning scenarios.

### Key Findings

| Factor | Best Practice | Expected Accuracy |
|--------|---------------|-------------------|
| **Best Method (2024)** | FedFSE, FedAF | 95.50%, +25% over baselines |
| **Reliable Baseline** | FedProx (μ=0.01) | 68-72% (Dir α=0.5) |
| **Best for Full Participation** | SCAFFOLD | 70-76% |
| **Best for Non-IID** | FedClust, MOON | +45% over FedAvg |
| **Standard Dirichlet α** | 0.5 | Moderate non-IID |
| **Extreme Non-IID α** | 0.1 or 0.03 | Severe heterogeneity |

---

## 1. Methods Achieving Highest Accuracy

### 1.1 State-of-the-Art (2024-2025)

| Method | CIFAR-10 Accuracy | Improvement | Source |
|--------|-------------------|-------------|--------|
| **FedFSE** | **95.50%** | +6-8% over FedAvg | [Wiley 2024](https://onlinelibrary.wiley.com/doi/full/10.1155/2024/8860376) |
| **FedAF** | +25.44% over SOTA | 80% faster convergence | [CVPR 2024](https://openaccess.thecvf.com/content/CVPR2024/papers/Wang_An_Aggregation-Free_Federated_Learning_for_Tackling_Data_Heterogeneity_CVPR_2024_paper.pdf) |
| **FedClust** | +45.5% over FedAvg | Best for label skew | [ACM 2024](https://dl.acm.org/doi/fullHtml/10.1145/3673038.3673151) |
| **MOON** | 70.7% (Dir 0.5) | +5% over FedProx | [Flower Baselines](https://flower.ai/docs/baselines/moon.html) |
| **CCVR** | New SOTA | Classifier calibration | [NeurIPS 2021](https://proceedings.neurips.cc/paper/2021/file/2f2b265625d76a6704b08093c652fd79-Paper.pdf) |

### 1.2 Classic Methods Comparison (Dir α=0.5, 100 rounds)

| Method | Accuracy | Notes |
|--------|----------|-------|
| FedAvg | 65.8% | Baseline |
| FedProx (μ=0.01) | 68.5% | +2.7% over FedAvg |
| SCAFFOLD | 70% | **Requires full participation** |
| MOON | 70.7% | Best classic method |
| FedNova | ~66% | Similar to FedProx |

Source: [NIID-Bench](https://github.com/Xtra-Computing/NIID-Bench), [arXiv Performance Study](https://arxiv.org/html/2403.17287v1)

---

## 2. Accuracy by Non-IID Level (Dirichlet α)

### 2.1 Dirichlet Alpha Effects

| α Value | Description | FedAvg Acc | FedProx Acc | Best Method |
|---------|-------------|------------|-------------|-------------|
| **100** | IID (uniform) | 82-85% | 82-85% | Any |
| **1.0** | Mild non-IID | 78-82% | 78-82% | FedAvg/FedProx |
| **0.5** | **Standard benchmark** | 65-70% | 68-72% | MOON, SCAFFOLD |
| **0.3** | High non-IID | 55-65% | 60-68% | FedClust, pFedMe |
| **0.1** | Severe non-IID | 45-55% | 50-60% | CFL, FedClust |
| **0.03** | Extreme (~2 classes/client) | 20-35% | 25-40% | Clustering methods |

Source: [PMC FedRAD Study](https://pmc.ncbi.nlm.nih.gov/articles/PMC10385861/), [arXiv Non-IID Study](https://arxiv.org/pdf/1806.00582)

### 2.2 Key Insight
> "As the degree of data heterogeneity decreases (i.e., α increases), the accuracy of all methods tends to increase."

---

## 3. Scale: Small vs Large Experiments

### 3.1 Client Scale Comparison

| Scale | Clients | Participation | Typical Accuracy | Notes |
|-------|---------|---------------|------------------|-------|
| **Small** | 10 | 100% (fraction=1.0) | 65-72% | NIID-Bench standard |
| **Medium** | 50 | 100% or 20% | 60-68% | Common research setup |
| **Large** | 100 | 10-15% | 55-65% | Production-like |
| **Very Large** | 500+ | 5% | 50-60% | Scaling studies |

Source: [arXiv Client Size Study](https://arxiv.org/html/2504.08198v1)

### 3.2 Impact of Client Count

| Study | Finding |
|-------|---------|
| Song et al. | Accuracy **decreases** as clients increase from 1→100 (IID) or 10→100 (non-IID) |
| Xu et al. | Accuracy **drops** from 100→500 clients |
| Li et al. | 50 clients (100% participation) **outperforms** 100 clients (20% participation) |

**Recommendation:** For benchmarking, use **10 clients with 100% participation** to match literature standards.

---

## 4. Fraction Fit (Participation Rate)

### 4.1 Participation Rate Effects

| Participation | Setting | Effect on Accuracy | Effect on Convergence |
|---------------|---------|-------------------|----------------------|
| **100%** | fraction_fit=1.0 | Best accuracy | Fastest convergence |
| **50%** | fraction_fit=0.5 | -2-5% accuracy | 1.5x more rounds |
| **20%** | fraction_fit=0.2 | -5-10% accuracy | 2-3x more rounds |
| **10%** | fraction_fit=0.1 | -10-15% accuracy | 3-5x more rounds |

### 4.2 Method-Specific Requirements

| Method | Min Participation | Notes |
|--------|-------------------|-------|
| **SCAFFOLD** | **100% required** | Diverges with partial participation |
| FedProx | Any | Works with any participation |
| FedAvg | Any | Performance degrades gracefully |
| MOON | 100% recommended | Tested with full participation |

Source: [ICML SCAFFOLD Paper](https://proceedings.mlr.press/), [arXiv Study](https://arxiv.org/html/2406.06340)

### 4.3 Critical Warning
> "SCAFFOLD cannot work effectively in partial participation settings" - NIID-Bench

---

## 5. Models and Architectures

### 5.1 Model Comparison on CIFAR-10

| Model | Parameters | FL Accuracy (Dir 0.5) | Centralized Accuracy |
|-------|------------|----------------------|---------------------|
| Simple-CNN | ~62K | 65-70% | 75-80% |
| LeNet-5 | ~62K | 65-70% | 75-80% |
| VGG-9 | ~3M | 75-82% | 85-88% |
| VGG-11 | ~9M | 78-85% | 88-90% |
| **ResNet-18** | ~11M | **80-88%** | **92-94%** |
| ResNet-50 | ~25M | 82-90% | 93-95% |

### 5.2 Recommendations by Use Case

| Use Case | Recommended Model | Why |
|----------|-------------------|-----|
| **Benchmarking** | Simple-CNN / LeNet-5 | Matches NIID-Bench |
| **Fair Comparison** | ResNet-18 | Common in 2024 papers |
| **Best Accuracy** | ResNet-50 | Highest capacity |
| **Resource-Limited** | Simple-CNN | Fastest training |

Source: [ResearchGate VGG/ResNet Study](https://www.researchgate.net/figure/CIFAR10-accuracy-against-the-number-of-network-parameters-for-the-VGG-and-ResNet-models_fig3_366240609)

---

## 6. Communication Rounds

### 6.1 Rounds to Convergence

| Setting | FedAvg | FedProx | SCAFFOLD | MOON |
|---------|--------|---------|----------|------|
| IID, 10 clients | 50-100 | 50-100 | 30-50 | 50-80 |
| Dir 0.5, 10 clients | 100-200 | 100-200 | 80-150 | 80-100 |
| Dir 0.1, 10 clients | 200-500 | 200-400 | Unstable | 150-200 |
| 100 clients, 10% part. | 500-1000 | 400-800 | N/A | 300-500 |

### 6.2 Typical Experiment Settings

| Paper/Benchmark | Rounds | Clients | Participation |
|-----------------|--------|---------|---------------|
| NIID-Bench | 50 | 10 | 100% |
| MOON | 100 | 10 | 100% |
| FedProx | 200 | 10 | 100% |
| SCAFFOLD | 800 | 10 | **100% required** |
| pFedMe | 600-800 | 5-20 | 100% |
| FedDyn | 5000 | 100 | 5% |

---

## 7. Optimal Settings for High Accuracy

### 7.1 Recommended Configuration (Matching Literature)

```toml
# For NIID-Bench comparison (Standard)
clients = 10
batch_size = 64
learning_rate = 0.01
local_epochs = 5
rounds = 100-200
dirichlet_alpha = 0.5
fraction_fit = 1.0  # 100% participation
model = "simple-cnn"  # or ResNet-18 for higher accuracy
```

### 7.2 For Maximum Accuracy

```toml
# For highest accuracy benchmarks
clients = 10-50
batch_size = 64
learning_rate = 0.01
local_epochs = 10
rounds = 200-500
dirichlet_alpha = 0.5
fraction_fit = 1.0
model = "resnet18"
method = "MOON" or "FedClust"
```

### 7.3 For Hierarchical FL (Your Research)

```toml
# Hierarchical FL settings
total_clients = 15  # 3 servers × 5 clients
batch_size = 64
learning_rate = 0.01
local_epochs = 5
global_rounds = 150
alpha_server = 0.5
alpha_client = 0.3  # More non-IID at client level
fraction_fit = 1.0
```

---

## 8. Summary Tables

### 8.1 Methods Ranked by Accuracy (Dir α=0.5)

| Rank | Method | Accuracy | Computation | Communication |
|------|--------|----------|-------------|---------------|
| 1 | FedFSE (2024) | 95.5% | High | Medium |
| 2 | FedAF (2024) | +25% SOTA | Medium | Low |
| 3 | FedClust | +45% FedAvg | Medium | Medium |
| 4 | MOON | 70.7% | Medium | Medium |
| 5 | SCAFFOLD | 70% | Low | **High (2x)** |
| 6 | FedProx | 68.5% | Low | Low |
| 7 | FedAvg | 65.8% | **Lowest** | **Lowest** |

### 8.2 Settings Impact on Accuracy

| Factor | Low Setting | High Setting | Accuracy Impact |
|--------|-------------|--------------|-----------------|
| Clients | 10 | 100 | -5-10% |
| Participation | 10% | 100% | -10-15% |
| Dirichlet α | 0.1 | 0.5 | -15-20% |
| Local Epochs | 1 | 5 | +5-10% |
| Batch Size | 32 | 64 | +2-3% |
| Rounds | 50 | 200 | +5-10% |

---

## 9. Key Takeaways for Your Research

1. **Your FedProx settings are now correct** - 5 local epochs, batch 64, matches NIID-Bench

2. **Expected accuracy range**: 65-72% for FedProx with Dir α=0.5

3. **HierFL advantage**: Your hierarchical structure with α_client=0.3 creates more non-IID scenario, which should show clearer benefits of your method

4. **Keep 100% participation** (fraction_fit=1.0) - Required for SCAFFOLD comparisons

5. **150 rounds is sufficient** - Most papers use 100-200 rounds

6. **No single best algorithm** - Performance varies by data heterogeneity level

---

## Sources

- [NIID-Bench GitHub](https://github.com/Xtra-Computing/NIID-Bench)
- [Performance Evaluation Study (arXiv 2024)](https://arxiv.org/html/2403.17287v1)
- [FedClust (ACM 2024)](https://dl.acm.org/doi/fullHtml/10.1145/3673038.3673151)
- [FedAF CVPR 2024](https://openaccess.thecvf.com/content/CVPR2024/papers/Wang_An_Aggregation-Free_Federated_Learning_for_Tackling_Data_Heterogeneity_CVPR_2024_paper.pdf)
- [FedFSE (Wiley 2024)](https://onlinelibrary.wiley.com/doi/full/10.1155/2024/8860376)
- [MOON Flower Baselines](https://flower.ai/docs/baselines/moon.html)
- [FedRAD Study (PMC)](https://pmc.ncbi.nlm.nih.gov/articles/PMC10385861/)
- [Client Size Study (arXiv 2025)](https://arxiv.org/html/2504.08198v1)
- [ProFed Benchmark (arXiv 2025)](https://arxiv.org/html/2503.20618v1)

---

*Report generated from web research, January 2026*

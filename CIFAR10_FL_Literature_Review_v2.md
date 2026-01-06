# CIFAR-10 Dataset: Literature Review for Federated Learning Research

**Author:** Satwat Bashir  
**Institution:** London South Bank University  
**Date:** January 2026  
**Purpose:** PhD Research - Hierarchical Federated Learning

---

## 1. Dataset Overview

### 1.1 Dataset Characteristics

| Property | Value |
|----------|-------|
| **Full Name** | Canadian Institute for Advanced Research (CIFAR-10) |
| **Images** | 60,000 (50,000 train, 10,000 test) |
| **Resolution** | 32×32 RGB (3 channels) |
| **Classes** | 10 (airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck) |
| **Images per Class** | 6,000 (5,000 train, 1,000 test) |
| **Source** | Alex Krizhevsky, 2009 |

### 1.2 Why CIFAR-10 for FL Research

CIFAR-10 is the **de facto standard** for FL benchmarking because:
- Large enough for meaningful evaluation, small enough for rapid iteration
- Complex enough to show algorithm differences (unlike MNIST)
- No natural user partitioning → requires synthetic non-IID strategies
- Extensively benchmarked across all major FL papers

---

## 2. Non-IID Partitioning Strategies

### 2.1 Dirichlet Distribution (Most Common)

The Dirichlet distribution `Dir(α)` controls label heterogeneity across clients.

| α Value | Heterogeneity | Expected Accuracy | Description |
|---------|---------------|-------------------|-------------|
| **0.1** | Extreme | 45-60% | Most labels concentrated in few clients |
| **0.3** | High | 55-70% | Strong label imbalance |
| **0.5** | Moderate | 65-75% | **Standard benchmark setting** |
| **1.0** | Mild | 78-82% | Nearly uniform distribution |
| **∞** | IID | 85-87% | Uniform label distribution |

### 2.2 Pathological Partitioning (#C Classes per Client)

| Setting | Description | Expected Accuracy |
|---------|-------------|-------------------|
| **#C = 1** | Each client has only 1 class | 20-35% (extreme) |
| **#C = 2** | Each client has 2 classes | 50-60% (severe) |
| **#C = 3** | Each client has 3 classes | 60-70% (high) |
| **#C = 5** | Each client has 5 classes | 75-80% (moderate) |

---

## 3. Summary of Key Papers on CIFAR-10

| Paper | Year | Methods | Clients | Rounds | Best Accuracy | Key Contribution |
|-------|------|---------|---------|--------|---------------|------------------|
| McMahan et al. | 2017 | FedAvg | 100 | - | 85-87% (IID) | Original FedAvg |
| Zhao et al. | 2018 | FedAvg | 10 | 500 | 51% (1-class non-IID) | Non-IID analysis |
| Li et al. (FedProx) | 2020 | FedProx | 10 | 200 | +22% over FedAvg | Proximal term |
| Karimireddy et al. | 2020 | SCAFFOLD | 10 | 800 | Best with full participation | Control variates |
| Li et al. (MOON) | 2021 | MOON | 10 | 100 | 70.7% (Dir 0.5) | Model-contrastive |
| Li et al. (NIID-Bench) | 2022 | FedAvg/FedProx/SCAFFOLD/FedNova | 10 | 50 | ~66% (Dir 0.5) | Comprehensive benchmark |
| Dinh et al. | 2020 | pFedMe | 5-20 | 800 | +1-6% personalized | Moreau envelopes |

---

## 4. Detailed Paper-by-Paper Experimental Settings

### 4.1 McMahan et al. (2017) - FedAvg Original Paper

**Paper:** "Communication-Efficient Learning of Deep Networks from Decentralized Data" - AISTATS 2017

| Parameter | Value |
|-----------|-------|
| **Dataset** | CIFAR-10 |
| **Clients** | 100 |
| **Data per Client** | 500 train, 100 test |
| **Partitioning** | IID (balanced) |
| **Batch Size (B)** | 50 |
| **Local Epochs (E)** | 5 |
| **Client Fraction (C)** | 0.1 (10 clients/round) |
| **Learning Rate** | Optimized per setting |
| **Model** | CNN (~10⁶ params) |

**Key Results:**
- FedAvg with E=5, B=50 reduces communication by 10-100× vs FedSGD
- Achieved same accuracy as centralized SGD with fewer communication rounds

---

### 4.2 Zhao et al. (2018) - Non-IID Data Analysis

**Paper:** "Federated Learning with Non-IID Data" - arXiv

| Parameter | Value |
|-----------|-------|
| **Dataset** | CIFAR-10 |
| **Clients** | 10 |
| **Partitioning** | 1-class non-IID, 2-class non-IID |
| **Batch Size** | 10 and 100 |
| **Local Epochs** | 1 and 5 |
| **Communication Rounds** | 500 |
| **Learning Rate** | 0.1 |
| **LR Decay Rate** | 0.992 |
| **Model** | CNN |

**Non-IID Settings:**

| Setting | Description |
|---------|-------------|
| 1-class non-IID | Each client has data from only 1 class |
| 2-class non-IID | Each client has data from 2 classes |

**Key Results:**

| Setting | SGD (Centralized) | FedAvg | Accuracy Drop |
|---------|-------------------|--------|---------------|
| IID | 88% | 85-87% | ~2% |
| 2-class non-IID | 88% | ~75% | ~13% |
| 1-class non-IID | 88% | **51%** | **37%** |

**Critical Finding:** Up to 37% accuracy drop for extreme non-IID on CIFAR-10.

---

### 4.3 Li et al. (2020) - FedProx

**Paper:** "Federated Optimization in Heterogeneous Networks" - MLSys 2020

| Parameter | Value |
|-----------|-------|
| **Dataset** | Synthetic, MNIST, FEMNIST, Shakespeare, Sent140 |
| **Learning Rate** | 0.01 |
| **Communication Rounds** | 200 |
| **Clients per Round** | 10 |
| **Batch Size** | 10 |
| **Local Epochs** | 20 |
| **μ (Proximal Term)** | Tuned from {0.001, 0.01, 0.1, 0.5, 1.0} |

**FedProx μ Selection Guide:**

| μ Value | When to Use |
|---------|-------------|
| 0.001 | Mild non-IID |
| **0.01** | **Moderate non-IID (default)** |
| 0.1 | High non-IID |
| 0.5-1.0 | Extreme non-IID |

**Key Results:**
- Up to **22% improvement** in absolute test accuracy over FedAvg in highly heterogeneous settings
- More stable convergence behavior

---

### 4.4 Karimireddy et al. (2020) - SCAFFOLD

**Paper:** "SCAFFOLD: Stochastic Controlled Averaging for Federated Learning" - ICML 2020

| Parameter | Value |
|-----------|-------|
| **Global Learning Rate** | 1.0 |
| **Local Learning Rate** | 0.01-0.03 |
| **Local Epochs** | 5-20 |
| **Communication Rounds** | Up to 800 |

**⚠️ CRITICAL LIMITATION:**
> "SCAFFOLD cannot work effectively in partial participation settings" - NIID-Bench

| Participation | SCAFFOLD Performance |
|---------------|---------------------|
| Full (100%) | **Best performance** |
| Partial (<100%) | May diverge or underperform |

**Key Results:**
- Requires **fraction-fit = 1.0** for optimal performance
- Up to 45.5% improvement over FedAvg with full participation

---

### 4.5 Li et al. (2021) - MOON

**Paper:** "Model-Contrastive Federated Learning" - CVPR 2021

| Parameter | Value |
|-----------|-------|
| **Dataset** | CIFAR-10, CIFAR-100, Tiny-ImageNet |
| **Clients** | 10 (sample ratio = 1.0) |
| **Partitioning** | Dirichlet (β = 0.5) |
| **Batch Size** | 64 |
| **Local Epochs** | 10 |
| **Communication Rounds** | 100 |
| **Learning Rate** | 0.01 |
| **Model** | Simple-CNN, ResNet-50 |

**MOON-Specific Parameters:**

| Parameter | Value | Tuning Range |
|-----------|-------|--------------|
| **μ (contrastive weight)** | 5 (CIFAR-10), 10 (CIFAR-100) | {0.001, 0.01, 0.1, 1, 5, 10} |
| **Temperature (τ)** | 0.5 | {0.1, 0.5, 1.0} |
| **Projection Head Dim** | 256 | {64, 128, 256} |

**Key Results (Dir β=0.5, 100 rounds):**

| Method | CIFAR-10 | CIFAR-100 |
|--------|----------|-----------|
| FedAvg | 65.8% | 41.5% |
| FedProx | 68.5% | 43.2% |
| **MOON** | **70.7%** | **48.6%** |

**Improvement:** +5% over FedProx, +2% minimum over existing methods

---

### 4.6 Li et al. (2022) - NIID-Bench

**Paper:** "Federated Learning on Non-IID Data Silos: An Experimental Study" - ICDE 2022

**This is the most comprehensive FL benchmark paper.**

| Parameter | Value |
|-----------|-------|
| **Datasets** | MNIST, CIFAR-10, Fashion-MNIST, SVHN, FEMNIST, 3 tabular |
| **Clients** | 10 |
| **Sample Rate** | 1.0 (full participation) |
| **Batch Size** | 64 |
| **Local Epochs** | 5 |
| **Communication Rounds** | 50 |
| **Learning Rate** | 0.01 |
| **FedProx μ** | 0.001 |
| **Momentum** | 0 |

**Model Architecture (NIID-Bench CNN):**
```
Conv2D(in=3, out=6, kernel=5×5) → MaxPool(2×2)
Conv2D(in=6, out=16, kernel=5×5) → MaxPool(2×2)
FC(16×5×5 → 120, ReLU) → FC(120 → 84, ReLU) → Output(10)
```

**Partitioning Strategies:**

| Strategy | Description |
|----------|-------------|
| `homo` | IID partition |
| `noniid-labeldir` | Dirichlet distribution (α=0.5 default) |
| `noniid-#label1` | 1 class per client |
| `noniid-#label2` | 2 classes per client |
| `noniid-#label3` | 3 classes per client |
| `iid-diff-quantity` | IID labels, different quantities |

**Key Results (CIFAR-10):**

| Method | IID | Dir(0.5) | #C=3 | #C=2 | #C=1 |
|--------|-----|----------|------|------|------|
| FedAvg | ~85% | 65.91% | ~60% | ~50% | ~25% |
| FedProx | ~85% | ~66% | ~60% | ~50% | ~25% |
| SCAFFOLD | ~85% | ~70% | **Unstable** | **Unstable** | **Diverges** |
| FedNova | ~85% | ~66% | ~60% | ~50% | ~25% |

**Critical Finding:**
> "None of the existing state-of-the-art FL algorithms outperforms others in all cases."

---

### 4.7 Dinh et al. (2020) - pFedMe

**Paper:** "Personalized Federated Learning with Moreau Envelopes" - NeurIPS 2020

| Parameter | MNIST | Synthetic |
|-----------|-------|-----------|
| **Clients** | 5 | 10 |
| **Batch Size** | 20 | 20 |
| **Learning Rate** | 0.005 | 0.01 |
| **Personal Learning Rate** | 0.1 | 0.01 |
| **β (Server Mixing)** | 1.0 | 2.0 |
| **λ (Moreau Envelope)** | 15.0 | 30.0 |
| **K (Inner Steps)** | 5 | 5 |
| **Communication Rounds** | 800 | 600 |
| **Local Epochs** | 20 | 20 |

**pFedMe Hyperparameter Guide:**

| Parameter | Description | Typical Range |
|-----------|-------------|---------------|
| **λ (lambda)** | Moreau envelope regularization | 1.0-30.0 |
| **K** | Inner optimization steps | 5 |
| **β** | Server mixing parameter | 1.0-2.0 |
| **inner_lr** | Personal model learning rate | 0.01-0.1 |
| **outer_lr** | Global model learning rate | 0.005-0.01 |

**Key Results:**
- Personalized model 1.1-6.1% more accurate than global model
- Outperforms FedAvg and Per-FedAvg in non-IID settings

---

### 4.8 Acar et al. (2021) - FedDyn

**Paper:** "Federated Learning Based on Dynamic Regularization" - ICLR 2021

| Parameter | CIFAR-10 |
|-----------|----------|
| **Clients** | 100 |
| **Batch Size** | 50 |
| **Learning Rate** | 0.1 |
| **Local Epochs** | 5 |
| **Weight Decay** | 10⁻³ |
| **LR Decay** | 0.992-0.998 per round |
| **α (FedDyn)** | 10⁻³ to 10⁻¹ |
| **μ (FedProx)** | 10⁻² to 10⁻⁴ |

**Comparison Settings:**

| Method | LR | Epochs | Special Params |
|--------|-----|--------|----------------|
| FedAvg | 0.1 | 5 | - |
| FedProx | 0.1 | 5 | μ = 10⁻³ to 10⁻⁴ |
| SCAFFOLD | 0.1 | - | K = 500 (equivalent steps) |
| FedDyn | 0.1 | 5 | α = 10⁻³ to 10⁻¹ |

**Key Results (CIFAR-10):**
- FedDyn achieves 85.2% test accuracy
- Significantly faster convergence than baselines

---

## 5. Model Architectures

### 5.1 Simple CNN (Most Common Baseline)

**McMahan et al. (2017) CNN:**
```
Input: (B, 3, 32, 32)
├── Conv2D(3→32, 5×5, padding=2) → ReLU → MaxPool(2×2)
├── Conv2D(32→64, 5×5, padding=2) → ReLU → MaxPool(2×2)
├── FC(64×8×8 → 512) → ReLU
└── FC(512 → 10)
Parameters: ~580K
Expected IID Accuracy: 70-75%
```

### 5.2 NIID-Bench CNN

```
Input: (B, 3, 32, 32)
├── Conv2D(3→6, 5×5) → MaxPool(2×2)
├── Conv2D(6→16, 5×5) → MaxPool(2×2)
├── FC(16×5×5 → 120) → ReLU
├── FC(120 → 84) → ReLU
└── FC(84 → 10)
Parameters: ~62K
```

### 5.3 VGG Networks

| Model | Parameters | Expected Accuracy |
|-------|------------|-------------------|
| VGG-9 | ~3M | 80-85% |
| VGG-11 | ~9M | 82-88% |

### 5.4 ResNet

| Model | Parameters | Expected Accuracy | Common Usage |
|-------|------------|-------------------|--------------|
| ResNet-8 | ~0.3M | 75-82% | Lightweight experiments |
| ResNet-18 | ~11M | 85-92% | Standard benchmark |
| ResNet-50 | ~25M | 88-94% | MOON, complex experiments |

---

## 6. Master Hyperparameter Comparison Table

### 6.1 Common Settings Across Papers

| Parameter | FedAvg | FedProx | SCAFFOLD | FedNova | pFedMe | MOON |
|-----------|--------|---------|----------|---------|--------|------|
| **Learning Rate** | 0.01-0.1 | 0.01 | 0.01-0.03 | 0.01 | 0.005-0.01 | 0.01 |
| **Batch Size** | 50-64 | 10-64 | 64 | 64 | 20 | 64 |
| **Local Epochs** | 5-20 | 5-20 | 5-20 | 5-10 | 20 | 10 |
| **Rounds** | 100-500 | 200 | 800 | 100-200 | 600-800 | 100 |
| **Clients** | 10-100 | 10 | 10 | 10-16 | 5-20 | 10 |
| **Fraction Fit** | 0.1-1.0 | 0.1-1.0 | **1.0 required** | 0.1-1.0 | 1.0 | 1.0 |

### 6.2 Method-Specific Parameters

| Method | Parameter | Default | Tuning Range |
|--------|-----------|---------|--------------|
| **FedProx** | μ (proximal) | 0.01 | {0.001, 0.01, 0.1, 0.5, 1.0} |
| **SCAFFOLD** | Global LR | 1.0 | - |
| **FedNova** | ρ (momentum) | 0.9 | 0.8-0.95 |
| **pFedMe** | λ (Moreau) | 15.0 | 1.0-30.0 |
| **pFedMe** | K (inner steps) | 5 | 3-10 |
| **pFedMe** | β (server mixing) | 1.0 | 1.0-2.0 |
| **MOON** | μ (contrastive) | 5 | {0.001-10} |
| **MOON** | τ (temperature) | 0.5 | {0.1, 0.5, 1.0} |

---

## 7. Comparison Results on Same Settings

### 7.1 NIID-Bench Comparison (10 clients, 50 rounds, Dir 0.5)

| Method | CIFAR-10 Accuracy |
|--------|-------------------|
| FedAvg | 65.91% |
| FedProx | ~66% |
| SCAFFOLD | ~70% (full participation) |
| FedNova | ~66% |

### 7.2 MOON Comparison (10 clients, 100 rounds, Dir 0.5)

| Method | CIFAR-10 | CIFAR-100 |
|--------|----------|-----------|
| FedAvg | 65.8% | 41.5% |
| FedProx (μ=0.01) | 68.5% | 43.2% |
| **MOON (μ=5)** | **70.7%** | **48.6%** |

### 7.3 Flower Baselines Reproduction (100 rounds)

| Method | CIFAR-10 | CIFAR-100 |
|--------|----------|-----------|
| MOON | 73.53% | 66.36% |
| FedProx | 68.52% | 64.94% |

---

## 8. Accuracy by Non-IID Level

### 8.1 Dirichlet α Comparison

| α | FedAvg | FedProx | SCAFFOLD | MOON |
|---|--------|---------|----------|------|
| IID | 85-87% | 85-87% | 85-87% | 85-87% |
| 1.0 | 78-82% | 78-82% | 80-84% | 80-84% |
| 0.5 | 65-70% | 68-72% | 72-76%* | 70-74% |
| 0.3 | 55-65% | 60-68% | 65-72%* | 65-70% |
| 0.1 | 45-55% | 50-60% | **Unstable** | 55-65% |

*SCAFFOLD requires full client participation

### 8.2 Pathological (#C classes) Comparison

| #C | FedAvg | FedProx | CFL |
|----|--------|---------|-----|
| 1 | 20-35% | 25-40% | 40-55% |
| 2 | 50-60% | 52-62% | 60-70% |
| 3 | 60-70% | 62-72% | 70-78% |

---

## 9. Practical Guidelines

### 9.1 When to Use Each Method

| Scenario | Recommended Method | Settings |
|----------|-------------------|----------|
| **Mild non-IID (α≥0.5)** | FedAvg | lr=0.01, E=5 |
| **Moderate non-IID (0.3≤α<0.5)** | FedProx or MOON | μ=0.01, E=10 |
| **Severe non-IID (α<0.3)** | CFL or pFedMe | Cluster-based |
| **Full participation** | SCAFFOLD | lr=0.03, E=10 |
| **Partial participation** | FedAvg or FedNova | fraction-fit=0.1 |
| **Personalization needed** | pFedMe | λ=15, K=5 |

### 9.2 Common Optimizer Settings

| Parameter | Standard Value |
|-----------|----------------|
| Optimizer | SGD |
| Momentum | 0.9 |
| Weight Decay | 0.0001 |
| LR Scheduler | Step decay or cosine |

### 9.3 Data Augmentation (Standard)

| Augmentation | Effect |
|--------------|--------|
| RandomCrop(32, padding=4) | +2-3% accuracy |
| RandomHorizontalFlip | +1-2% accuracy |
| Normalization | Required |

**Normalization Constants:**
- Mean: (0.4914, 0.4822, 0.4465)
- Std: (0.2023, 0.1994, 0.2010)

---

## 10. Comparison with Your Current HHAR Setup

### 10.1 Current HHAR vs CIFAR-10 Literature

| Parameter | Your HHAR | CIFAR-10 Literature | Match? |
|-----------|-----------|---------------------|--------|
| Clients | 9 | 10-100 | ⚠️ Slightly below |
| Batch Size | 32-64 | 50-64 | ✅ Aligned |
| Local Epochs | 5 | 5-10 | ✅ Aligned |
| Learning Rate | 0.01-0.05 | 0.01-0.1 | ✅ Aligned |
| Rounds | 100 | 50-200 | ✅ Aligned |
| Partitioning | Natural (user-based) | Dirichlet (synthetic) | ℹ️ Different |
| Fraction Fit | 1.0 | 0.1-1.0 | ✅ Aligned |

### 10.2 Recommendations for CIFAR-10 Experiments

If you plan to run CIFAR-10 experiments:

1. **Use NIID-Bench settings as baseline:**
   - 10 clients, batch=64, lr=0.01, E=5, 50+ rounds
   
2. **Standard non-IID: Dirichlet α=0.5**
   - Most comparable to literature
   
3. **For hierarchical experiments:**
   - Consider 3 edge servers × 10 clients = 30 total
   - Or match NIID-Bench: 10 clients flat, then add hierarchy

4. **Model recommendation:**
   - NIID-Bench CNN for direct comparison
   - ResNet-18 for higher accuracy benchmarks

---

## 11. Key References

### Core Algorithms
1. McMahan et al. (2017) - FedAvg, AISTATS
2. Li et al. (2020) - FedProx, MLSys
3. Karimireddy et al. (2020) - SCAFFOLD, ICML
4. Wang et al. (2020) - FedNova, NeurIPS
5. Dinh et al. (2020) - pFedMe, NeurIPS
6. Li et al. (2021) - MOON, CVPR

### Benchmarks
7. Li et al. (2022) - NIID-Bench, ICDE
8. Zhao et al. (2018) - Non-IID analysis, arXiv

### Non-IID Analysis
9. Hsu et al. (2019) - Dirichlet partitioning

---

## 12. Quick Reference Tables

### 12.1 Methods by Type

**Baseline:**
- FedAvg - Weighted averaging

**Regularization-based:**
- FedProx - Proximal term
- FedDyn - Dynamic regularization

**Variance Reduction:**
- SCAFFOLD - Control variates (requires full participation)
- FedNova - Normalized gradients

**Contrastive/Clustering:**
- MOON - Model-contrastive learning
- CFL - Clustered FL

**Personalization:**
- pFedMe - Moreau envelopes
- Per-FedAvg - Meta-learning

### 12.2 Key Numbers to Remember

| Setting | Value |
|---------|-------|
| NIID-Bench clients | 10 |
| NIID-Bench batch | 64 |
| NIID-Bench epochs | 5 |
| NIID-Bench rounds | 50 |
| NIID-Bench LR | 0.01 |
| Standard Dirichlet α | 0.5 |
| FedProx default μ | 0.01 |
| MOON default μ | 5 (CIFAR-10) |
| pFedMe default λ | 15 |

---

*Document prepared for PhD research at London South Bank University*  
*Last updated: January 2026*

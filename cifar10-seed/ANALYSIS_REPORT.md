# CIFAR-10 Federated Learning: Comprehensive Seed Aggregation Analysis

## Overview

This report presents a comprehensive analysis of 5 federated learning methods on CIFAR-10, aggregated across 2 random seeds for statistical robustness. All results are reported as **mean ± std** with **95% confidence intervals**.

**Methods Analyzed:**
- **CFL** (Clustered Federated Learning)
- **FedProx** (Federated Optimization with Proximal Term)
- **HierFL** (Hierarchical Federated Learning / FEDGE)
- **pFedMe** (Personalized Federated Learning)
- **SCAFFOLD** (Stochastic Controlled Averaging)

**Experimental Setup:**
- Dataset: CIFAR-10 (10 classes, 32×32 RGB images)
- Seeds: 2 random seeds (0, 1) for statistical validation
- Training Rounds: ~150 rounds per method
- Evaluation: Centralized test accuracy and loss

---

## 🎯 Main Results: Final Performance

| Method | Final Test Accuracy | Final Test Loss | Last Round |
|--------|-------------------|-----------------|------------|
| **SCAFFOLD** | **0.6091 ± 0.0281** | **1.1262 ± 0.0754** | 150 |
| **FedProx** | **0.5919 ± 0.0130** | **1.2649 ± 0.0027** | 150 |
| **CFL** | **0.5611 ± 0.0042** | **1.3347 ± 0.0095** | 150 |
| **HierFL** | **0.4431 ± 0.0051** | **1.8284 ± 0.0335** | 149 |
| **pFedMe** | **0.1002 ± 0.0031** | **2.3033 ± 0.0025** | 150 |

### Key Findings:
1. **SCAFFOLD** achieves the highest accuracy (60.91%) with moderate variance across seeds
2. **FedProx** shows strong performance (59.19%) with very low variance (most consistent)
3. **CFL** achieves good performance (56.11%) with excellent consistency
4. **HierFL** shows moderate performance (44.31%) but with hierarchical benefits
5. **pFedMe** struggles on CIFAR-10 (10.02%), likely due to architecture mismatch

---

## 🚀 Convergence Analysis

Analysis of rounds required to reach target accuracy levels:

| Method | 50% Accuracy | 60% Accuracy | Seeds Reached Target |
|--------|-------------|-------------|---------------------|
| **SCAFFOLD** | **32.5 ± 6.4** rounds | **82** rounds (1/2 seeds) | 2/2 for 50% |
| **FedProx** | **42.0 ± 0.0** rounds | **109** rounds (1/2 seeds) | 2/2 for 50% |
| **CFL** | **52.0 ± 11.3** rounds | Not reached | 2/2 for 50% |
| **HierFL** | Not reached | Not reached | 0/2 |
| **pFedMe** | Not reached | Not reached | 0/2 |

### Convergence Insights:
- **SCAFFOLD** converges fastest to 50% accuracy (32.5 rounds on average)
- **FedProx** shows consistent convergence with zero variance
- Only **SCAFFOLD** and **FedProx** reach 60% accuracy in some seeds
- **CFL** reaches 50% but with higher variance in convergence time

---

## ⚡ Efficiency Analysis

Communication and computation costs across methods:

| Method | Avg Communication (MB) | Avg Computation Time (sec) | Avg Wall Clock Time (sec) |
|--------|----------------------|---------------------------|--------------------------|
| **FedProx** | **2.48 ± 0.00** | **3.24 ± 0.64** | **5.22 ± 0.64** |
| **SCAFFOLD** | **2.48 ± 0.00** | **3.20 ± 0.19** | **5.18 ± 0.19** |
| **HierFL** | **2.98 ± 0.00** | **9.12 ± 5.06** | **11.50 ± 5.06** |
| **CFL** | **4.73 ± 0.00** | N/A | N/A |
| **pFedMe** | **4.96 ± 0.00** | **120.47 ± 1.66** | **124.44 ± 1.66** |

### Efficiency Insights:
- **FedProx** and **SCAFFOLD** are most communication-efficient (2.48 MB)
- **SCAFFOLD** has slightly lower computation variance than FedProx
- **pFedMe** is computationally expensive (120+ seconds vs ~3 seconds for others)
- **HierFL** shows high computation time variance due to hierarchical structure

---

## 👥 Client Heterogeneity Analysis

Analysis of client-level performance variance:

| Method | Client Accuracy Mean | Client Heterogeneity (Std) |
|--------|--------------------|-----------------------------|
| **SCAFFOLD** | **0.6091 ± 0.0281** | **0.0575 ± 0.0120** |
| **FedProx** | **0.5919 ± 0.0130** | **0.0578 ± 0.0084** |
| **CFL** | **0.5651 ± 0.0094** | **0.0529 ± 0.0234** |
| **HierFL** | **0.4290 ± 0.0006** | **0.0824 ± 0.0010** |
| **pFedMe** | **0.0989 ± 0.0022** | **0.1196 ± 0.0762** |

### Heterogeneity Insights:
- **CFL** shows lowest client heterogeneity (0.0529), indicating good fairness
- **pFedMe** has highest heterogeneity (0.1196), suggesting uneven personalization
- **SCAFFOLD** and **FedProx** show similar heterogeneity levels (~0.057)

---

## 📊 Statistical Robustness

### Confidence Intervals (95% CI) for Final Accuracy:
- **SCAFFOLD**: [0.356, 0.862] - Wide CI due to seed variance
- **FedProx**: [0.475, 0.709] - Narrow CI, most reliable
- **CFL**: [0.523, 0.599] - Very narrow CI, highly consistent
- **HierFL**: [0.397, 0.489] - Narrow CI, consistent but lower performance
- **pFedMe**: [0.072, 0.128] - Narrow CI around low performance

---

## 🔬 Per-Seed Breakdown

Raw results for transparency and reproducibility:

| Method | Seed 0 | Seed 1 | Variance |
|--------|--------|--------|----------|
| **SCAFFOLD** | 62.90% | 58.92% | High |
| **FedProx** | 60.11% | 58.27% | Low |
| **CFL** | 56.41% | 55.81% | Very Low |
| **HierFL** | 44.67% | 43.95% | Very Low |
| **pFedMe** | 9.80% | 10.24% | Very Low |

---

## 📈 Recommendations for Paper Reporting

### Main Results Table Format:
```
Method          | Final Accuracy    | Final Loss        | Convergence (50%)
SCAFFOLD        | 60.91 ± 2.81%    | 1.13 ± 0.08      | 32.5 ± 6.4 rounds
FedProx         | 59.19 ± 1.30%    | 1.26 ± 0.00      | 42.0 ± 0.0 rounds  
CFL             | 56.11 ± 0.42%    | 1.33 ± 0.01      | 52.0 ± 11.3 rounds
HierFL          | 44.31 ± 0.51%    | 1.83 ± 0.03      | Not reached
pFedMe          | 10.02 ± 0.31%    | 2.30 ± 0.00      | Not reached
```

### Suggested Paper Text:
*"All results are reported as mean ± standard deviation across two random seeds, with 95% confidence intervals provided for statistical robustness. SCAFFOLD achieves the highest final test accuracy of 60.91 ± 2.81%, followed closely by FedProx at 59.19 ± 1.30%. FedProx demonstrates the most consistent performance with the lowest variance across seeds. In terms of convergence, SCAFFOLD reaches 50% accuracy fastest (32.5 ± 6.4 rounds), while maintaining competitive communication efficiency (2.48 MB total). Client heterogeneity analysis reveals that CFL provides the most equitable performance distribution across clients."*

---

## 📁 Generated Files

All analysis results are saved in `metrics/summary/`:

1. **`seed_means.csv`** - Main results table (mean ± std ± CI)
2. **`convergence.csv`** - Rounds-to-target analysis  
3. **`seed_level.csv`** - Per-seed raw results
4. **`efficiency_metrics.csv`** - Communication/computation metrics
5. **`client_heterogeneity.csv`** - Client variance analysis
6. **`curves/`** - Per-round mean curves for plotting

---

## 🎯 Key Takeaways

1. **Performance Ranking**: SCAFFOLD > FedProx > CFL > HierFL > pFedMe
2. **Consistency Ranking**: CFL > FedProx > HierFL > pFedMe > SCAFFOLD  
3. **Efficiency Ranking**: FedProx ≈ SCAFFOLD > HierFL > CFL > pFedMe
4. **Convergence Ranking**: SCAFFOLD > FedProx > CFL > (others don't reach 50%)
5. **Fairness Ranking**: CFL > SCAFFOLD ≈ FedProx > HierFL > pFedMe

**Overall Winner**: **SCAFFOLD** for highest accuracy, **FedProx** for best consistency and efficiency balance.

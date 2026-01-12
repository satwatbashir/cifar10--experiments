# Fedge Version History - CIFAR-10

## Results Summary

| Method | Accuracy | Rounds | Rank |
|--------|----------|--------|------|
| **Fedge v2** | **59.16%** | 100 | **1st** |
| FedProx | 56.29% | 200 | 2nd |
| HierFL | 50.58% | 200 | 3rd |
| Fedge v1 | 45.07% | 200 | 4th |

---

## v2 Final Results (seed 42, 100 rounds)

| Metric | Value |
|--------|-------|
| Final accuracy | 59.16% |
| Peak accuracy | 59.68% (round 98) |
| Plateau start | ~round 70-80 |
| num_clusters | 1 (always) |

### v2 vs Baselines

| Baseline | v2 Improvement |
|----------|----------------|
| FedProx (56.29%) | **+2.87%** |
| HierFL (50.58%) | **+8.58%** |
| Fedge v1 (45.07%) | **+14.09%** |

### v2 Accuracy Progression

| Rounds | Accuracy | Gain per 10 rounds |
|--------|----------|-------------------|
| 1→10 | 25% → 42% | +17% |
| 10→20 | 42% → 49% | +7% |
| 20→30 | 49% → 51.5% | +2.5% |
| 30→40 | 51.5% → 54.4% | +2.9% |
| 40→50 | 54.4% → 55.1% | +0.7% |
| 50→60 | 55.1% → 56.2% | +1.1% |
| 60→70 | 56.2% → 58.1% | +1.9% |
| 70→80 | 58.1% → 58.3% | +0.2% ← plateau |
| 80→90 | 58.3% → 58.5% | +0.2% |
| 90→100 | 58.5% → 59.2% | +0.7% |

---

## v1 → v2 Changes

### Configuration Changes

| Parameter | v1 | v2 | Reason |
|-----------|-----|-----|--------|
| `prox_mu` | 0.001 | **0.01** | Match FedProx regularization strength |
| `momentum` | 0.0 | **0.9** | Standard SGD momentum for faster convergence |
| `start_round` | 1 | **30** | Let models differentiate before clustering |

### What Worked in v2

| Change | Impact |
|--------|--------|
| momentum=0.9 | Faster, smoother convergence |
| prox_mu=0.01 | Reduced client drift |

### What Didn't Work in v2

| Issue | Observation |
|-------|-------------|
| Clustering | Still 1 cluster always (tau=0.7 too high) |
| start_round=30 | No effect (similarities still > 0.7) |

---

## v3 Attempt: SCAFFOLD Failed

### What We Tried

| Parameter | v2 | v3 (attempted) |
|-----------|-----|----------------|
| `scaffold_enabled` | false | **true** |
| `tau` | 0.7 | **0.4** |
| `label_smoothing` | 0.1 | **0.0** |

### SCAFFOLD Failure (seed 42)

```
Round 6:  38.97%  ← SCAFFOLD activates after warmup (5 rounds)
Round 7:  21.08%  ← CATASTROPHIC COLLAPSE
Round 8:  18.20%
...
Round 44: 19.48%  ← stuck near random
```

### Root Cause Analysis

1. **SCAFFOLD warmup too short** - Only 5 rounds, not enough for stable control variates
2. **Control variate explosion** - Division by `(local_epochs * lr)` = 0.05 amplifies by 20x
3. **Conflict with FedProx** - Both methods try to correct drift differently
4. **No gradient clipping on SCAFFOLD corrections** - Unbounded corrections

### Lesson Learned

SCAFFOLD needs careful tuning:
- Longer warmup (30+ rounds)
- Scaled-down corrections
- Possibly incompatible with FedProx

---

## v3 Revised Plan (No SCAFFOLD)

### Configuration Changes

| Parameter | v2 | v3 | Reason |
|-----------|-----|-----|--------|
| `scaffold_enabled` | false | **false** | SCAFFOLD caused collapse |
| `tau` | 0.7 | **0.4** | Force multiple clusters |
| `label_smoothing` | 0.1 | **0.0** | Match baselines |

### Expected Outcome

| Metric | v2 | v3 Target |
|--------|-----|-----------|
| avg_accuracy | 59.16% | **60-63%** |
| num_clusters | 1 | >1 after round 30 |

### Expected Gains

| Change | Expected Gain |
|--------|---------------|
| Lower tau (0.4) | +1-2% (if clusters form) |
| Remove label smoothing | +1-2% |
| **Total** | **+2-4%** → **61-63%** |

---

## v2 Issues & Plateau Analysis

### Issue A: Clustering Never Activated
- `num_clusters = 1` for all 100 rounds
- tau=0.7 too high (all server similarities > 0.7)
- No server specialization happening

### Issue B: Label Smoothing Mismatch
- v2 uses `label_smoothing=0.1`
- FedProx baseline uses `label_smoothing=0.0`
- Cost: ~1-2% accuracy

### Issue C: Plateau at ~59%
- Gains dropped to +0.2% per 10 rounds after round 70
- Possible causes:
  1. LeNet capacity ceiling (~62-65% theoretical max)
  2. Fixed learning rate (no decay)
  3. Clustering not helping (single cluster)
  4. Non-IID drift accumulation

---

## Brainstorming: Breaking the Plateau

### Ideas Within Sacred Constraints

| Idea | Expected Gain | Risk | Priority |
|------|---------------|------|----------|
| **Lower tau (0.4→0.3)** | +1-3% | May over-fragment | High |
| **LR decay (cosine)** | +1-2% | Slower early | Medium |
| **Gradient clipping adjustment** | +0.5-1% | May destabilize | Low |
| **Warmup LR schedule** | +0.5-1% | Complexity | Low |

### Ideas Requiring Sacred Parameter Changes

| Idea | Expected Gain | Constraint Violation |
|------|---------------|---------------------|
| ResNet-18 | +10-15% | Model (LeNet) |
| More local epochs | +2-3% | local_epochs=5 |
| Lower alpha_server | +2-4% | alpha_server=0.5 |

### Clustering Improvements

| Idea | Description |
|------|-------------|
| **Gradient-based clustering** | Cluster by gradient direction, not weight similarity |
| **Adaptive tau** | Start high (0.7), decay to 0.3 over rounds |
| **Per-layer clustering** | Cluster based on last FC layer only (already doing this) |
| **Cluster-specific LR** | Different learning rates per cluster |

### Novel Approaches

| Idea | Description | Complexity |
|------|-------------|------------|
| **FedDyn** | Dynamic regularizer instead of FedProx | High |
| **MOON** | Contrastive learning between local/global | High |
| **Personalization layers** | Freeze backbone, personalize last layer per cluster | Medium |

---

## Sacred Parameters (Cannot Change)

| Parameter | Value | Reason |
|-----------|-------|--------|
| `lr_init` | 0.01 | NIID-Bench standard |
| `local_epochs` | 5 | Match baselines |
| `batch_size` | 64 | Match baselines |
| `num_servers` | 3 | Match HierFL |
| `clients_per_server` | [5, 5, 5] | Match HierFL |
| `alpha_server` | 0.5 | Non-IID standard |
| `alpha_client` | 1000.0 | IID within server |
| Model | LeNet | Match all baselines |

---

## File Locations

- Config: `/mnt/d/learn/CIFAR-10/Fedge-Simulation/pyproject.toml`
- Task: `/mnt/d/learn/CIFAR-10/Fedge-Simulation/fedge/task.py`
- Orchestrator: `/mnt/d/learn/CIFAR-10/Fedge-Simulation/orchestrator.py`

## Git History

- v3: (in progress) - tau=0.4, no label smoothing, SCAFFOLD disabled
- v2: commit `b524d42`
- v1: commit `32a5d29` and earlier

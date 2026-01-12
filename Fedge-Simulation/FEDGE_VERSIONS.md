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

## v2 Issues Identified

### Issue A: Clustering Still Ineffective
- `num_clusters = 1` for all 100 rounds
- tau=0.7 still too high (all similarities > 0.7)
- No server specialization happening

### Issue B: Label Smoothing
- v2 uses `label_smoothing=0.1` in task.py
- FedProx baseline uses `label_smoothing=0.0`
- Likely costing 1-2% accuracy

### Issue C: SCAFFOLD Disabled
- `scaffold_enabled=false` in v2
- Could help with client drift correction

### Issue D: Plateau at ~59%
- Gains slowed dramatically after round 70
- Model may be hitting LeNet capacity ceiling

---

## v2 → v3 Changes

### Configuration Changes

| Parameter | v2 | v3 | Reason |
|-----------|-----|-----|--------|
| `scaffold_enabled` | false | **true** | Reduce client drift, +2-4% expected |
| `tau` | 0.7 | **0.4** | Force multiple clusters to form |
| `label_smoothing` | 0.1 | **0.0** | Match baselines, +1-2% expected |

### Sacred Parameters (Unchanged)

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
| `prox_mu` | 0.01 | Working well in v2 |
| `momentum` | 0.9 | Working well in v2 |

---

## v3 Expected Outcome

| Metric | v2 (100 rounds) | v3 Target |
|--------|-----------------|-----------|
| avg_accuracy | 59.16% | **65-70%** |
| num_clusters | 1 | >1 after round 30 |

### Expected Gains

| Change | Expected Gain |
|--------|---------------|
| SCAFFOLD | +2-4% |
| Lower tau (0.4) | +1-3% |
| Remove label smoothing | +1-2% |
| **Total** | **+4-9%** → **63-68%** |

---

## File Locations

- Config: `/mnt/d/learn/CIFAR-10/Fedge-Simulation/pyproject.toml`
- Task: `/mnt/d/learn/CIFAR-10/Fedge-Simulation/fedge/task.py`

## Git History

- v3: (pending)
- v2: commit `b524d42`
- v1: commit `32a5d29` and earlier
